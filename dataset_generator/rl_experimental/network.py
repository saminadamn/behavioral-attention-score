"""A recurrent Q-network: a single-layer LSTM encoder over a short window
of past states, feeding a linear head that outputs Q-values — the DRQN
architecture (Hausknecht & Stone, 2015), implemented directly in NumPy
since this repository has no PyTorch/TensorFlow dependency (see
`requirements.txt`). Forward pass, full backpropagation-through-time
(BPTT), and the Adam update below are the real computation an LSTM
needs; there is no autograd framework doing this underneath.

Why a recurrent encoder at all: `InterventionObservation` is a snapshot
of one interaction. Whether a student's low correctness right now means
"just started struggling" or "third bad interaction in a row" depends on
the trend, not the snapshot — exactly what `observation.bas_trend`/
`reward_trend` already approximate with two numbers, and what feeding a
short sequence through an LSTM lets the network learn to weigh for
itself instead.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class RecurrentQNetwork:
    """sequence of states (batch, seq_len, state_dim) -> Q-values (batch, action_dim).

    One LSTM layer (combined gate weight matrices, each
    `(state_dim + hidden_dim, hidden_dim)`) followed by a linear head
    `(hidden_dim, action_dim)`. `h_0 = c_0 = 0` (not learned — the window
    is short and fixed-length, so there is nothing upstream of frame 0
    to carry state from).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, seed: int) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        rng = np.random.default_rng(seed)
        concat_dim = state_dim + hidden_dim
        scale = np.sqrt(2.0 / concat_dim)

        self.gate_names = ("i", "f", "g", "o")
        self.W: dict[str, np.ndarray] = {
            name: rng.normal(0.0, scale, size=(concat_dim, hidden_dim)) for name in self.gate_names
        }
        self.b: dict[str, np.ndarray] = {name: np.zeros(hidden_dim) for name in self.gate_names}
        # Forget-gate bias initialized to 1.0 (Jozefowicz et al., 2015) so the
        # network starts by remembering, not forgetting, before it learns.
        self.b["f"] = np.ones(hidden_dim)

        self.w_out = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, action_dim))
        self.b_out = np.zeros(action_dim)

        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._t = 0
        for name, param in self._params().items():
            self._m[name] = np.zeros_like(param)
            self._v[name] = np.zeros_like(param)

    def _params(self) -> dict[str, np.ndarray]:
        params = {f"W_{g}": self.W[g] for g in self.gate_names}
        params.update({f"b_{g}": self.b[g] for g in self.gate_names})
        params["w_out"] = self.w_out
        params["b_out"] = self.b_out
        return params

    def forward(self, sequences: np.ndarray) -> tuple[np.ndarray, dict]:
        """`sequences`: (batch, seq_len, state_dim). Returns (Q-values, cache)."""

        batch_size, seq_len, _ = sequences.shape
        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))

        steps = []
        for t in range(seq_len):
            x_t = sequences[:, t, :]
            concat = np.concatenate([x_t, h], axis=1)

            i_t = _sigmoid(concat @ self.W["i"] + self.b["i"])
            f_t = _sigmoid(concat @ self.W["f"] + self.b["f"])
            g_t = np.tanh(concat @ self.W["g"] + self.b["g"])
            o_t = _sigmoid(concat @ self.W["o"] + self.b["o"])

            c_prev = c
            c = f_t * c_prev + i_t * g_t
            h = o_t * np.tanh(c)

            steps.append(
                {"concat": concat, "c_prev": c_prev, "i": i_t, "f": f_t, "g": g_t, "o": o_t, "c": c}
            )

        q = h @ self.w_out + self.b_out
        cache = {"steps": steps, "h_final": h, "seq_len": seq_len}
        return q, cache

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        q, _ = self.forward(sequences)
        return q

    def apply_output_gradient(
        self, cache: dict, grad_q: np.ndarray, learning_rate: float
    ) -> None:
        """BPTT + Adam update given `grad_q = dL/d(Q-values)` (batch, action_dim).

        This is the one backward pass every training algorithm in this
        package (vanilla/Double DQN, CQL, IQL's value regression, BCQ's
        behavior-cloning head) shares — only the formula for `grad_q`
        differs per algorithm (plain TD-error gradient, TD-error plus a
        conservative penalty term, expectile-weighted regression, or a
        softmax cross-entropy gradient). Keeping one BPTT implementation
        means a bug fixed here is fixed for every algorithm at once,
        instead of four near-identical copies drifting apart.
        """

        steps = cache["steps"]
        h_final = cache["h_final"]

        grad_w_out = h_final.T @ grad_q
        grad_b_out = grad_q.sum(axis=0)
        dh_next = grad_q @ self.w_out.T
        dc_next = np.zeros_like(dh_next)

        grads = {f"W_{g}": np.zeros_like(self.W[g]) for g in self.gate_names}
        grads.update({f"b_{g}": np.zeros_like(self.b[g]) for g in self.gate_names})

        for step in reversed(steps):
            c_prev, i_t, f_t, g_t, o_t, c_t = (
                step["c_prev"], step["i"], step["f"], step["g"], step["o"], step["c"],
            )
            tanh_c = np.tanh(c_t)

            dh = dh_next
            do_raw = dh * tanh_c * (o_t * (1.0 - o_t))
            dc = dc_next + dh * o_t * (1.0 - tanh_c**2)

            df_raw = (dc * c_prev) * (f_t * (1.0 - f_t))
            di_raw = (dc * g_t) * (i_t * (1.0 - i_t))
            dg_raw = (dc * i_t) * (1.0 - g_t**2)

            concat = step["concat"]
            grads["W_i"] += concat.T @ di_raw
            grads["W_f"] += concat.T @ df_raw
            grads["W_g"] += concat.T @ dg_raw
            grads["W_o"] += concat.T @ do_raw
            grads["b_i"] += di_raw.sum(axis=0)
            grads["b_f"] += df_raw.sum(axis=0)
            grads["b_g"] += dg_raw.sum(axis=0)
            grads["b_o"] += do_raw.sum(axis=0)

            dconcat = (
                di_raw @ self.W["i"].T + df_raw @ self.W["f"].T
                + dg_raw @ self.W["g"].T + do_raw @ self.W["o"].T
            )
            dh_next = dconcat[:, self.state_dim :]
            dc_next = dc * f_t

        grads["w_out"] = grad_w_out
        grads["b_out"] = grad_b_out

        self._adam_update(grads, learning_rate)

    def train_step(
        self,
        sequences: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """One gradient step of (importance-weighted) MSE(Q(s,a), target).

        Returns `(loss, td_errors)` — `td_errors` (predicted - target, one
        per sample, unweighted) are what Prioritized Experience Replay
        uses to re-rank the transitions just trained on.
        """

        batch_size = sequences.shape[0]
        weights = np.ones(batch_size) if sample_weights is None else sample_weights

        q, cache = self.forward(sequences)
        predicted = q[np.arange(batch_size), actions]
        td_errors = predicted - targets
        loss = float(np.mean(weights * td_errors**2))

        grad_q = np.zeros_like(q)
        grad_q[np.arange(batch_size), actions] = 2.0 * weights * td_errors / batch_size

        self.apply_output_gradient(cache, grad_q, learning_rate)
        return loss, td_errors

    def _adam_update(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        params = self._params()
        for name, grad in grads.items():
            self._m[name] = beta1 * self._m[name] + (1 - beta1) * grad
            self._v[name] = beta2 * self._v[name] + (1 - beta2) * (grad**2)
            m_hat = self._m[name] / (1 - beta1**self._t)
            v_hat = self._v[name] / (1 - beta2**self._t)
            params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def copy_weights_from(self, other: "RecurrentQNetwork") -> None:
        for gate in self.gate_names:
            self.W[gate] = other.W[gate].copy()
            self.b[gate] = other.b[gate].copy()
        self.w_out = other.w_out.copy()
        self.b_out = other.b_out.copy()
