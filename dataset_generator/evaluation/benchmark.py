"""Module 13, Step 3: Benchmark Runner.

Every benchmarked call is a single, unmodified call into an existing
Module 12 agent (`ObserverAgent.generate`, `BASAgent.compute`,
`RewardAgent.compute`, `InterventionAgent.plan`) or the orchestration
graph itself (`build_graph`/`compile_graph`) — this module measures
timing/memory around those calls, it never reimplements what they do.
Reusing the orchestration agents (rather than calling `BASEngine`/
`RewardEngine`/`InterventionPlanner` a second, differently-shaped way)
means there is exactly one code path per engine in the whole project.

Memory is measured with `tracemalloc` (stdlib, cross-platform) rather than
`resource.getrusage` (POSIX-only, unavailable on Windows) or `psutil` (a
new dependency this project doesn't otherwise need). `tracemalloc` tracks
Python-level object allocations; it does not account for memory held in
untracked C-extension buffers (e.g. some NumPy/pandas internals), so
`peak_memory_bytes` here is a lower bound on true peak RSS, not an exact
figure — stated plainly rather than implied to be more precise than it is.
"""

from __future__ import annotations

import time
import tracemalloc
from datetime import datetime, timezone
from typing import Callable, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from dataset_generator.bas.models import BASArtifact
from dataset_generator.intervention.models import InterventionArtifact
from dataset_generator.models.dataset import DatasetArtifact
from dataset_generator.orchestration.agents import (
    BASAgent,
    InterventionAgent,
    ObserverAgent,
    RewardAgent,
)
from dataset_generator.orchestration.graph import build_graph, compile_graph
from dataset_generator.orchestration.state import new_workflow_state
from dataset_generator.reward.models import RewardArtifact

from dataset_generator.evaluation.config import (
    BenchmarkOptions,
    ExperimentConfig,
    compute_experiment_config_fingerprint,
    default_experiment_config,
)

SCHEMA_VERSION = "1.0"

T = TypeVar("T")


class ThroughputMetrics(BaseModel):
    """Rates derived from `mean_runtime_seconds` and whichever counts apply
    to the benchmarked module — `None` for a rate that doesn't apply
    (e.g. `sessions_per_second` for the BAS benchmark, which operates on
    already-generated records, not sessions).
    """

    model_config = ConfigDict(frozen=True)

    rows_per_second: float | None = None
    sessions_per_second: float | None = None
    interactions_per_second: float | None = None
    workflows_per_second: float | None = None


class BenchmarkResult(BaseModel):
    """One module's benchmark: raw per-repetition measurements plus their means.

    Raw lists are kept (not just the mean) so `statistics.py` (Step 6) can
    compute confidence intervals/variance across repetitions without this
    module needing to expose that itself — matching the project's
    convention of storing raw records alongside summary statistics rather
    than only the summary (e.g. `RewardStatistics.reward_distribution`
    alongside individual `RewardRecord`s).
    """

    model_config = ConfigDict(frozen=True)

    module_name: str
    repetitions: int = Field(gt=0)
    warmup_runs: int = Field(ge=0)
    runtime_seconds: list[float]
    cpu_seconds: list[float]
    peak_memory_bytes: list[int]
    mean_runtime_seconds: float
    mean_cpu_seconds: float
    mean_peak_memory_bytes: float
    record_count: int = Field(ge=0)
    throughput: ThroughputMetrics


class BenchmarkArtifact(BaseModel):
    """The single source of truth for one benchmark run's results."""

    model_config = ConfigDict(frozen=True)

    results: list[BenchmarkResult]
    config_fingerprint: str
    schema_version: str
    generation_timestamp: str


def _time_call(fn: Callable[[], T]) -> tuple[float, float, int, T]:
    """Run `fn()` once, returning `(wall_seconds, cpu_seconds, peak_memory_bytes, result)`."""

    tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = fn()
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return wall_seconds, cpu_seconds, peak_memory_bytes, result


class BenchmarkRunner:
    """Benchmarks each of Modules 7/9/10/11/12 independently.

    Every agent is dependency-injectable, defaulting to the same defaults
    each agent class itself already uses (`BASEngine()`, `default_config()`,
    etc.) — consistent with how `orchestration.graph.build_graph` injects
    agents.
    """

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        observer: ObserverAgent | None = None,
        bas_agent: BASAgent | None = None,
        reward_agent: RewardAgent | None = None,
        intervention_agent: InterventionAgent | None = None,
    ) -> None:
        self._config = config or default_experiment_config()
        self._observer = observer or ObserverAgent()
        self._bas_agent = bas_agent or BASAgent()
        self._reward_agent = reward_agent or RewardAgent()
        self._intervention_agent = intervention_agent or InterventionAgent()

    def _options(self) -> BenchmarkOptions:
        return self._config.benchmark_options

    def _run_repeated(
        self, module_name: str, record_count_fn: Callable[[T], int], fn: Callable[[], T]
    ) -> tuple[BenchmarkResult, T]:
        """Shared warmup+repetition loop used by every `benchmark_*` method.

        Warmup calls are discarded (their timing is not measured, and only
        their existence — exercising the code path once — matters); timed
        repetitions are all measured, and the *last* timed repetition's
        result is returned for the caller to chain into the next stage,
        since every module here is deterministic and repetitions with the
        same input are expected to be identical.
        """

        options = self._options()
        for _ in range(options.warmup_runs):
            fn()

        runtimes: list[float] = []
        cpu_times: list[float] = []
        peaks: list[int] = []
        last_result: T | None = None
        for _ in range(options.repetitions):
            wall, cpu, peak, result = _time_call(fn)
            runtimes.append(wall)
            cpu_times.append(cpu)
            peaks.append(peak)
            last_result = result

        assert last_result is not None  # repetitions is validated > 0
        record_count = record_count_fn(last_result)
        mean_runtime = sum(runtimes) / len(runtimes)

        result_model = BenchmarkResult(
            module_name=module_name,
            repetitions=options.repetitions,
            warmup_runs=options.warmup_runs,
            runtime_seconds=runtimes,
            cpu_seconds=cpu_times,
            peak_memory_bytes=peaks,
            mean_runtime_seconds=mean_runtime,
            mean_cpu_seconds=sum(cpu_times) / len(cpu_times),
            mean_peak_memory_bytes=sum(peaks) / len(peaks),
            record_count=record_count,
            throughput=ThroughputMetrics(),  # filled in by the caller, which knows the right rate(s)
        )
        return result_model, last_result

    def benchmark_dataset_generation(
        self, student_count: int, sessions_per_student: int
    ) -> tuple[BenchmarkResult, DatasetArtifact]:
        """Benchmark `ObserverAgent.generate` (Modules 1-7)."""

        result, artifact = self._run_repeated(
            "dataset_generation",
            lambda a: len(a.records),
            lambda: self._observer.generate(student_count=student_count, sessions_per_student=sessions_per_student),
        )
        session_count = len({r.session_id for r in artifact.records})
        throughput = ThroughputMetrics(
            rows_per_second=result.record_count / result.mean_runtime_seconds,
            sessions_per_second=session_count / result.mean_runtime_seconds,
            interactions_per_second=result.record_count / result.mean_runtime_seconds,
        )
        return result.model_copy(update={"throughput": throughput}), artifact

    def benchmark_bas(self, dataset_artifact: DatasetArtifact) -> tuple[BenchmarkResult, BASArtifact]:
        """Benchmark `BASAgent.compute` (Module 9)."""

        result, artifact = self._run_repeated(
            "bas", lambda a: len(a.records), lambda: self._bas_agent.compute(dataset_artifact)
        )
        throughput = ThroughputMetrics(
            rows_per_second=result.record_count / result.mean_runtime_seconds,
            interactions_per_second=result.record_count / result.mean_runtime_seconds,
        )
        return result.model_copy(update={"throughput": throughput}), artifact

    def benchmark_reward(
        self, dataset_artifact: DatasetArtifact, bas_artifact: BASArtifact
    ) -> tuple[BenchmarkResult, RewardArtifact]:
        """Benchmark `RewardAgent.compute` (Module 10)."""

        result, artifact = self._run_repeated(
            "reward",
            lambda a: len(a.records),
            lambda: self._reward_agent.compute(dataset_artifact, bas_artifact),
        )
        throughput = ThroughputMetrics(
            rows_per_second=result.record_count / result.mean_runtime_seconds,
            interactions_per_second=result.record_count / result.mean_runtime_seconds,
        )
        return result.model_copy(update={"throughput": throughput}), artifact

    def benchmark_intervention(
        self, dataset_artifact: DatasetArtifact, bas_artifact: BASArtifact, reward_artifact: RewardArtifact
    ) -> tuple[BenchmarkResult, InterventionArtifact]:
        """Benchmark `InterventionAgent.plan` (Module 11)."""

        result, artifact = self._run_repeated(
            "intervention",
            lambda a: len(a.decisions),
            lambda: self._intervention_agent.plan(dataset_artifact, bas_artifact, reward_artifact),
        )
        throughput = ThroughputMetrics(
            rows_per_second=result.record_count / result.mean_runtime_seconds,
            interactions_per_second=result.record_count / result.mean_runtime_seconds,
        )
        return result.model_copy(update={"throughput": throughput}), artifact

    def benchmark_workflow(self, student_count: int, sessions_per_student: int) -> BenchmarkResult:
        """Benchmark one full orchestration run (Module 12): batch phase +
        per-interaction walk, via `build_graph`/`compile_graph` exactly as
        `orchestration/graph.py` itself is used everywhere else.
        """

        def run_once() -> dict:
            graph = build_graph(student_count=student_count, sessions_per_student=sessions_per_student)
            compiled = compile_graph(graph)
            return compiled.invoke(new_workflow_state())

        result, state = self._run_repeated(
            "workflow", lambda s: len(s.get("tutor_actions", [])), run_once
        )
        throughput = ThroughputMetrics(
            interactions_per_second=result.record_count / result.mean_runtime_seconds,
            workflows_per_second=1.0 / result.mean_runtime_seconds,
        )
        return result.model_copy(update={"throughput": throughput})

    def run_all(self, student_count: int, sessions_per_student: int) -> BenchmarkArtifact:
        """Benchmark Modules 7/9/10/11 (chained, so reward sees the same BAS
        artifact intervention sees, etc.) plus Module 12 as a separate,
        whole-workflow measurement.
        """

        dataset_result, dataset_artifact = self.benchmark_dataset_generation(student_count, sessions_per_student)
        bas_result, bas_artifact = self.benchmark_bas(dataset_artifact)
        reward_result, reward_artifact = self.benchmark_reward(dataset_artifact, bas_artifact)
        intervention_result, _ = self.benchmark_intervention(dataset_artifact, bas_artifact, reward_artifact)
        workflow_result = self.benchmark_workflow(student_count, sessions_per_student)

        return BenchmarkArtifact(
            results=[dataset_result, bas_result, reward_result, intervention_result, workflow_result],
            config_fingerprint=compute_experiment_config_fingerprint(self._config),
            schema_version=SCHEMA_VERSION,
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )
