"""Response wording, keyed by (attention state, correctness).

Rebuilt as a **compositional** generator rather than a fixed template list.
The original design (~16 pre-expanded strings per cell) produced a 54.85%
duplicate rate at 10,000 responses — several cells included at least one
*fully static* anchor string (e.g. "I don't know.") with no `{keyword}` or
`{topic}` placeholder at all, so every draw of that anchor collided
byte-for-byte with every other draw of it. A first compositional pass
(starters x cores x endings, ~5-10 options per axis) got duplicates down to
28.9% — better, but still well above the <10% target — because Impulsive
responses are short by design and its pools were smallest. This version
substantially widens every pool (roughly 3-5x), especially Impulsive's.

Each (attention state, correctness) cell defines three independent
component pools — `starters`, `cores`, `endings` — assembled by `compose()`.
Capacity is the *product* of the pool sizes, not their sum, which is what
actually fixes duplication at scale; combined with the keyword/topic
substitution varying across prompts, the realized text space is large
enough that duplicate collisions become rare.

`cores` is also where the semantic-similarity fix lives: every Focused core
and every Distracted-correct/Impulsive core references `{keyword}` and often
`{topic}`, while Distracted-incorrect cores mostly do not — reflecting "an
unfocused student's answer genuinely has little to do with the prompt".
"""

from __future__ import annotations

import numpy as np

from dataset_generator.config.attention_state import AttentionState

# (starters, cores, endings) per (attention_state, correct)
_ComponentPool = tuple[list[str], list[str], list[str]]

COMPONENTS: dict[AttentionState, dict[bool, _ComponentPool]] = {
    AttentionState.FOCUSED: {
        True: (
            [
                "", "Well, ", "So, ", "Thinking it through, ", "Here's my answer: ",
                "Let me explain: ", "To answer this, ", "In short, ",
            ],
            [
                "{keyword} is central to understanding {topic}, since it explains the underlying mechanism step by step",
                "in {topic}, {keyword} works by connecting each part logically",
                "the key idea is {keyword}, which directly explains {topic} in a clear way",
                "{keyword} accounts for {topic} because each step follows logically from the last",
                "{topic} makes sense once you see how {keyword} fits into it",
                "the reasoning is that {keyword} drives {topic}, and each part builds on the one before",
                "{keyword} is the piece that ties {topic} together",
                "once you understand {keyword}, {topic} follows naturally",
                "{topic} can be explained through {keyword}, step by step",
            ],
            [
                ".", ", and that's why it works.", ", which ties everything together.",
                ", step by step.", ", if I'm reasoning about it correctly.",
                ", which is the core idea here.",
            ],
        ),
        False: (
            [
                "", "I believe ", "My best attempt: ", "As far as I can tell, ",
                "I think ", "If I recall correctly, ", "Let me try: ",
            ],
            [
                "{keyword} relates to {topic}, though I may be missing one detail",
                "the answer involves {keyword} and {topic}, but I'm not fully certain it's complete",
                "it connects to {keyword}, though I could be wrong about {topic}",
                "{topic} depends on {keyword}, but I'm not sure I have every step right",
                "{keyword} is part of the answer for {topic}, even if I'm missing something",
                "{topic} seems to involve {keyword}, though I might have a detail backwards",
                "I worked through {topic} using {keyword}, but I'm not confident it's fully right",
                "{keyword} should explain {topic}, though I may have made an error somewhere",
            ],
            [
                ".", ", though I'm not fully sure.", ", but please correct me if I'm wrong.",
                ", so let me know if that's off.", ".",
            ],
        ),
    },
    AttentionState.DISTRACTED: {
        True: (
            [
                "I think ", "Maybe ", "I guess ", "Possibly ", "I'm not sure, but ",
                "", "Hmm, ", "I want to say ", "Kind of think ", "Sort of think ",
                "My gut says ", "If I had to guess, ",
            ],
            [
                "it might be {keyword}",
                "it could be {keyword}, related to {topic}",
                "it's probably {keyword}",
                "it has something to do with {keyword} in {topic}",
                "it's {keyword}? I think",
                "it might connect to {topic} through {keyword}",
                "it's {keyword}, or close to it",
                "{keyword} sounds right for {topic}",
                "it's something like {keyword}",
                "it feels like {keyword}",
                "it's probably related to {keyword} and {topic}",
                "maybe it's {keyword}, if I remember right",
                "it's {keyword}? not one hundred percent though",
                "could be {keyword}, from what I recall about {topic}",
                "it's maybe {keyword}, loosely tied to {topic}",
                "I'll go with {keyword}",
                "{keyword}, I think, for {topic}",
            ],
            [
                "", ".", ", I think.", ", maybe.", "?", ", but don't quote me on it.",
                ", but I could be wrong.", ", if that's right.", ", probably.",
                ", though I'm guessing.",
            ],
        ),
        False: (
            [
                "I ", "Honestly, I ", "Sorry, I ", "", "Not sure — I ",
                "Ugh, I ", "I mean, I ", "Uhh, I ", "Actually, I ",
                "My mind wandered — I ", "Kind of — I ", "Look, I ",
                "To be honest, I ", "Well, I ", "Sorry, I think I ",
                "No idea — I ", "Wait, I ", "Ok so, I ",
            ],
            [
                "don't remember exactly",
                "forgot how {keyword} works",
                "lost track of {topic}",
                "am confused about {topic}",
                "don't know the answer",
                "don't know",
                "forgot the steps",
                "am not completely sure",
                "can't remember right now",
                "wasn't really following {topic}",
                "zoned out during {topic}",
                "mixed up {topic} with something else",
                "am blanking on this one",
                "don't really remember {topic}",
                "spaced out during {topic}",
                "totally blanked on {keyword}",
                "can't focus on {topic} right now",
                "missed that part about {topic}",
                "got distracted before finishing {topic}",
                "drew a blank on {keyword}",
                "need you to repeat {topic}",
                "am drawing a blank here",
                "totally spaced on {topic}",
                "can't think straight about {keyword}",
                "keep losing my train of thought on {topic}",
                "am stuck on {topic}",
            ],
            [
                "", ".", ", sorry.", ", can we come back to it?", ", to be honest.",
                ", if you can give me a hint.", ", I zoned out.", ", my bad.",
                ", can you repeat the question?", ", I need a minute.",
                ", it's just not clicking.", ", today.",
            ],
        ),
    },
    AttentionState.IMPULSIVE: {
        True: (
            [
                "", "It's ", "Obviously ", "Quick answer: ", "Easy — ", "Simple: ", "Yep, ",
                "Duh, ", "No question, ", "For sure, ",
            ],
            [
                "{keyword}", "{keyword}, easy", "{keyword}, done", "definitely {keyword}",
                "gotta be {keyword}", "{keyword} for sure", "{keyword}, that's it",
                "{keyword}, obviously", "100% {keyword}", "{keyword}, no doubt",
            ],
            [".", "!", ", next.", ", moving on.", ", easy one.", ", next question."],
        ),
        False: (
            [
                "", "Umm, ", "It's ", "Pretty sure it's ", "Gonna say ", "I'll guess ",
                "Quick guess: ", "Fine, ", "Whatever, ",
            ],
            [
                "{keyword}", "{keyword}? whatever", "{keyword}, I guess", "{keyword}, moving on",
                "{keyword} probably", "{keyword}, next", "{keyword}, who knows",
                "{keyword}, good enough", "{keyword}, close enough",
            ],
            ["!", ", next question.", ".", ", whatever.", ", moving on.", ", who cares."],
        ),
    },
}


def compose(
    rng: np.random.Generator,
    attention_state: AttentionState,
    correct: bool,
    keyword: str,
    topic: str,
) -> str:
    """Assemble one response string for `(attention_state, correct)`.

    Only the fully-assembled string's first character is capitalized —
    every `core` is authored in natural mid-sentence casing (the pronoun
    "I" stays capitalized wherever it appears; everything else is
    lowercase), so this one rule is correct whether or not a `starter`
    precedes it.
    """

    starters, cores, endings = COMPONENTS[attention_state][correct]
    starter = str(rng.choice(starters))
    core = str(rng.choice(cores)).format(keyword=keyword, topic=topic)
    ending = str(rng.choice(endings))

    text = f"{starter}{core}{ending}"
    text = " ".join(text.split())
    if text:
        text = text[0].upper() + text[1:]
    return text
