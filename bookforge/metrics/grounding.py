"""Grounding: is every claim traceable to something actually said?

The ADK pipeline's only defence against hallucination was a sentence in the
prompt ("prefer omission over invention"). Nothing verified it, so a fabricated
benchmark number would sail through the critic, get rendered as a matplotlib
chart by the asset stage, and land in the printed book looking authoritative.

These functions turn that instruction into a measurement. They run in two
places: as a runtime guard that rejects and resamples an ungrounded analysis,
and as an optimizer metric so GEPA can *learn* instructions that produce
grounded extractions.

The check is deliberately conservative — it looks for evidence a number was
spoken, and treats absence of evidence as a flag for review rather than proof
of fabrication. Precision matters more than recall here: a noisy metric teaches
an optimizer the wrong lesson.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bookforge.schemas import (
    ChapterAnalysis,
    extract_numbers,
    normalize_number,
)

# Spoken numbers appear as words in auto-captions ("twenty five percent"),
# so a digit-only search under-reports grounding badly.
_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000,
    "billion": 1_000_000_000,
}

_SCALE_SUFFIXES = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "bn": 1_000_000_000}

_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "out", "she", "so", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "to", "too", "us", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your"]
)


@dataclass
class GroundingReport:
    """Which asserted facts could be traced back to the transcript."""

    checked: int = 0
    grounded: int = 0
    ungrounded: list[str] = field(default_factory=list)
    #: Concepts whose vocabulary barely overlaps the source at all.
    unsupported_concepts: list[str] = field(default_factory=list)
    #: Mean share of each concept's vocabulary that appears in the source.
    lexical_overlap: float = 1.0
    concepts_checked: int = 0

    @property
    def numeric_score(self) -> float:
        """Fraction of verifiable numbers found in the source."""
        if self.checked == 0:
            return 1.0  # asserting nothing is not a failure
        return self.grounded / self.checked

    @property
    def score(self) -> float:
        """Composite 0..1 grounding score.

        Numeric verification carries the score; lexical overlap enters only as
        a penalty for concepts that fall *below the flagging threshold*. Using
        raw overlap as a continuous term would punish good writing — a textbook
        chapter is supposed to introduce vocabulary the speaker never used, and
        a metric that rewards parroting teaches an optimizer to do exactly that.
        """
        if self.concepts_checked:
            flagged = len(self.unsupported_concepts) / self.concepts_checked
        else:
            flagged = 0.0
        return round(max(0.0, self.numeric_score * (1.0 - 0.5 * flagged)), 4)

    def feedback(self) -> str:
        """Actionable text for the refiner and for GEPA's reflection LM."""
        if not self.ungrounded and not self.unsupported_concepts:
            return "All numeric claims and concepts trace back to the transcript."
        parts: list[str] = []
        if self.ungrounded:
            shown = ", ".join(self.ungrounded[:8])
            parts.append(
                f"{len(self.ungrounded)} of {self.checked} numbers do not appear in "
                f"the transcript: {shown}. Remove them or replace them with the "
                f"figures the speaker actually gave. Never estimate a number to "
                f"fill a chart."
            )
        if self.unsupported_concepts:
            shown = "; ".join(self.unsupported_concepts[:4])
            parts.append(
                f"These concepts use vocabulary largely absent from the transcript, "
                f"suggesting they were invented rather than extracted: {shown}."
            )
        return " ".join(parts)


def _numeric_vocabulary(text: str) -> set[str]:
    """Every number the transcript could be said to contain, normalized."""
    vocabulary = {normalize_number(n) for n in extract_numbers(text)}

    lowered = text.lower()
    vocabulary |= _contextual_word_numbers(lowered)
    vocabulary |= _compound_word_numbers(lowered)

    # "1.5k", "3M", "2bn" are spoken as their expanded value.
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(k|m|b|bn)\b", lowered):
        try:
            scaled = float(match.group(1)) * _SCALE_SUFFIXES[match.group(2)]
        except (ValueError, KeyError):
            continue
        vocabulary.add(normalize_number(str(scaled)))
    return vocabulary


#: Units / quantifiers that make a spoken number-word count as evidence.
#: Bare "one approach" / "hundreds of ideas" must not invent a vocabulary hit.
_WORD_NUMBER_UNIT = (
    r"(?:%|percent(?:age)?|pct|nodes?|ms|sec(?:onds?)?|minutes?|hours?|"
    r"requests?|qps|rps|bytes?|kib|mib|gib|gb|mb|kb|bits?|epochs?|steps?|"
    r"layers?|gpus?|cpus?|tokens?|parameters?|times?|fold|x)\b"
)

_WORD_NUMBER_CONTEXT_RE = re.compile(
    rf"\b({'|'.join(re.escape(w) for w in _WORD_NUMBERS)})\b"
    rf"(?:\s*[-–]?\s*{_WORD_NUMBER_UNIT}|\s+{_WORD_NUMBER_UNIT})"
)


def _contextual_word_numbers(lowered: str) -> set[str]:
    """Word numbers only when tied to a unit/quantifier (not bare English)."""
    found: set[str] = set()
    for match in _WORD_NUMBER_CONTEXT_RE.finditer(lowered):
        found.add(str(_WORD_NUMBERS[match.group(1)]))
    return found


_TENS = {k: v for k, v in _WORD_NUMBERS.items() if v >= 20 and v < 100}
_UNITS = {k: v for k, v in _WORD_NUMBERS.items() if 1 <= v <= 9}
_MULTIPLIERS = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}

_COMPOUND_TENS_RE = re.compile(
    rf"\b({'|'.join(_TENS)})[\s-]+({'|'.join(_UNITS)})\b"
)
_MULTIPLIER_RE = re.compile(
    rf"\b({'|'.join(w for w in _WORD_NUMBERS if _WORD_NUMBERS[w] < 100)}|\d+(?:\.\d+)?)"
    rf"[\s-]+({'|'.join(_MULTIPLIERS)})\b"
)


def _compound_word_numbers(lowered: str) -> set[str]:
    """Spoken compounds: 'twenty five' -> 25, 'three hundred' -> 300.

    Auto-captions spell numbers out, so without this a transcript saying
    "twenty five nodes" would make an analysis saying "25 nodes" look invented.
    """
    found: set[str] = set()
    for match in _COMPOUND_TENS_RE.finditer(lowered):
        found.add(str(_TENS[match.group(1)] + _UNITS[match.group(2)]))
    for match in _MULTIPLIER_RE.finditer(lowered):
        head = match.group(1)
        base = _WORD_NUMBERS.get(head)
        if base is None:
            try:
                base = float(head)
            except ValueError:
                continue
        found.add(normalize_number(str(base * _MULTIPLIERS[match.group(2)])))
    return found


_PERCENT_CUE_RE = re.compile(r"%|\bpercent(?:age)?\b|\bpct\b")


def _mentions_percent(transcript_lower: str) -> bool:
    return bool(_PERCENT_CUE_RE.search(transcript_lower))


def _is_grounded(number: str, vocabulary: set[str], transcript_lower: str) -> bool:
    """Is this asserted number defensible against the transcript?"""
    normalized = normalize_number(number)
    if normalized in vocabulary:
        return True
    # A percentage stated as "0.25" and spoken as "25 percent", or vice versa —
    # but only when the transcript actually discusses percentages. Otherwise
    # "2" and "200" would falsely equate via ×100.
    try:
        value = float(normalized)
    except ValueError:
        return normalized.lower() in transcript_lower
    if _mentions_percent(transcript_lower):
        for alternate in (value * 100, value / 100):
            if normalize_number(str(alternate)) in vocabulary:
                return True
    # Integers spoken with a decimal ("5" vs "5.0") or lightly rounded.
    return value.is_integer() and normalize_number(f"{value:.1f}") in vocabulary


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9\-]{2,}", str(text).lower())
    return {w for w in words if w not in _STOPWORDS}


def check_grounding(
    analysis: ChapterAnalysis,
    transcript: str,
    min_concept_overlap: float = 0.15,
) -> GroundingReport:
    """Verify an analysis against the transcript it claims to summarize."""
    report = GroundingReport()
    transcript_lower = str(transcript or "").lower()
    if not transcript_lower.strip():
        return report

    vocabulary = _numeric_vocabulary(transcript)
    transcript_words = _content_words(transcript)

    seen: set[str] = set()
    for number in analysis.all_numbers():
        normalized = normalize_number(number)
        if normalized in seen:
            continue
        seen.add(normalized)
        report.checked += 1
        if _is_grounded(number, vocabulary, transcript_lower):
            report.grounded += 1
        else:
            report.ungrounded.append(number)

    # Lexical support: a concept written from the model's own knowledge rather
    # than from the talk shows up as vocabulary the speaker never used.
    #
    # Two conditions must hold before flagging, because a good textbook
    # explanation is *supposed* to introduce vocabulary the speaker did not use.
    # Penalising elaboration would teach an optimizer to parrot the transcript,
    # which is the opposite of the goal. So a concept is only called unsupported
    # when its body barely overlaps AND its own heading is absent from the talk
    # — the signature of an off-topic invention rather than an expansion.
    overlaps: list[float] = []
    for concept in analysis.concepts:
        words = _content_words(f"{concept.heading} {concept.explanation}")
        if not words:
            continue
        overlap = len(words & transcript_words) / len(words)
        overlaps.append(overlap)
        heading_words = _content_words(concept.heading)
        heading_is_on_topic = bool(heading_words & transcript_words)
        if overlap < min_concept_overlap and not heading_is_on_topic:
            report.unsupported_concepts.append(
                f"'{concept.heading}' ({overlap:.0%} of its terms appear in the talk, "
                "and its heading is never mentioned)"
            )
    report.concepts_checked = len(overlaps)
    report.lexical_overlap = round(sum(overlaps) / len(overlaps), 4) if overlaps else 1.0
    return report


def check_chapter_grounding(
    chapter_tex: str,
    analysis: ChapterAnalysis,
    extra_allowed_text: str = "",
) -> GroundingReport:
    """Verify the written chapter against its analysis.

    Separate concern from ``check_grounding``: the analysis is checked against
    the transcript, and the chapter is checked against the analysis. A number
    that appears in the prose but nowhere in the analysis was invented by the
    writer, which is exactly the failure the original critic was asked to catch
    by eye.

    ``extra_allowed_text`` carries content the chapter may legitimately quote
    that did not come from the analysis — in practice the asset manifest's
    captions, which the pipeline generates itself and which routinely contain
    timestamps ("the slide shown at 90 seconds"). Without this the writer is
    accused of inventing numbers that BookForge handed it.
    """
    from bookforge.tools.latex import strip_asset_declarations

    report = GroundingReport()
    allowed = {normalize_number(n) for n in analysis.all_numbers()}
    # Exercise answers are the analyst's own worked arithmetic (e.g. a storage
    # estimate), handed to the writer to reproduce. They are not spoken claims,
    # so they stay out of the transcript check, but the chapter may quote them.
    analysis_text = analysis.number_sources() + [
        f"{e.question} {e.answer}" for e in analysis.exercises
    ]
    allowed |= _lenient_numbers(" ".join(analysis_text))
    allowed |= {normalize_number(n) for n in extract_numbers(extra_allowed_text)}
    # Numbers the chapter legitimately owns: labels, widths, chapter numbers,
    # and BookForge-owned asset declarations (paths like
    # chapters/01_01-27-13-25-40/figures/frame_01_2s.jpg must not count as
    # numeric claims — they made every local-upload chapter fail QA).
    structural = strip_asset_declarations(chapter_tex)
    structural = re.sub(
        r"\\(?:label|ref|autoref|cref|includegraphics|input|begin|end|chapter|"
        r"bfFigure(?:Here|Pair)?|bfTable|bfDiagram|bfsection|section|subsection)"
        r"(?:\[[^\]]*\])?(?:\{[^}]*\})*",
        " ",
        structural,
    )
    structural = re.sub(
        r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", structural, flags=re.DOTALL
    )
    structural = re.sub(r"0\.\d+\\textwidth", " ", structural)
    structural = re.sub(r"chapters/[^\s{}]+", " ", structural)
    structural = re.sub(r"frame_\d+_\d+s\.(?:jpg|png|pdf)", " ", structural, flags=re.I)
    # LaTeX thousands separators: 44{,}000 and 44\,000 are one number, not 44
    # and 000.
    structural = re.sub(r"(?<=\d)(?:\{,\}|\\,)(?=\d{3})", ",", structural)
    # The base of scientific notation (0.8\times 10^{9}) is not a claim; the
    # mantissa and a multi-digit exponent still are.
    structural = re.sub(r"(?<![\d.])10\s*\^", "^", structural)
    structural = _unglue_latex_commands(structural)

    seen: set[str] = set()
    for number in extract_numbers(structural):
        normalized = normalize_number(number)
        if normalized in seen:
            continue
        seen.add(normalized)
        report.checked += 1
        if normalized in allowed:
            report.grounded += 1
        else:
            report.ungrounded.append(number)

    analysis_words = _content_words(
        " ".join(
            [analysis.summary]
            + [c.explanation for c in analysis.concepts]
            + [c.heading for c in analysis.concepts]
        )
    )
    chapter_words = _content_words(re.sub(r"\\[a-zA-Z@]+", " ", chapter_tex))
    if chapter_words and analysis_words:
        report.lexical_overlap = round(
            len(chapter_words & analysis_words) / len(chapter_words), 4
        )
    return report


def _lenient_numbers(text: str) -> set[str]:
    """Numbers in a blob, including ones glued to a unit ("200ms", "44k").

    ``extract_numbers`` skips those on purpose, but on the *allowed* side that
    made "p95 < 200ms" in the analysis fail to license "200 ms" in the chapter.
    """
    spaced = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", _unglue_latex_commands(str(text or "")))
    return {normalize_number(n) for n in extract_numbers(spaced)}


def _unglue_latex_commands(text: str) -> str:
    r"""``\times10`` -> ``\times 10``: a control word ends before a digit.

    ``extract_numbers`` skips digits glued to letters (so "p95" is not 95),
    which also hid every number written straight after a LaTeX command.
    """
    return re.sub(r"(\\[A-Za-z]+)(?=\d)", r"\1 ", text)


def numbers_in(text: str) -> set[str]:
    """Normalized numeric literals in a blob — shared by tests and diagnostics."""
    return {normalize_number(n) for n in extract_numbers(text)}
