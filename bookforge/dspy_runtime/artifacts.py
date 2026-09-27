"""Persistence for optimized DSPy programs.

An optimizer run is expensive and its output is just data — rewritten
instructions and selected demonstrations. Saving that data and loading it at
startup is what turns "we ran GEPA once in a notebook" into "production uses
optimized prompts". Artifacts are versioned by content hash and carry the
metadata needed to know whether they are still trustworthy: which optimizer
produced them, against which signature, at what score.

A stale artifact is worse than none — instructions optimized for an older
signature can reference fields that no longer exist. Loading therefore verifies
the signature fingerprint and refuses a mismatch rather than half-applying it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dspy

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 1


@dataclass
class ArtifactMetadata:
    """Provenance for one compiled program."""

    name: str
    optimizer: str
    created_at: str
    signature_fingerprint: str
    train_score: float = 0.0
    val_score: float = 0.0
    baseline_score: float = 0.0
    trainset_size: int = 0
    valset_size: int = 0
    notes: str = ""
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        return round(self.val_score - self.baseline_score, 4)


def signature_fingerprint(program: dspy.Module) -> str:
    """Stable hash of a program's predictor signatures.

    Changes when a field is added, removed, renamed or retyped — exactly the
    circumstances under which old instructions stop being valid.
    """
    parts: list[str] = []
    for name, predictor in sorted(program.named_predictors(), key=lambda kv: kv[0]):
        signature = getattr(predictor, "signature", None)
        if signature is None:
            continue
        fields = sorted(
            f"{field_name}:{getattr(info.annotation, '__name__', info.annotation)}"
            for field_name, info in signature.model_fields.items()
        )
        parts.append(f"{name}|{signature.__name__}|{','.join(fields)}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def save_program(
    program: dspy.Module,
    directory: Path,
    name: str,
    optimizer: str,
    train_score: float = 0.0,
    val_score: float = 0.0,
    baseline_score: float = 0.0,
    trainset_size: int = 0,
    valset_size: int = 0,
    notes: str = "",
    **extra: Any,
) -> Path:
    """Write ``<directory>/<name>.json`` plus ``<name>.meta.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    program_path = directory / f"{name}.json"
    meta_path = directory / f"{name}.meta.json"

    program.save(str(program_path), save_program=False)

    metadata = ArtifactMetadata(
        name=name,
        optimizer=optimizer,
        created_at=datetime.now(timezone.utc).isoformat(),
        signature_fingerprint=signature_fingerprint(program),
        train_score=round(train_score, 4),
        val_score=round(val_score, 4),
        baseline_score=round(baseline_score, 4),
        trainset_size=trainset_size,
        valset_size=valset_size,
        notes=notes,
        extra=extra,
    )
    meta_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    logger.info(
        "saved optimized program '%s' (%s): val %.3f vs baseline %.3f (%+.3f)",
        name, optimizer, metadata.val_score, metadata.baseline_score,
        metadata.improvement,
    )
    return program_path


def load_metadata(directory: Path, name: str) -> ArtifactMetadata | None:
    meta_path = Path(directory) / f"{name}.meta.json"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        payload.pop("schema_version", None)
        return ArtifactMetadata(schema_version=ARTIFACT_SCHEMA_VERSION, **payload)
    except Exception as exc:
        logger.warning("unreadable artifact metadata at %s: %s", meta_path, exc)
        return None


def load_program(
    program: dspy.Module,
    directory: Path,
    name: str,
    strict: bool = False,
) -> bool:
    """Load optimized state into ``program`` in place.

    Returns True when state was applied. A missing artifact is normal (the
    system runs on its authored instructions); a *mismatched* artifact is a
    problem and is refused, because partially applying instructions written for
    a different signature produces prompts that reference fields that no longer
    exist.
    """
    directory = Path(directory)
    program_path = directory / f"{name}.json"
    if not program_path.exists():
        logger.debug("no optimized artifact for '%s' at %s", name, program_path)
        return False

    metadata = load_metadata(directory, name)
    current = signature_fingerprint(program)
    if metadata and metadata.signature_fingerprint != current:
        message = (
            f"optimized artifact '{name}' was compiled against signature "
            f"{metadata.signature_fingerprint} but the code is now at {current}. "
            "Re-run optimization; the saved instructions may reference fields "
            "that no longer exist."
        )
        if strict:
            raise RuntimeError(message)
        logger.warning("%s Skipping load.", message)
        return False

    try:
        program.load(str(program_path))
    except Exception as exc:
        if strict:
            raise
        logger.warning("could not load optimized artifact '%s': %s", name, exc)
        return False

    if metadata:
        logger.info(
            "loaded optimized '%s' (%s, val %.3f, %+.3f over baseline)",
            name, metadata.optimizer, metadata.val_score, metadata.improvement,
        )
    else:
        logger.info("loaded optimized '%s' (no metadata)", name)
    return True


def describe_artifacts(directory: Path) -> str:
    """Human-readable inventory of compiled programs, for the CLI."""
    directory = Path(directory)
    if not directory.exists():
        return f"No artifacts directory at {directory}."
    metas = [
        load_metadata(directory, path.stem.removesuffix(".meta"))
        for path in sorted(directory.glob("*.meta.json"))
    ]
    metas = [m for m in metas if m]
    if not metas:
        return f"No optimized programs in {directory}."
    lines = [f"Optimized programs in {directory}:"]
    for meta in metas:
        lines.append(
            f"  {meta.name:<16} {meta.optimizer:<9} "
            f"val={meta.val_score:.3f} baseline={meta.baseline_score:.3f} "
            f"({meta.improvement:+.3f})  sig={meta.signature_fingerprint}  "
            f"{meta.created_at[:19]}"
        )
    return "\n".join(lines)
