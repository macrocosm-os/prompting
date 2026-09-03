"""Typed artifact manifest (APEX-105).

Replaces `eval_file_paths` basename conventions: every evaluation artifact is
declared with a role and (where it applies) the envelope unit it belongs to,
so neither the FE nor the download endpoint ever infers meaning from a
file name again. Rows live in `submission_artifact`; legacy submissions get a
manifest synthesized on read from `eval_file_paths` + `submit_metadata.onnx`.
"""

import mimetypes
import posixpath
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

ArtifactRole = Literal["history", "predictions", "truth", "log", "model"]

# The wire's declared `role` (JobFile.role) must be validated against this before it ever
# reaches the DB: an out-of-vocabulary role would make EvaluationArtifact.role (typed
# `ArtifactRole`) raise on read, 500ing SubmissionDetail for that submission.
VALID_ROLES: frozenset[str] = frozenset(get_args(ArtifactRole))

# Legacy FileType -> role. CODE is deliberately absent: code is served by its
# own gated endpoint and never appears in the manifest.
_FILE_TYPE_ROLES: dict[str, ArtifactRole] = {"log": "log", "history": "history", "onnx": "model"}

_CONTENT_TYPES = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".log": "text/plain",
    ".txt": "text/plain",
    ".onnx": "application/octet-stream",
}


class EvaluationArtifact(BaseModel):
    """One declared evaluation artifact, as served on SubmissionDetail."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str  # deterministic: "{role}:{file_name}"
    unit_id: str | None = None  # ties the artifact to a units[].id in the envelope
    role: ArtifactRole
    file_name: str
    s3_key: str
    content_type: str | None = None
    size: int | None = None
    # JSON name is `schema` (FE contract); `schema_ref` avoids shadowing
    # pydantic's deprecated BaseModel.schema classmethod.
    schema_ref: str | None = Field(default=None, alias="schema")


def artifact_id(role: str, file_name: str) -> str:
    return f"{role}:{file_name}"


def infer_role(file_type: str) -> ArtifactRole | None:
    """Role for a legacy FileType-tagged file; None for types with no manifest home."""
    return _FILE_TYPE_ROLES.get(file_type)


def guess_content_type(file_name: str) -> str | None:
    ext = posixpath.splitext(file_name)[1].lower()
    if ext in _CONTENT_TYPES:
        return _CONTENT_TYPES[ext]
    return mimetypes.guess_type(file_name)[0]


def synthesize_legacy_artifacts(eval_file_paths: dict | None, submit_metadata: dict | None) -> list[EvaluationArtifact]:
    """Manifest for a pre-APEX-105 row. `eval_file_paths` values are full S3
    keys; ONNX lives in its parallel `submit_metadata.onnx` mechanism. Unit
    linkage is unknowable here, so `unit_id` stays None."""
    # Duel bracket matches reuse file names, and worker retries can double-append the same
    # key into eval_file_paths; dedupe on the artifact id (first occurrence wins) so a
    # repeated key doesn't synthesize duplicate manifest entries.
    artifacts: list[EvaluationArtifact] = []
    seen: set[str] = set()
    for file_type, keys in (eval_file_paths or {}).items():
        role = infer_role(file_type)
        if role is None or not isinstance(keys, list):
            continue
        for key in keys:
            file_name = posixpath.basename(key)
            aid = artifact_id(role, file_name)
            if aid in seen:
                continue
            seen.add(aid)
            artifacts.append(
                EvaluationArtifact(
                    id=aid,
                    role=role,
                    file_name=file_name,
                    s3_key=key,
                    content_type=guess_content_type(file_name),
                )
            )
    onnx = (submit_metadata or {}).get("onnx") or {}
    if isinstance(onnx, dict) and onnx.get("path"):
        file_name = posixpath.basename(onnx["path"])
        aid = artifact_id("model", file_name)
        if aid not in seen:
            seen.add(aid)
            artifacts.append(
                EvaluationArtifact(
                    id=aid,
                    role="model",
                    file_name=file_name,
                    s3_key=onnx["path"],
                    content_type=guess_content_type(file_name),
                )
            )
    return artifacts


def resolve_unit_id(declared: str | None, stored_eval_metadata: dict | None) -> str | None:
    """Map a runner-declared unit id onto the stored envelope's ids.

    The duel accumulator re-ids game units to "match-N-game-M"; results land
    before files and bracket matches are sequential per submission, so a
    declared "game-M" belongs to the LATEST match unit. Remap only when the
    target id actually exists; otherwise keep the declaration verbatim.
    """
    if not declared:
        return None
    from common.models.api.eval_metadata import coerce_eval_metadata

    meta = coerce_eval_metadata(stored_eval_metadata)
    unit_ids = {u.id for u in meta.units}
    if declared in unit_ids:
        return declared
    matches = [u for u in meta.units if u.type == "match"]
    if not matches:
        return declared
    latest = max(matches, key=lambda u: u.index)
    remapped = f"{latest.id}-{declared}"
    return remapped if remapped in unit_ids else declared
