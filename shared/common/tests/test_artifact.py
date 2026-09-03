"""EvaluationArtifact model + legacy synthesis (APEX-105).

The manifest replaces basename pattern-matching: role and unit linkage are
declared, and legacy rows get a manifest synthesized at the read boundary from
eval_file_paths + submit_metadata.onnx so clients only ever read one field.
"""

from common.models.api.artifact import (
    EvaluationArtifact,
    artifact_id,
    guess_content_type,
    infer_role,
    synthesize_legacy_artifacts,
)


def test_serializes_schema_ref_under_json_name_schema():
    a = EvaluationArtifact(
        id="history:game_1.json",
        role="history",
        file_name="game_1.json",
        s3_key="competition_id=1/round_number=2/hotkey=hk/history/game_1.json",
        schema_ref="gym_v1.history",
    )
    dumped = a.model_dump(by_alias=True)
    assert dumped["schema"] == "gym_v1.history"
    assert "schema_ref" not in dumped
    # And the FE-contract name round-trips back in.
    assert EvaluationArtifact.model_validate(dumped).schema_ref == "gym_v1.history"


def test_artifact_id_is_role_scoped():
    assert artifact_id("history", "game_1.json") == "history:game_1.json"


def test_infer_role_maps_legacy_file_types():
    assert infer_role("log") == "log"
    assert infer_role("history") == "history"
    assert infer_role("onnx") == "model"
    assert infer_role("code") is None
    assert infer_role("mystery") is None


def test_guess_content_type():
    assert guess_content_type("game_1.json") == "application/json"
    assert guess_content_type("trace.jsonl") == "application/x-ndjson"
    assert guess_content_type("sandbox.log") == "text/plain"
    assert guess_content_type("model.onnx") == "application/octet-stream"


def test_synthesize_from_eval_file_paths_and_onnx():
    paths = {
        "log": ["competition_id=8/round_number=40/hotkey=hk/log/a.log"],
        "history": ["competition_id=8/round_number=40/hotkey=hk/history/game_1.json"],
    }
    meta = {"onnx": {"path": "competition_id=8/round_number=40/hotkey=hk/onnx/model.onnx"}}
    arts = synthesize_legacy_artifacts(paths, meta)
    by_id = {a.id: a for a in arts}
    assert by_id["log:a.log"].role == "log"
    assert by_id["log:a.log"].s3_key.endswith("/log/a.log")
    assert by_id["history:game_1.json"].content_type == "application/json"
    assert by_id["model:model.onnx"].role == "model"
    # Synthesized entries never claim unit linkage they don't have.
    assert all(a.unit_id is None for a in arts)


def test_synthesize_skips_unknown_types_and_handles_empty():
    assert synthesize_legacy_artifacts(None, None) == []
    assert synthesize_legacy_artifacts({"code": ["x/y.py"]}, {}) == []


def test_synthesize_dedupes_duplicate_keys():
    # Duel bracket matches reuse file names, and worker retries can double-append the same
    # key into eval_file_paths; a repeated key must yield exactly one manifest entry.
    paths = {
        "history": [
            "competition_id=8/round_number=40/hotkey=hk/history/game_1.json",
            "competition_id=8/round_number=40/hotkey=hk/history/game_1.json",
        ]
    }
    meta = {
        "onnx": {
            "path": "competition_id=8/round_number=40/hotkey=hk/onnx/model.onnx",
        }
    }
    arts = synthesize_legacy_artifacts(paths, meta)
    history_arts = [a for a in arts if a.role == "history"]
    assert len(history_arts) == 1

    # The ONNX entry is covered by the same dedupe: a legacy `eval_file_paths["onnx"]` entry
    # sharing the file name with `submit_metadata.onnx` produces the same artifact id
    # ("model:model.onnx") — only the first occurrence should survive.
    paths_with_onnx = {"onnx": ["competition_id=8/round_number=40/hotkey=hk/onnx/model.onnx"]}
    arts2 = synthesize_legacy_artifacts(paths_with_onnx, meta)
    model_arts = [a for a in arts2 if a.role == "model"]
    assert len(model_arts) == 1


def test_resolve_unit_id_keeps_existing_ids_and_remaps_duel_games():
    from common.models.api.artifact import resolve_unit_id

    duel_meta = {
        "schema_version": 1,
        "units": [
            {"id": "match-1", "type": "match", "index": 1, "label": "Match 1"},
            {"id": "match-1-game-1", "type": "game", "index": 1, "label": "Game 1", "parent_id": "match-1"},
            {"id": "match-2", "type": "match", "index": 2, "label": "Match 2"},
            {"id": "match-2-game-1", "type": "game", "index": 1, "label": "Game 1", "parent_id": "match-2"},
        ],
    }
    # game-1 doesn't exist verbatim -> remapped onto the LATEST match.
    assert resolve_unit_id("game-1", duel_meta) == "match-2-game-1"
    # An id that exists verbatim is kept.
    assert resolve_unit_id("match-1-game-1", duel_meta) == "match-1-game-1"
    # No declaration -> nothing.
    assert resolve_unit_id(None, duel_meta) is None
    # No match units (solo) -> declared id kept verbatim.
    assert resolve_unit_id("game-1", {"schema_version": 1, "units": []}) == "game-1"
    # Remap target that doesn't exist -> declared id kept verbatim.
    assert resolve_unit_id("game-9", duel_meta) == "game-9"
