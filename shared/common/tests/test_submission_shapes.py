"""APEX-106/APEX-170: canonical submission shapes.

Every submission-bearing response model subclasses SubmissionBase, so the
core names and types cannot drift apart between endpoints. The deprecated
dual-emitted names were dropped in APEX-170; the `assert_no_deprecated_names`
checks below are the regression guard that keeps them off the wire.
"""

from datetime import datetime

from common.models.api.miner_profile import SubmissionHistoryRecord
from common.models.api.pagination import Pagination
from common.models.api.ranks import RanksResponse
from common.models.api.submission import (
    RankRecord,
    SubmissionBase,
    SubmissionDetail,
    SubmissionRecord,
)

CORE = dict(
    id=7,
    competition_id=3,
    round_number=2,
    state="scored",
    hotkey="hk1",
    coldkey="ck1",
    version=1,
    submitted_at=datetime(2026, 8, 1, 12, 0, 0),
    score=0.5,
    raw_score=1.5,
    top_score=True,
)

# Names dropped in APEX-170. None of them may reappear on any payload.
DEPRECATED_NAMES = {
    "submit_at",
    "eval_score",
    "eval_raw_score",
    "top_scorer",
    "submission_date",
    "submission_id",
    "score_render",
    "incentive_weight_render",
}


def assert_no_deprecated_names(dumped: dict) -> None:
    leaked = DEPRECATED_NAMES & dumped.keys()
    assert not leaked, f"deprecated names back on the wire: {sorted(leaked)}"


def test_pagination_shape():
    p = Pagination(start_idx=0, count=10, total=25, has_more=True)
    assert p.model_dump() == {"start_idx": 0, "count": 10, "total": 25, "has_more": True}


def test_submission_base_canonical_fields():
    base = SubmissionBase(**CORE)
    assert base.submitted_at == datetime(2026, 8, 1, 12, 0, 0)
    assert base.score == 0.5
    assert base.raw_score == 1.5
    assert base.top_score is True


def test_submission_record_canonical_shape():
    rec = SubmissionRecord(**CORE, eval_error="boom", eval_time_in_seconds=1.25)
    dumped = rec.model_dump(mode="json")
    assert dumped["submitted_at"] == "2026-08-01T12:00:00"
    assert dumped["score"] == 0.5
    assert dumped["raw_score"] == 1.5
    assert dumped["eval_error"] == "boom"
    assert dumped["eval_time_in_seconds"] == 1.25
    assert_no_deprecated_names(dumped)


def test_submission_detail_carries_core_identity():
    detail = SubmissionDetail(**CORE, rank=4, code_path="a/b.py")
    dumped = detail.model_dump(mode="json")
    # Core fields the FE used to reconstruct with a client-side join.
    assert dumped["hotkey"] == "hk1"
    assert dumped["competition_id"] == 3
    assert dumped["state"] == "scored"
    assert dumped["version"] == 1
    assert dumped["submitted_at"] == "2026-08-01T12:00:00"
    assert dumped["rank"] == 4
    assert dumped["score"] == 0.5
    assert dumped["raw_score"] == 1.5
    assert_no_deprecated_names(dumped)


def test_rank_record_canonical_shape():
    rec = RankRecord(
        **CORE,
        rank=1,
        submissions_count=5,
        join_date=datetime(2026, 7, 1, 0, 0, 0),
    )
    dumped = rec.model_dump(mode="json")
    assert dumped["rank"] == 1
    assert dumped["top_score"] is True
    assert dumped["submitted_at"] == "2026-08-01T12:00:00"
    assert dumped["id"] == 7
    assert dumped["state"] == "scored"
    assert_no_deprecated_names(dumped)


def test_history_record_canonical_shape():
    rec = SubmissionHistoryRecord(**CORE, rank=2)
    dumped = rec.model_dump(mode="json")
    assert dumped["id"] == 7
    assert dumped["rank"] == 2
    assert dumped["submitted_at"] == "2026-08-01T12:00:00"
    assert_no_deprecated_names(dumped)


def test_ranks_response_envelope_shape():
    resp = RanksResponse(
        competition_id=3,
        miners=[],
        pagination=Pagination(start_idx=0, count=0, total=0, has_more=False),
        total_submissions=0,
    )
    dumped = resp.model_dump(mode="json")
    assert dumped["competition_id"] == 3
    assert dumped["total_submissions"] == 0
    assert_no_deprecated_names(dumped)
