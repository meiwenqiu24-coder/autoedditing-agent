import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_video_agent.orchestrator.utils import sanitize_timeline


def test_sanitize_removes_gaps_and_overlaps_without_crossfade():
    timeline = [
        {
            "shot_index": 0,
            "clip": {"src": "input.mp4", "in": 0.0, "out": 4.0},
            "transition_in": {"type": "none", "dur": 0.0},
            "transition_out": {"type": "fade", "dur": 1.0},
            "text_overlays": [],
            "t_start": 0.0,
            "t_end": 4.0,
        },
        {
            "shot_index": 1,
            "clip": {"src": "input.mp4", "in": 10.0, "out": 13.0},
            "transition_in": {"type": "fade", "dur": 1.0},
            "transition_out": {"type": "none", "dur": 0.0},
            "text_overlays": [],
            "t_start": 2.0,
            "t_end": 5.0,
        },
        {
            "shot_index": 2,
            "clip": {"src": "input.mp4", "in": 20.0, "out": 23.0},
            "transition_in": {"type": "none", "dur": 0.0},
            "transition_out": {"type": "none", "dur": 0.0},
            "text_overlays": [],
            "t_start": 10.0,
            "t_end": 13.0,
        },
    ]
    sanitized, _issues = sanitize_timeline(timeline, video_duration_sec=100.0)
    assert sanitized[0]["t_start"] == 0.0
    assert sanitized[0]["t_end"] == 4.0
    assert sanitized[1]["t_start"] == sanitized[0]["t_end"]
    assert sanitized[2]["t_start"] == sanitized[1]["t_end"]


def test_crossfade_allows_overlap_and_validates_duration():
    timeline = [
        {
            "shot_index": 0,
            "clip": {"src": "input.mp4", "in": 0.0, "out": 2.0},
            "transition_in": {"type": "none", "dur": 0.0},
            "transition_out": {"type": "crossfade", "dur": 5.0},
            "text_overlays": [],
        },
        {
            "shot_index": 1,
            "clip": {"src": "input.mp4", "in": 3.0, "out": 5.0},
            "transition_in": {"type": "crossfade", "dur": 5.0},
            "transition_out": {"type": "none", "dur": 0.0},
            "text_overlays": [],
        },
    ]
    sanitized, _issues = sanitize_timeline(timeline, video_duration_sec=100.0)
    dur = sanitized[0]["transition_out"]["dur"]
    assert 0.0 <= dur <= 1.8
    assert sanitized[1]["t_start"] == sanitized[0]["t_end"] - dur


def test_transition_handle_budget_preserves_min_body():
    timeline = [
        {
            "shot_index": 0,
            "clip": {"src": "input.mp4", "in": 0.0, "out": 0.6},
            "transition_in": {"type": "none", "dur": 0.0},
            "transition_out": {"type": "fade", "dur": 1.0},
            "text_overlays": [],
        },
        {
            "shot_index": 1,
            "clip": {"src": "input.mp4", "in": 1.0, "out": 1.6},
            "transition_in": {"type": "fade", "dur": 1.0},
            "transition_out": {"type": "none", "dur": 0.0},
            "text_overlays": [],
        },
    ]
    sanitized, _issues = sanitize_timeline(timeline, video_duration_sec=100.0)
    d = sanitized[0]["transition_out"]["dur"]
    assert 0.0 <= d <= 0.4
