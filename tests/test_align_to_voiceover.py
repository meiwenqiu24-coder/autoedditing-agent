import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auto_video_agent.orchestrator.utils import align_timeline_to_voiceover


def test_align_timeline_to_voiceover_scales_clips():
    plan = {
        "voiceover_script": [
            {"text": "第一句", "real_sec": 2.0, "est_sec": 1.0},
            {"text": "第二句", "real_sec": 3.0, "est_sec": 1.0},
        ],
        "timeline": [
            {
                "shot_index": 0,
                "clip": {"src": "input.mp4", "in": 0.0, "out": 2.0},
                "transition_in": {"type": "none", "dur": 0.0},
                "transition_out": {"type": "none", "dur": 0.0},
                "text_overlays": [],
            },
            {
                "shot_index": 1,
                "clip": {"src": "input.mp4", "in": 5.0, "out": 7.0},
                "transition_in": {"type": "none", "dur": 0.0},
                "transition_out": {"type": "none", "dur": 0.0},
                "text_overlays": [],
            },
        ],
    }
    aligned, _issues = align_timeline_to_voiceover(plan, video_duration_sec=100.0)
    assert abs(aligned["timeline_meta"]["duration_sec"] - 5.0) < 1e-3

