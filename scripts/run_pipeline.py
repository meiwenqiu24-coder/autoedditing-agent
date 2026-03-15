from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from auto_video_agent.agents.audio_agent import AudioAgent
    from auto_video_agent.agents.director_agent import DirectorAgent
    from auto_video_agent.agents.editor_agent import EditorAgent, EditorAgentConfig
    from auto_video_agent.agents.vision_agent import VisionAgent

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    workspace = Path(args.workspace).expanduser().resolve()
    run_dir = _make_run_dir(workspace)

    print(f"Run directory: {run_dir}")
    print("")

    print("Starting Vision Agent...")
    try:
        vision = VisionAgent()
        visual_log = vision.run(video_path=str(input_path), artifacts_dir=str(run_dir))
        visual_log_path = run_dir / "visual_log.json"
        print(f"Vision Agent finished. Artifact: {visual_log_path}")
    except Exception as e:
        print(f"Vision Agent failed: {e}")
        raise

    print("")
    print("Starting Director Agent...")
    try:
        director = DirectorAgent()
        director_plan = director.run(
            visual_log_path=str(visual_log_path),
            artifacts_dir=str(run_dir),
            style_hint=args.style_hint,
        )
        director_plan_path = run_dir / "director_plan.json"
        print(f"Director Agent finished. Artifact: {director_plan_path}")
    except Exception as e:
        print(f"Director Agent failed: {e}")
        raise

    print("")
    print("Starting Audio Agent...")
    try:
        duration_sec = _extract_video_duration_sec(visual_log)
        audio = AudioAgent()
        director_plan = audio.run(
            director_plan_path=str(director_plan_path),
            artifacts_dir=str(run_dir),
            video_duration_sec=duration_sec,
        )
        director_plan_path.write_text(json.dumps(director_plan, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Audio Agent finished. Artifact: {director_plan_path}")
    except Exception as e:
        print(f"Audio Agent failed: {e}")
        raise

    print("")
    print("Starting Editor Agent...")
    try:
        font = os.environ.get("AUTO_VIDEO_FONT") or None
        editor = EditorAgent(EditorAgentConfig(font=font))
        output_path = editor.run(director_plan_path=str(director_plan_path), artifacts_dir=str(run_dir))
        print(f"Editor Agent finished. Artifact: {output_path}")
    except Exception as e:
        print(f"Editor Agent failed: {e}")
        raise

    print("")
    print("Pipeline finished successfully.")
    return 0


def _parse_args():
    parser = argparse.ArgumentParser(prog="run_pipeline", description="Auto-Video-Agent end-to-end pipeline")
    parser.add_argument("--input", required=True, help="Input video path (mp4/mov/etc.)")
    parser.add_argument("--workspace", required=True, help="Workspace directory (will create runs/<run_id>/)")
    parser.add_argument("--style_hint", default=None, help="Optional style hint (e.g., 幽默/专业技术讲解)")
    return parser.parse_args()


def _make_run_dir(workspace: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{ts}_{uuid.uuid4().hex[:8]}"
    run_dir = workspace / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _extract_video_duration_sec(visual_log: dict) -> float | None:
    try:
        meta = visual_log.get("video_meta", {})
        if isinstance(meta, dict):
            v = meta.get("duration_sec")
            if v is not None:
                return float(v)
    except Exception:
        return None
    return None


if __name__ == "__main__":
    raise SystemExit(main())

