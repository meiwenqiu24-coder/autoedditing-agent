from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from auto_video_agent.media.frame_extractor import Keyframe, Shot, VideoMeta, extract_keyframes
from auto_video_agent.orchestrator.llm_clients import OpenAIClient, OpenAIConfig


@dataclass(frozen=True)
class VisionAgentConfig:
    model: str = "gpt-4o-mini"
    backend: Literal["pyscenedetect", "opencv"] = "pyscenedetect"
    content_threshold: float = 27.0
    min_shot_len_sec: float = 0.5
    max_keyframes_per_shot: int = 3
    max_images_per_request: int = 12
    max_output_tokens: int = 1400


class VisionAgent:
    def __init__(self, config: VisionAgentConfig | None = None):
        self.config = config or VisionAgentConfig()
        self._llm = OpenAIClient(
            OpenAIConfig(
                model=self.config.model,
                max_output_tokens=self.config.max_output_tokens,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        )

    def run(
        self,
        *,
        video_path: str | os.PathLike[str],
        artifacts_dir: str | os.PathLike[str],
    ) -> dict[str, Any]:
        artifacts_dir = str(artifacts_dir)
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        frames_dir = Path(artifacts_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        meta, shots, keyframes = extract_keyframes(
            video_path,
            frames_dir,
            max_per_shot=self.config.max_keyframes_per_shot,
            backend=self.config.backend,
            content_threshold=self.config.content_threshold,
            min_shot_len_sec=self.config.min_shot_len_sec,
        )

        analyses = self._analyze_keyframes(meta=meta, shots=shots, keyframes=keyframes)
        visual_log = self._assemble_visual_log(meta=meta, shots=shots, keyframes=keyframes, analyses=analyses)

        out_path = Path(artifacts_dir) / "visual_log.json"
        out_path.write_text(json.dumps(visual_log, ensure_ascii=False, indent=2), encoding="utf-8")
        return visual_log

    def _analyze_keyframes(
        self,
        *,
        meta: VideoMeta,
        shots: list[Shot],
        keyframes: list[Keyframe],
    ) -> dict[str, Any]:
        if not keyframes:
            return {"frames": [], "shot_summaries": []}

        by_shot: dict[int, list[Keyframe]] = {}
        for k in keyframes:
            by_shot.setdefault(k.shot_index, []).append(k)

        shot_meta = {s.index: s for s in shots}
        shot_indices = sorted(by_shot.keys())

        batches: list[list[Keyframe]] = []
        current: list[Keyframe] = []
        for shot_idx in shot_indices:
            frames = sorted(by_shot[shot_idx], key=lambda x: x.t_sec)
            if len(current) + len(frames) > int(self.config.max_images_per_request) and current:
                batches.append(current)
                current = []
            current.extend(frames)
        if current:
            batches.append(current)

        merged_frames: dict[str, Any] = {}
        merged_shots: dict[int, Any] = {}

        for batch in batches:
            involved_shots = sorted({k.shot_index for k in batch})
            schema = _vision_response_schema()

            system_prompt = (
                "你是资深视觉分析师。你必须只输出严格 JSON，且必须符合给定 JSON Schema。"
                "不要输出多余文本，不要使用 markdown。"
                "如果无法确定，请使用 unknown 或空数组，不要编造。"
            )
            user_prompt = _build_user_prompt(meta=meta, shots=shot_meta, keyframes=batch)
            images = [k.path for k in batch]

            resp = self._llm.responses_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                images=images,
                json_schema=schema,
                schema_name="vision_log_batch",
                strict_schema=True,
            )

            frames = resp.get("frames", []) if isinstance(resp, dict) else []
            for f in frames:
                fid = f.get("frame_id")
                if isinstance(fid, str) and fid:
                    merged_frames[fid] = f

            shot_summaries = resp.get("shot_summaries", []) if isinstance(resp, dict) else []
            for s in shot_summaries:
                idx = s.get("shot_index")
                if isinstance(idx, int) and idx in involved_shots:
                    merged_shots[idx] = s

        return {
            "frames": list(merged_frames.values()),
            "shot_summaries": list(merged_shots.values()),
        }

    def _assemble_visual_log(
        self,
        *,
        meta: VideoMeta,
        shots: list[Shot],
        keyframes: list[Keyframe],
        analyses: dict[str, Any],
    ) -> dict[str, Any]:
        frames_by_id = {f.get("frame_id"): f for f in analyses.get("frames", []) if isinstance(f, dict)}
        shot_summaries_by_idx = {
            s.get("shot_index"): s for s in analyses.get("shot_summaries", []) if isinstance(s, dict)
        }

        keyframes_by_shot: dict[int, list[Keyframe]] = {}
        for k in keyframes:
            keyframes_by_shot.setdefault(k.shot_index, []).append(k)

        shots_out: list[dict[str, Any]] = []
        for shot in shots:
            kfs = sorted(keyframes_by_shot.get(shot.index, []), key=lambda x: x.t_sec)
            kf_out: list[dict[str, Any]] = []
            for k in kfs:
                a = frames_by_id.get(k.id, {})
                kf_out.append(
                    {
                        "frame_id": k.id,
                        "kind": k.kind,
                        "t_sec": float(k.t_sec),
                        "frame_path": str(Path(k.path).as_posix()),
                        "scene": a.get("scene", "unknown"),
                        "actions": a.get("actions", []),
                        "objects": a.get("objects", []),
                        "emotion": a.get("emotion", "unknown"),
                        "summary": a.get("summary", ""),
                        "saliency": a.get("saliency", 0.5),
                    }
                )

            shot_summary = shot_summaries_by_idx.get(shot.index, {})
            shots_out.append(
                {
                    "shot_index": int(shot.index),
                    "t_start": float(shot.start_sec),
                    "t_end": float(shot.end_sec),
                    "summary": shot_summary.get("summary", ""),
                    "dominant_emotion": shot_summary.get("dominant_emotion", "unknown"),
                    "top_actions": shot_summary.get("top_actions", []),
                    "top_objects": shot_summary.get("top_objects", []),
                    "keyframes": kf_out,
                }
            )

        return {
            "video_meta": {
                "path": meta.path,
                "duration_sec": meta.duration_sec,
                "fps": meta.fps,
                "frame_count": meta.frame_count,
                "width": meta.width,
                "height": meta.height,
            },
            "shots": shots_out,
        }


def _build_user_prompt(*, meta: VideoMeta, shots: dict[int, Shot], keyframes: list[Keyframe]) -> str:
    lines: list[str] = []
    lines.append("请分析下面这些关键帧图像。每张图像都对应 frame_id。")
    lines.append("视频元信息：")
    lines.append(
        json.dumps(
            {
                "duration_sec": meta.duration_sec,
                "fps": meta.fps,
                "width": meta.width,
                "height": meta.height,
            },
            ensure_ascii=False,
        )
    )
    lines.append("")
    lines.append("关键帧列表（按顺序与随附图片一致）：")
    for k in keyframes:
        shot = shots.get(k.shot_index)
        lines.append(
            json.dumps(
                {
                    "frame_id": k.id,
                    "shot_index": k.shot_index,
                    "kind": k.kind,
                    "t_sec": round(float(k.t_sec), 3),
                    "shot_t_start": round(float(shot.start_sec), 3) if shot else None,
                    "shot_t_end": round(float(shot.end_sec), 3) if shot else None,
                },
                ensure_ascii=False,
            )
        )
    lines.append("")
    lines.append(
        "输出要求：为每个 frame_id 生成 scene/actions/objects/emotion/summary/saliency。"
        "同时为每个 shot_index 生成一条 shot_summaries（summary/dominant_emotion/top_actions/top_objects）。"
        "actions/objects 使用中文短语数组，saliency 是 0~1 浮点数。"
        "不要做任何时间计算或推断不存在的事件。"
    )
    return "\n".join(lines)


def _vision_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "frame_id": {"type": "string"},
                        "shot_index": {"type": "integer"},
                        "scene": {"type": "string"},
                        "actions": {"type": "array", "items": {"type": "string"}},
                        "objects": {"type": "array", "items": {"type": "string"}},
                        "emotion": {"type": "string"},
                        "summary": {"type": "string"},
                        "saliency": {"type": "number"},
                    },
                    "required": [
                        "frame_id",
                        "shot_index",
                        "scene",
                        "actions",
                        "objects",
                        "emotion",
                        "summary",
                        "saliency",
                    ],
                },
            },
            "shot_summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "shot_index": {"type": "integer"},
                        "summary": {"type": "string"},
                        "dominant_emotion": {"type": "string"},
                        "top_actions": {"type": "array", "items": {"type": "string"}},
                        "top_objects": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "shot_index",
                        "summary",
                        "dominant_emotion",
                        "top_actions",
                        "top_objects",
                    ],
                },
            },
        },
        "required": ["frames", "shot_summaries"],
    }

