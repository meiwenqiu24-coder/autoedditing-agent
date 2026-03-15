from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_video_agent.orchestrator.llm_clients import OpenAIClient, OpenAIConfig
from auto_video_agent.orchestrator.utils import sanitize_director_plan


@dataclass(frozen=True)
class DirectorAgentConfig:
    model: str = "gpt-4o-mini"
    max_output_tokens: int = 2200


class DirectorAgent:
    def __init__(self, config: DirectorAgentConfig | None = None):
        self.config = config or DirectorAgentConfig()
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
        visual_log_path: str | os.PathLike[str],
        artifacts_dir: str | os.PathLike[str],
        style_hint: str | None = None,
    ) -> dict[str, Any]:
        artifacts_dir = str(artifacts_dir)
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

        visual_log = json.loads(Path(visual_log_path).read_text(encoding="utf-8"))
        if not isinstance(visual_log, dict):
            raise ValueError("visual_log.json must be a JSON object")

        video_meta = visual_log.get("video_meta", {})
        duration_sec = None
        if isinstance(video_meta, dict):
            try:
                duration_sec = float(video_meta.get("duration_sec"))
            except Exception:
                duration_sec = None

        system_prompt = (
            "你是专业的视频导演与编剧，擅长根据视觉日志写旁白并规划镜头。"
            "你必须只输出严格 JSON，且必须符合给定 JSON Schema。"
            "不要输出多余文本，不要使用 markdown。"
            "你必须避免时间计算幻觉：不要手算复杂时间轴，只需要给出每段素材的源视频 in/out，以及建议的转场类型与时长。"
            "旁白每句必须给出 est_sec（中文语速约 4~5 字/秒）。"
        )

        user_prompt = _build_director_prompt(visual_log=visual_log, style_hint=style_hint)
        schema = director_response_schema()
        plan = self._llm.responses_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=(),
            json_schema=schema,
            schema_name="director_plan",
            strict_schema=True,
        )
        if not isinstance(plan, dict):
            raise ValueError("Director response must be a JSON object")

        sanitized, _issues = sanitize_director_plan(plan, video_duration_sec=duration_sec)
        out_path = Path(artifacts_dir) / "director_plan.json"
        out_path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8")
        return sanitized


def _build_director_prompt(*, visual_log: dict[str, Any], style_hint: str | None) -> str:
    shots = visual_log.get("shots", [])
    if not isinstance(shots, list):
        shots = []

    shot_summaries: list[dict[str, Any]] = []
    for s in shots:
        if not isinstance(s, dict):
            continue
        shot_summaries.append(
            {
                "shot_index": s.get("shot_index"),
                "t_start": s.get("t_start"),
                "t_end": s.get("t_end"),
                "summary": s.get("summary", ""),
                "dominant_emotion": s.get("dominant_emotion", "unknown"),
                "top_actions": s.get("top_actions", []),
                "top_objects": s.get("top_objects", []),
            }
        )

    payload = {
        "style_hint": style_hint or "",
        "shots": shot_summaries,
    }

    return (
        "输入是视觉日志摘要（shots）。请你完成两件事：\n"
        "1) 撰写旁白脚本 voiceover_script：每一句用中文，适配画面，避免臆造不存在的信息；每句必须给 est_sec。\n"
        "2) 规划 timeline：从 shots 中挑选合适的镜头顺序，每个 timeline item 绑定一个 shot_index，"
        "并给出 clip.in/clip.out（必须落在该 shot 的 t_start~t_end 内）。每个 item 可选转场：transition_out 与下一段 transition_in 必须一致。\n"
        "强制要求：\n"
        "- 只用给定 shots 的内容，不要编造额外事件。\n"
        "- timeline 中每段 clip 时长建议 2~6 秒，尽量覆盖旁白内容。\n"
        "- 转场优先用 fade 或 none；只有确实需要才用 crossfade，crossfade 会允许片段重叠。\n"
        "- 输出必须符合 JSON Schema。\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def director_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "style": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "pace": {"type": "string"},
                    "subtitle": {"type": "boolean"},
                    "transition": {"type": "string"},
                    "color": {"type": "string"},
                },
                "required": ["name", "pace", "subtitle", "transition", "color"],
            },
            "voiceover_script": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "est_sec": {"type": "number"},
                        "real_sec": {"type": "number"},
                        "audio_path": {"type": "string"},
                        "t_start": {"type": "number"},
                        "t_end": {"type": "number"},
                    },
                    "required": ["text", "est_sec"],
                },
            },
            "voiceover_audio_path": {"type": "string"},
            "bgm_path": {"type": "string"},
            "timeline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "shot_index": {"type": "integer"},
                        "clip": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "src": {"type": "string"},
                                "in": {"type": "number"},
                                "out": {"type": "number"},
                            },
                            "required": ["src", "in", "out"],
                        },
                        "transition_in": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string"},
                                "dur": {"type": "number"},
                            },
                            "required": ["type", "dur"],
                        },
                        "transition_out": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string"},
                                "dur": {"type": "number"},
                            },
                            "required": ["type", "dur"],
                        },
                        "text_overlays": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "t": {"type": "number"},
                                    "dur": {"type": "number"},
                                    "text": {"type": "string"},
                                    "pos": {"type": "string"},
                                },
                                "required": ["t", "dur", "text", "pos"],
                            },
                        },
                    },
                    "required": ["shot_index", "clip", "transition_in", "transition_out", "text_overlays"],
                },
            },
            "audio_plan": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tts_voice": {"type": "string"},
                    "bgm_tag": {"type": "array", "items": {"type": "string"}},
                    "bgm_ducking": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "target_db": {"type": "number"},
                        },
                        "required": ["enabled", "target_db"],
                    },
                },
                "required": ["tts_voice", "bgm_tag", "bgm_ducking"],
            },
            "render": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "resolution": {"type": "string"},
                    "fps": {"type": "number"},
                    "codec": {"type": "string"},
                    "crf": {"type": "number"},
                },
                "required": ["resolution", "fps", "codec", "crf"],
            },
        },
        "required": ["style", "voiceover_script", "timeline", "audio_plan", "render"],
    }
