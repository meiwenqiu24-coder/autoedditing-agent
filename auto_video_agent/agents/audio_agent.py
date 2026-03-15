from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_video_agent.orchestrator.utils import align_timeline_to_voiceover


@dataclass(frozen=True)
class AudioAgentConfig:
    bgm_library_dir: str = "media/bgm_library"
    edge_tts_output_format: str = "riff-24khz-16bit-mono-pcm"
    default_voice: str = "zh-CN-XiaoxiaoNeural"


class AudioAgent:
    def __init__(self, config: AudioAgentConfig | None = None):
        self.config = config or AudioAgentConfig()

    def run(
        self,
        *,
        director_plan_path: str | os.PathLike[str],
        artifacts_dir: str | os.PathLike[str],
        video_duration_sec: float | None = None,
    ) -> dict[str, Any]:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        plan_path = Path(director_plan_path)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise ValueError("director_plan.json must be a JSON object")

        audio_dir = artifacts_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        voice = self._get_tts_voice(plan)
        voiceover = plan.get("voiceover_script")
        if not isinstance(voiceover, list) or not voiceover:
            raise ValueError("director_plan.voiceover_script must be a non-empty array")

        wav_paths: list[Path] = []
        for i, item in enumerate(voiceover):
            if not isinstance(item, dict):
                continue
            text = item.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            out_path = audio_dir / f"vo_{i}.wav"
            _synthesize_edge_tts_wav(
                text=text,
                voice=voice,
                output_path=out_path,
                output_format=self.config.edge_tts_output_format,
            )
            real_sec = _wav_duration_sec(out_path)
            item["audio_path"] = str(out_path.relative_to(artifacts_dir).as_posix())
            item["real_sec"] = float(real_sec)
            item["est_sec"] = float(real_sec)
            wav_paths.append(out_path)

        merged_path = audio_dir / "voiceover.wav"
        _concat_wav_files(wav_paths, merged_path)
        plan["voiceover_audio_path"] = str(merged_path.relative_to(artifacts_dir).as_posix())

        bgm_selected = _select_bgm_file(
            library_dir=Path(self.config.bgm_library_dir),
            tags=plan.get("audio_plan", {}).get("bgm_tag") if isinstance(plan.get("audio_plan"), dict) else None,
        )
        if bgm_selected is not None:
            dst = audio_dir / f"bgm{bgm_selected.suffix.lower()}"
            shutil.copyfile(bgm_selected, dst)
            plan["bgm_path"] = str(dst.relative_to(artifacts_dir).as_posix())
        else:
            plan["bgm_path"] = ""

        aligned, _issues = align_timeline_to_voiceover(plan, video_duration_sec=video_duration_sec)
        out_path = artifacts_dir / "director_plan.json"
        out_path.write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")
        return aligned

    def _get_tts_voice(self, plan: dict[str, Any]) -> str:
        audio_plan = plan.get("audio_plan", {})
        if isinstance(audio_plan, dict):
            v = audio_plan.get("tts_voice")
            if isinstance(v, str) and v.strip():
                return v.strip()
        return self.config.default_voice


def _synthesize_edge_tts_wav(
    *,
    text: str,
    voice: str,
    output_path: Path,
    output_format: str,
) -> None:
    try:
        import edge_tts
    except Exception as e:
        raise RuntimeError("edge-tts not installed. Please install edge-tts.") from e

    async def _run():
        communicate = edge_tts.Communicate(text=text, voice=voice, output_format=output_format)
        await communicate.save(str(output_path))

    try:
        asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()


def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            return 0.0
        return float(frames) / float(rate)


def _concat_wav_files(paths: list[Path], out_path: Path) -> None:
    if not paths:
        raise ValueError("No wav files to concat")

    with wave.open(str(paths[0]), "rb") as wf0:
        params = wf0.getparams()

    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        for p in paths:
            with wave.open(str(p), "rb") as wf:
                if wf.getparams() != params:
                    raise ValueError("All wav files must have identical params to concat")
                out.writeframes(wf.readframes(wf.getnframes()))


def _select_bgm_file(*, library_dir: Path, tags: Any) -> Path | None:
    library_dir.mkdir(parents=True, exist_ok=True)
    _ = tags
    candidates = []
    for ext in ("*.wav", "*.mp3"):
        candidates.extend(library_dir.glob(ext))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return random.choice(candidates)

