from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class EditorAgentConfig:
    font: str | None = None
    font_size: int = 48
    subtitle_font_size: int = 44
    text_color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 2
    bottom_margin_px: int = 90
    original_audio_enabled: bool = False
    original_audio_volume: float = 0.3


class EditorAgent:
    def __init__(self, config: EditorAgentConfig | None = None):
        self.config = config or EditorAgentConfig()

    def run(
        self,
        *,
        director_plan_path: str | os.PathLike[str],
        artifacts_dir: str | os.PathLike[str],
    ) -> str:
        CompositeAudioClip, AudioFileClip, VideoFileClip, TextClip, CompositeVideoClip = _moviepy_classes()

        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        plan_path = Path(director_plan_path)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise ValueError("director_plan.json must be a JSON object")

        timeline = plan.get("timeline")
        if not isinstance(timeline, list) or not timeline:
            raise ValueError("director_plan.timeline must be a non-empty array")

        style = plan.get("style", {}) if isinstance(plan.get("style"), dict) else {}
        subtitle_enabled = bool(style.get("subtitle", False))

        audio_plan = plan.get("audio_plan", {}) if isinstance(plan.get("audio_plan"), dict) else {}
        ducking_cfg = audio_plan.get("bgm_ducking", {}) if isinstance(audio_plan.get("bgm_ducking"), dict) else {}
        ducking_enabled = bool(ducking_cfg.get("enabled", True))
        target_db = ducking_cfg.get("target_db", -14.0)
        duck_factor = _db_to_amp(target_db)

        render = plan.get("render", {}) if isinstance(plan.get("render"), dict) else {}
        fps = float(render.get("fps", 30))
        resolution = render.get("resolution", "1080p")
        codec = str(render.get("codec", "libx264"))
        crf = render.get("crf", 18)

        output_path = artifacts_dir / "output.mp4"

        video_clips: list[Any] = []
        text_clips: list[Any] = []
        audio_clips: list[Any] = []
        final_video = None
        final_audio = None

        try:
            stitched = self._build_video_track(
                timeline=timeline,
                plan_base_dir=plan_path.parent,
                artifacts_dir=artifacts_dir,
                fps=fps,
                video_clips_out=video_clips,
                text_clips_out=text_clips,
                subtitle_enabled=subtitle_enabled,
                voiceover_script=plan.get("voiceover_script"),
            )

            final_duration = float(max(0.0, stitched.duration or 0.0))
            voiceover_audio_path = plan.get("voiceover_audio_path", "")
            bgm_path = plan.get("bgm_path", "")

            voiceover_clip = None
            if isinstance(voiceover_audio_path, str) and voiceover_audio_path.strip():
                vo_abs = self._resolve_path(voiceover_audio_path, artifacts_dir=artifacts_dir, plan_base_dir=plan_path.parent)
                voiceover_clip = AudioFileClip(str(vo_abs)).set_start(0)
                audio_clips.append(voiceover_clip)

            bgm_clip = None
            if isinstance(bgm_path, str) and bgm_path.strip():
                bgm_abs = self._resolve_path(bgm_path, artifacts_dir=artifacts_dir, plan_base_dir=plan_path.parent)
                bgm_clip = AudioFileClip(str(bgm_abs)).set_start(0)
                audio_clips.append(bgm_clip)

            original_audio = None
            if self.config.original_audio_enabled:
                original_audio = stitched.audio
                if original_audio is not None:
                    original_audio = original_audio.volumex(float(self.config.original_audio_volume))

            bgm_final = None
            if bgm_clip is not None and final_duration > 0:
                bgm_looped = _loop_or_trim_audio(bgm_clip, final_duration)
                if ducking_enabled and voiceover_clip is not None:
                    intervals = _voiceover_intervals(plan.get("voiceover_script"))
                    bgm_final = _apply_ducking(bgm_looped, intervals, duck_factor=duck_factor)
                else:
                    bgm_final = bgm_looped

            audio_tracks = []
            if original_audio is not None:
                audio_tracks.append(original_audio)
            if voiceover_clip is not None:
                audio_tracks.append(voiceover_clip)
            if bgm_final is not None:
                audio_tracks.append(bgm_final)

            if audio_tracks:
                final_audio = CompositeAudioClip(audio_tracks)
                audio_clips.append(final_audio)
                stitched = stitched.set_audio(final_audio)
            else:
                stitched = stitched.without_audio()

            stitched = self._apply_render_settings(stitched, fps=fps, resolution=resolution)
            final_video = stitched

            ffmpeg_params = []
            if crf is not None and str(crf).strip() != "":
                ffmpeg_params.extend(["-crf", str(crf)])

            final_video.write_videofile(
                str(output_path),
                fps=int(round(fps)),
                codec=codec,
                audio_codec="aac",
                ffmpeg_params=ffmpeg_params,
                threads=4,
                preset="medium",
                temp_audiofile=str(artifacts_dir / "temp_audio.m4a"),
                remove_temp=True,
            )

            return str(output_path)
        finally:
            for c in reversed(text_clips):
                _safe_close(c)
            for c in reversed(video_clips):
                _safe_close(c)
            for c in reversed(audio_clips):
                _safe_close(c)
            _safe_close(final_audio)
            _safe_close(final_video)

    def _build_video_track(
        self,
        *,
        timeline: list[Any],
        plan_base_dir: Path,
        artifacts_dir: Path,
        fps: float,
        video_clips_out: list[Any],
        text_clips_out: list[Any],
        subtitle_enabled: bool,
        voiceover_script: Any,
    ):
        _, _, VideoFileClip, TextClip, CompositeVideoClip = _moviepy_classes()
        concatenate_videoclips = _moviepy_concatenate_videoclips()

        clips: list[Any] = []
        paddings: list[float] = []
        items = [x for x in timeline if isinstance(x, dict)]
        if not items:
            raise ValueError("timeline has no valid items")

        for i, item in enumerate(items):
            clip_spec = item.get("clip", {})
            if not isinstance(clip_spec, dict):
                raise ValueError("timeline item missing clip")
            src = clip_spec.get("src", "")
            if not isinstance(src, str) or not src.strip():
                raise ValueError("clip.src must be a non-empty string")

            src_abs = self._resolve_path(src, artifacts_dir=artifacts_dir, plan_base_dir=plan_base_dir)
            v = VideoFileClip(str(src_abs))
            video_clips_out.append(v)

            t_in = float(clip_spec.get("in", 0.0))
            t_out = float(clip_spec.get("out", t_in))
            if t_out <= t_in:
                raise ValueError("clip.out must be > clip.in")

            sub = v.subclip(t_in, t_out)
            video_clips_out.append(sub)

            tin = item.get("transition_in", {}) if isinstance(item.get("transition_in"), dict) else {"type": "none", "dur": 0.0}
            tout = item.get("transition_out", {}) if isinstance(item.get("transition_out"), dict) else {"type": "none", "dur": 0.0}
            sub = _apply_transition_effects(sub, transition_in=tin, transition_out=tout)

            clips.append(sub)

            if i == 0:
                continue
            prev = items[i - 1]
            boundary = _boundary_transition(prev, item)
            if boundary["type"] == "crossfade":
                dur = float(boundary["dur"])
                clips[-1] = clips[-1].crossfadein(dur)
                paddings.append(-dur)
            else:
                paddings.append(0.0)

        stitched = concatenate_videoclips(clips, method="compose", padding=0.0)
        if any(p < 0 for p in paddings):
            stitched = _concat_with_variable_padding(clips, paddings)

        base_w = int(stitched.w)
        base_h = int(stitched.h)

        overlays = []
        for item in items:
            if not isinstance(item.get("text_overlays"), list):
                continue
            base_t = float(item.get("t_start", 0.0))
            for ov in item.get("text_overlays", []):
                if not isinstance(ov, dict):
                    continue
                txt = ov.get("text", "")
                if not isinstance(txt, str) or not txt.strip():
                    continue
                t = float(ov.get("t", 0.0))
                dur = float(ov.get("dur", 0.0))
                if dur <= 0:
                    continue
                start = base_t + t
                if start < 0:
                    start = 0.0
                overlays.append(
                    self._make_text_clip(
                        TextClip,
                        txt,
                        w=base_w,
                        h=base_h,
                        start=start,
                        dur=dur,
                        font=self.config.font,
                        fontsize=int(self.config.font_size),
                    )
                )

        if subtitle_enabled:
            subtitles = _subtitles_from_voiceover(
                TextClip,
                voiceover_script,
                w=base_w,
                h=base_h,
                font=self.config.font,
                fontsize=int(self.config.subtitle_font_size),
                bottom_margin_px=int(self.config.bottom_margin_px),
                text_color=self.config.text_color,
                stroke_color=self.config.stroke_color,
                stroke_width=int(self.config.stroke_width),
            )
            overlays.extend(subtitles)

        for tclip in overlays:
            text_clips_out.append(tclip)

        if overlays:
            composed = CompositeVideoClip([stitched, *overlays], size=(base_w, base_h))
            video_clips_out.append(composed)
            return composed

        return stitched

    def _make_text_clip(
        self,
        TextClip,
        text: str,
        *,
        w: int,
        h: int,
        start: float,
        dur: float,
        font: str | None,
        fontsize: int,
    ):
        kwargs: dict[str, Any] = {
            "fontsize": int(fontsize),
            "color": self.config.text_color,
            "stroke_color": self.config.stroke_color,
            "stroke_width": int(self.config.stroke_width),
            "method": "caption",
            "size": (int(w * 0.92), None),
        }
        if font:
            kwargs["font"] = font
        clip = TextClip(text, **kwargs)
        clip = clip.set_duration(float(dur)).set_start(float(start))
        clip = clip.set_position(("center", h - int(self.config.bottom_margin_px)))
        return clip

    def _apply_render_settings(self, clip, *, fps: float, resolution: Any):
        w, h = _parse_resolution(resolution, default=(clip.w, clip.h))
        if w and h and (clip.w != w or clip.h != h):
            clip = clip.resize(newsize=(w, h))
        clip = clip.set_fps(int(round(float(fps))))
        return clip

    def _resolve_path(self, p: str, *, artifacts_dir: Path, plan_base_dir: Path) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        candidate = artifacts_dir / path
        if candidate.exists():
            return candidate
        return plan_base_dir / path


def _safe_close(obj: Any) -> None:
    try:
        if obj is not None and hasattr(obj, "close"):
            obj.close()
    except Exception:
        pass


def _db_to_amp(db: Any) -> float:
    try:
        v = float(db)
    except Exception:
        v = -14.0
    if v >= 0:
        return 1.0
    return float(min(1.0, math.pow(10.0, v / 20.0)))


def _voiceover_intervals(voiceover_script: Any) -> list[tuple[float, float]]:
    if not isinstance(voiceover_script, list):
        return []
    out: list[tuple[float, float]] = []
    for item in voiceover_script:
        if not isinstance(item, dict):
            continue
        a = item.get("t_start")
        b = item.get("t_end")
        try:
            t0 = float(a)
            t1 = float(b)
        except Exception:
            continue
        if t1 <= t0:
            continue
        out.append((t0, t1))
    out.sort()
    merged: list[tuple[float, float]] = []
    for a, b in out:
        if not merged:
            merged.append((a, b))
            continue
        pa, pb = merged[-1]
        if a <= pb + 1e-3:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    return merged


def _apply_ducking(audio_clip, intervals: list[tuple[float, float]], *, duck_factor: float):
    if not intervals:
        return audio_clip

    def multiplier(t: float) -> float:
        for a, b in intervals:
            if a <= t <= b:
                return float(duck_factor)
        return 1.0

    def fl(gf: Callable, t: float):
        return gf(t) * multiplier(float(t))

    return audio_clip.fl(fl, keep_duration=True)


def _loop_or_trim_audio(audio_clip, duration: float):
    audio_loop = _moviepy_audio_loop()

    duration = float(max(0.0, duration))
    if duration <= 0:
        return audio_clip
    if audio_clip.duration is None:
        return audio_clip.set_duration(duration)
    if audio_clip.duration >= duration:
        return audio_clip.subclip(0, duration)
    return audio_loop(audio_clip, duration=duration)


def _apply_transition_effects(clip, *, transition_in: dict[str, Any], transition_out: dict[str, Any]):
    t_in = str(transition_in.get("type", "none"))
    d_in = float(max(0.0, float(transition_in.get("dur", 0.0) or 0.0)))
    t_out = str(transition_out.get("type", "none"))
    d_out = float(max(0.0, float(transition_out.get("dur", 0.0) or 0.0)))

    if t_in == "fade" and d_in > 0:
        clip = clip.fadein(d_in)
    if t_out == "fade" and d_out > 0:
        clip = clip.fadeout(d_out)
    if t_out == "crossfade" and d_out > 0:
        try:
            clip = clip.crossfadeout(d_out)
        except Exception:
            pass
    return clip


def _boundary_transition(prev_item: dict[str, Any], next_item: dict[str, Any]) -> dict[str, Any]:
    a_out = prev_item.get("transition_out", {}) if isinstance(prev_item.get("transition_out"), dict) else {}
    b_in = next_item.get("transition_in", {}) if isinstance(next_item.get("transition_in"), dict) else {}
    t_a = str(a_out.get("type", "none"))
    t_b = str(b_in.get("type", "none"))
    d_a = float(max(0.0, float(a_out.get("dur", 0.0) or 0.0)))
    d_b = float(max(0.0, float(b_in.get("dur", 0.0) or 0.0)))
    dur = float(max(d_a, d_b))
    if t_a == "crossfade" or t_b == "crossfade":
        return {"type": "crossfade", "dur": dur}
    if t_a == "none" and t_b == "none":
        return {"type": "none", "dur": 0.0}
    t = t_a if t_a != "none" else t_b
    return {"type": t, "dur": dur}


def _concat_with_variable_padding(clips: list[Any], paddings: list[float]):
    _, _, _, _, CompositeVideoClip = _moviepy_classes()

    if len(clips) <= 1:
        return clips[0]
    if len(paddings) != len(clips) - 1:
        raise ValueError("padding length mismatch")

    starts: list[float] = [0.0]
    for i in range(1, len(clips)):
        prev_start = starts[-1]
        prev_dur = float(clips[i - 1].duration or 0.0)
        pad = float(paddings[i - 1])
        starts.append(prev_start + prev_dur + pad)

    layers = []
    for c, s in zip(clips, starts, strict=False):
        layers.append(c.set_start(float(s)))

    end_t = 0.0
    for c, s in zip(clips, starts, strict=False):
        end_t = max(end_t, float(s) + float(c.duration or 0.0))

    return CompositeVideoClip(layers).set_duration(end_t)


def _subtitles_from_voiceover(
    TextClip,
    voiceover_script: Any,
    *,
    w: int,
    h: int,
    font: str | None,
    fontsize: int,
    bottom_margin_px: int,
    text_color: str,
    stroke_color: str,
    stroke_width: int,
) -> list[Any]:
    if not isinstance(voiceover_script, list):
        return []
    out: list[Any] = []
    for item in voiceover_script:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            t0 = float(item.get("t_start"))
            t1 = float(item.get("t_end"))
        except Exception:
            continue
        if t1 <= t0:
            continue
        kwargs: dict[str, Any] = {
            "fontsize": int(fontsize),
            "color": text_color,
            "stroke_color": stroke_color,
            "stroke_width": int(stroke_width),
            "method": "caption",
            "size": (int(w * 0.92), None),
        }
        if font:
            kwargs["font"] = font
        try:
            clip = TextClip(text, **kwargs)
        except Exception:
            continue
        clip = clip.set_start(t0).set_duration(t1 - t0)
        clip = clip.set_position(("center", h - int(bottom_margin_px)))
        out.append(clip)
    return out


def _parse_resolution(resolution: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(resolution, (tuple, list)) and len(resolution) == 2:
        try:
            return int(resolution[0]), int(resolution[1])
        except Exception:
            return default
    if not isinstance(resolution, str):
        return default
    r = resolution.strip().lower()
    if r in {"1080p", "fullhd"}:
        return 1920, 1080
    if r in {"720p", "hd"}:
        return 1280, 720
    if r in {"480p"}:
        return 854, 480
    if "x" in r:
        parts = r.split("x", 1)
        try:
            return int(parts[0]), int(parts[1])
        except Exception:
            return default
    return default


def _moviepy_classes():
    try:
        from moviepy.audio.AudioClip import CompositeAudioClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.video.VideoClip import TextClip
        from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
        from moviepy.video.io.VideoFileClip import VideoFileClip

        return CompositeAudioClip, AudioFileClip, VideoFileClip, TextClip, CompositeVideoClip
    except Exception:
        from moviepy.audio.AudioClip import CompositeAudioClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.editor import VideoFileClip
        from moviepy.video.VideoClip import TextClip
        from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

        return CompositeAudioClip, AudioFileClip, VideoFileClip, TextClip, CompositeVideoClip


def _moviepy_concatenate_videoclips():
    try:
        from moviepy.video.compositing.concatenate import concatenate_videoclips

        return concatenate_videoclips
    except Exception:
        from moviepy.editor import concatenate_videoclips

        return concatenate_videoclips


def _moviepy_audio_loop():
    try:
        from moviepy.audio.fx.all import audio_loop

        return audio_loop
    except Exception:
        from moviepy.audio.fx.AudioLoop import AudioLoop

        def _loop(clip, duration: float):
            return AudioLoop(duration=duration)(clip)

        return _loop
