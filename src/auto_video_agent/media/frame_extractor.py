from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Iterable, Literal

import cv2


@dataclasses.dataclass(frozen=True)
class VideoMeta:
    path: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int


@dataclasses.dataclass(frozen=True)
class Shot:
    index: int
    start_sec: float
    end_sec: float


@dataclasses.dataclass(frozen=True)
class Keyframe:
    id: str
    shot_index: int
    kind: Literal["start", "mid", "end"]
    t_sec: float
    frame_index: int
    path: str


def probe_video(video_path: str | os.PathLike[str]) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"Invalid FPS/frame_count for video: {video_path} (fps={fps}, frames={frame_count})")
        duration_sec = frame_count / fps
        return VideoMeta(
            path=str(video_path),
            duration_sec=float(duration_sec),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
    finally:
        cap.release()


def detect_shots(
    video_path: str | os.PathLike[str],
    *,
    backend: Literal["pyscenedetect", "opencv"] = "pyscenedetect",
    content_threshold: float = 27.0,
    min_shot_len_sec: float = 0.5,
) -> list[Shot]:
    meta = probe_video(video_path)
    min_scene_len_frames = max(1, int(round(min_shot_len_sec * meta.fps)))

    if backend == "pyscenedetect":
        try:
            scenes = _detect_shots_pyscenedetect(
                str(video_path),
                content_threshold=content_threshold,
                min_scene_len_frames=min_scene_len_frames,
            )
        except Exception:
            scenes = _detect_shots_opencv(
                str(video_path),
                fps=meta.fps,
                content_threshold=content_threshold,
                min_scene_len_frames=min_scene_len_frames,
            )
    else:
        scenes = _detect_shots_opencv(
            str(video_path),
            fps=meta.fps,
            content_threshold=content_threshold,
            min_scene_len_frames=min_scene_len_frames,
        )

    if not scenes:
        return [Shot(index=0, start_sec=0.0, end_sec=meta.duration_sec)]

    shots: list[Shot] = []
    for i, (start_sec, end_sec) in enumerate(scenes):
        start_sec = float(max(0.0, min(start_sec, meta.duration_sec)))
        end_sec = float(max(0.0, min(end_sec, meta.duration_sec)))
        if end_sec <= start_sec:
            continue
        shots.append(Shot(index=i, start_sec=start_sec, end_sec=end_sec))

    if not shots:
        return [Shot(index=0, start_sec=0.0, end_sec=meta.duration_sec)]

    shots = _normalize_shots(shots, meta.duration_sec)
    return shots


def extract_keyframes(
    video_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    max_per_shot: int = 3,
    backend: Literal["pyscenedetect", "opencv"] = "pyscenedetect",
    content_threshold: float = 27.0,
    min_shot_len_sec: float = 0.5,
    jpeg_quality: int = 92,
) -> tuple[VideoMeta, list[Shot], list[Keyframe]]:
    meta = probe_video(video_path)
    shots = detect_shots(
        video_path,
        backend=backend,
        content_threshold=content_threshold,
        min_shot_len_sec=min_shot_len_sec,
    )
    keyframes = _extract_keyframes_for_shots(
        str(video_path),
        meta=meta,
        shots=shots,
        output_dir=str(output_dir),
        max_per_shot=max_per_shot,
        jpeg_quality=jpeg_quality,
    )
    return meta, shots, keyframes


def _extract_keyframes_for_shots(
    video_path: str,
    *,
    meta: VideoMeta,
    shots: list[Shot],
    output_dir: str,
    max_per_shot: int,
    jpeg_quality: int,
) -> list[Keyframe]:
    max_per_shot = int(max(1, min(max_per_shot, 3)))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        keyframes: list[Keyframe] = []
        for shot in shots:
            requested_kinds = _kinds_for_shot(max_per_shot=max_per_shot)
            t_candidates = _time_candidates(shot.start_sec, shot.end_sec, requested_kinds)

            frame_indices: list[int] = []
            for t in t_candidates:
                idx = _sec_to_frame_index(t, meta.fps, meta.frame_count)
                frame_indices.append(idx)

            frame_indices = _dedupe_preserve_order(frame_indices)
            if not frame_indices:
                continue

            kinds = _kinds_from_count(len(frame_indices))
            for kind, frame_index in zip(kinds, frame_indices, strict=False):
                frame = _read_frame_at(cap, frame_index)
                if frame is None:
                    continue

                t_sec = frame_index / meta.fps
                frame_id = f"s{shot.index:04d}_{kind}"
                frame_path = out_dir / f"{frame_id}.jpg"

                ok = cv2.imwrite(
                    str(frame_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not ok:
                    raise RuntimeError(f"Failed to write frame: {frame_path}")

                keyframes.append(
                    Keyframe(
                        id=frame_id,
                        shot_index=shot.index,
                        kind=kind,
                        t_sec=float(t_sec),
                        frame_index=int(frame_index),
                        path=str(frame_path),
                    )
                )

        return keyframes
    finally:
        cap.release()


def _read_frame_at(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def _sec_to_frame_index(t_sec: float, fps: float, frame_count: int) -> int:
    idx = int(round(float(t_sec) * float(fps)))
    return int(max(0, min(idx, max(0, frame_count - 1))))


def _kinds_for_shot(*, max_per_shot: int) -> list[Literal["start", "mid", "end"]]:
    if max_per_shot <= 1:
        return ["mid"]
    if max_per_shot == 2:
        return ["start", "end"]
    return ["start", "mid", "end"]


def _kinds_from_count(n: int) -> list[Literal["start", "mid", "end"]]:
    if n <= 1:
        return ["mid"]
    if n == 2:
        return ["start", "end"]
    return ["start", "mid", "end"][:n]


def _time_candidates(
    start_sec: float,
    end_sec: float,
    kinds: list[Literal["start", "mid", "end"]],
) -> list[float]:
    dur = float(max(0.0, end_sec - start_sec))
    if dur <= 0:
        return []

    mid = start_sec + dur * 0.5
    eps = min(0.05, dur * 0.1)

    mapping: dict[str, float] = {
        "start": start_sec + eps,
        "mid": mid,
        "end": end_sec - eps,
    }
    return [mapping[k] for k in kinds if k in mapping]


def _dedupe_preserve_order(items: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _normalize_shots(shots: list[Shot], duration_sec: float) -> list[Shot]:
    shots_sorted = sorted(shots, key=lambda s: (s.start_sec, s.end_sec))
    merged: list[Shot] = []
    for shot in shots_sorted:
        if not merged:
            merged.append(shot)
            continue
        prev = merged[-1]
        if shot.start_sec <= prev.end_sec + 1e-6:
            merged[-1] = Shot(
                index=prev.index,
                start_sec=prev.start_sec,
                end_sec=max(prev.end_sec, shot.end_sec),
            )
        else:
            merged.append(shot)

    normalized: list[Shot] = []
    cursor = 0.0
    for i, shot in enumerate(merged):
        if shot.start_sec > cursor + 1e-6:
            normalized.append(Shot(index=len(normalized), start_sec=cursor, end_sec=shot.start_sec))
        normalized.append(Shot(index=len(normalized), start_sec=shot.start_sec, end_sec=shot.end_sec))
        cursor = shot.end_sec

    if cursor < duration_sec - 1e-6:
        normalized.append(Shot(index=len(normalized), start_sec=cursor, end_sec=duration_sec))

    if not normalized:
        return [Shot(index=0, start_sec=0.0, end_sec=duration_sec)]

    return [Shot(index=i, start_sec=s.start_sec, end_sec=s.end_sec) for i, s in enumerate(normalized)]


def _detect_shots_pyscenedetect(
    video_path: str,
    *,
    content_threshold: float,
    min_scene_len_frames: int,
) -> list[tuple[float, float]]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=float(content_threshold),
                min_scene_len=int(min_scene_len_frames),
            )
        )
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        out: list[tuple[float, float]] = []
        for start, end in scene_list:
            out.append((_timecode_to_sec(start), _timecode_to_sec(end)))
        return out
    except Exception:
        from scenedetect import ContentDetector, SceneManager, VideoManager

        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=float(content_threshold),
                min_scene_len=int(min_scene_len_frames),
            )
        )
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        out = []
        for start, end in scene_list:
            out.append((_timecode_to_sec(start), _timecode_to_sec(end)))
        video_manager.release()
        return out


def _timecode_to_sec(timecode) -> float:
    if hasattr(timecode, "get_seconds"):
        return float(timecode.get_seconds())
    if hasattr(timecode, "get_frames") and hasattr(timecode, "framerate"):
        frames = float(timecode.get_frames())
        fps = float(timecode.framerate)
        if fps > 0:
            return frames / fps
    if isinstance(timecode, (int, float)):
        return float(timecode)
    s = str(timecode)
    if ":" in s:
        parts = s.split(":")
        try:
            parts_f = [float(p) for p in parts]
            if len(parts_f) == 3:
                return parts_f[0] * 3600 + parts_f[1] * 60 + parts_f[2]
            if len(parts_f) == 2:
                return parts_f[0] * 60 + parts_f[1]
        except Exception:
            pass
    try:
        return float(s)
    except Exception as e:
        raise ValueError(f"Unsupported timecode: {timecode!r}") from e


def _detect_shots_opencv(
    video_path: str,
    *,
    fps: float,
    content_threshold: float,
    min_scene_len_frames: int,
) -> list[tuple[float, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        prev_gray = None
        cuts: list[int] = [0]
        last_cut = 0
        frame_index = 0
        sample_stride = max(1, int(round(fps / 5)))

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % sample_stride != 0:
                frame_index += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(diff.mean())
                if (
                    score >= float(content_threshold)
                    and (frame_index - last_cut) >= int(min_scene_len_frames)
                ):
                    cuts.append(frame_index)
                    last_cut = frame_index
            prev_gray = gray
            frame_index += 1

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or frame_index)
        if not cuts:
            return []

        if cuts[-1] >= total_frames:
            cuts[-1] = max(0, total_frames - 1)

        cuts = _dedupe_preserve_order([max(0, min(c, max(0, total_frames - 1))) for c in cuts])
        cuts_sorted = sorted(cuts)
        if cuts_sorted[0] != 0:
            cuts_sorted.insert(0, 0)

        scenes: list[tuple[float, float]] = []
        for a, b in zip(cuts_sorted, cuts_sorted[1:] + [total_frames], strict=False):
            if b <= a:
                continue
            scenes.append((a / fps, b / fps))

        return scenes
    finally:
        cap.release()
