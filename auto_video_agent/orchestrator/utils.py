from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class TimelineIssue:
    kind: Literal["overlap", "gap", "transition_shrink", "clip_trim", "clip_fixup", "reorder"]
    message: str
    index_a: int | None = None
    index_b: int | None = None


def estimate_zh_speech_seconds(text: str, *, chars_per_sec: float = 4.5, min_sec: float = 0.6) -> float:
    if not isinstance(text, str):
        return float(min_sec)
    stripped = text.strip()
    if not stripped:
        return float(min_sec)
    n = _count_speech_chars(stripped)
    sec = n / float(chars_per_sec)
    return float(max(min_sec, sec))


def align_timeline_to_voiceover(
    plan: dict[str, Any],
    *,
    video_duration_sec: float | None = None,
    min_body_sec: float = 0.2,
) -> tuple[dict[str, Any], list[TimelineIssue]]:
    issues: list[TimelineIssue] = []
    if not isinstance(plan, dict):
        raise ValueError("plan must be a dict")

    voiceover = plan.get("voiceover_script")
    if not isinstance(voiceover, list) or not voiceover:
        return plan, issues

    timeline = plan.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        return plan, issues

    voiceover_sanitized, v_issues = _sanitize_voiceover_script(voiceover, prefer_real=True)
    plan["voiceover_script"] = voiceover_sanitized
    issues.extend(v_issues)

    target_total = float(voiceover_sanitized[-1]["t_end"]) if voiceover_sanitized else 0.0
    if target_total <= 0:
        return plan, issues

    clips = [x for x in timeline if isinstance(x, dict)]
    if not clips:
        return plan, issues

    clips, _ = _order_clips(clips)
    for item in clips:
        _ensure_transitions(item)

    clips, t_issues = sanitize_timeline(clips, video_duration_sec=video_duration_sec)
    issues.extend(t_issues)

    current_total = _effective_timeline_duration(clips)
    if current_total <= 0:
        return plan, issues

    scale = float(target_total / current_total)
    desired: list[float] = []
    for item in clips:
        desired.append(_clip_duration(item) * scale)

    adjusted, a_issues = _apply_desired_clip_durations(
        clips,
        desired_durations=desired,
        target_total_sec=target_total,
        video_duration_sec=video_duration_sec,
        min_body_sec=min_body_sec,
    )
    issues.extend(a_issues)

    adjusted, s_issues = sanitize_timeline(adjusted, video_duration_sec=video_duration_sec)
    issues.extend(s_issues)
    plan["timeline"] = adjusted
    plan["timeline_meta"] = {
        "duration_sec": float(adjusted[-1]["t_end"]) if adjusted else 0.0,
        "issues": [i.__dict__ for i in issues],
    }
    return plan, issues


def sanitize_director_plan(
    plan: dict[str, Any],
    *,
    video_duration_sec: float | None = None,
) -> tuple[dict[str, Any], list[TimelineIssue]]:
    issues: list[TimelineIssue] = []
    if not isinstance(plan, dict):
        raise ValueError("Director plan must be a dict")

    voiceover = plan.get("voiceover_script")
    if isinstance(voiceover, list):
        sanitized_voiceover, v_issues = _sanitize_voiceover_script(voiceover)
        plan["voiceover_script"] = sanitized_voiceover
        issues.extend(v_issues)

    timeline = plan.get("timeline")
    if not isinstance(timeline, list):
        return plan, issues

    sanitized_timeline, t_issues = sanitize_timeline(
        timeline,
        video_duration_sec=video_duration_sec,
    )
    plan["timeline"] = sanitized_timeline
    issues.extend(t_issues)
    plan["timeline_meta"] = {
        "duration_sec": float(sanitized_timeline[-1]["t_end"]) if sanitized_timeline else 0.0,
        "issues": [i.__dict__ for i in issues],
    }
    return plan, issues


def sanitize_timeline(
    timeline: list[dict[str, Any]],
    *,
    video_duration_sec: float | None = None,
) -> tuple[list[dict[str, Any]], list[TimelineIssue]]:
    issues: list[TimelineIssue] = []
    clips = [x for x in timeline if isinstance(x, dict)]
    if not clips:
        return [], issues

    clips, reorder_issue = _order_clips(clips)
    if reorder_issue is not None:
        issues.append(reorder_issue)

    for i, item in enumerate(clips):
        _ensure_transitions(item)
        clip = item.get("clip")
        if not isinstance(clip, dict):
            item["clip"] = {"src": "input.mp4", "in": 0.0, "out": 0.0}
            issues.append(TimelineIssue(kind="clip_fixup", message="missing clip object", index_a=i))
            clip = item["clip"]

        clip_in = _as_float(clip.get("in"), default=0.0)
        clip_out = _as_float(clip.get("out"), default=clip_in)

        if video_duration_sec is not None:
            clip_in = float(max(0.0, min(clip_in, video_duration_sec)))
            clip_out = float(max(0.0, min(clip_out, video_duration_sec)))

        if clip_out < clip_in:
            clip_in, clip_out = clip_out, clip_in
            issues.append(TimelineIssue(kind="clip_fixup", message="swapped clip in/out", index_a=i))

        min_clip = 0.2
        if clip_out - clip_in < min_clip:
            clip_out = clip_in + min_clip
            if video_duration_sec is not None:
                clip_out = float(min(clip_out, video_duration_sec))
                clip_in = float(max(0.0, clip_out - min_clip))
            issues.append(TimelineIssue(kind="clip_trim", message="extended tiny clip to minimum length", index_a=i))

        clip["in"] = float(clip_in)
        clip["out"] = float(clip_out)

    for i in range(len(clips) - 1):
        a = clips[i]
        b = clips[i + 1]
        a_dur = _clip_duration(a)
        b_dur = _clip_duration(b)

        boundary = _boundary_transition(a, b)
        validated, changed = _validate_transition(boundary, a_dur, b_dur)
        if changed:
            issues.append(
                TimelineIssue(
                    kind="transition_shrink",
                    message="transition shortened to fit clip durations",
                    index_a=i,
                    index_b=i + 1,
                )
            )
        _apply_boundary_transition(a, b, validated)

    handle_issues = _enforce_transition_handle_budget(clips)
    issues.extend(handle_issues)

    prev_end = 0.0
    for i, item in enumerate(clips):
        dur = _clip_duration(item)
        if i == 0:
            t_start = 0.0
        else:
            boundary = _boundary_transition(clips[i - 1], item)
            if boundary["type"] == "crossfade" and boundary["dur"] > 0:
                t_start = prev_end - boundary["dur"]
            else:
                t_start = prev_end

        if "t_start" in item and _as_float(item.get("t_start"), default=t_start) < prev_end - 1e-6:
            issues.append(TimelineIssue(kind="overlap", message="overlap removed by re-timing", index_a=i - 1, index_b=i))
        if "t_start" in item and _as_float(item.get("t_start"), default=t_start) > prev_end + 1e-6:
            issues.append(TimelineIssue(kind="gap", message="gap removed by re-timing", index_a=i - 1, index_b=i))

        t_end = t_start + dur
        item["t_start"] = float(max(0.0, t_start))
        item["t_end"] = float(max(item["t_start"], t_end))
        prev_end = item["t_end"]
        t_cursor = prev_end

    return clips, issues


def _sanitize_voiceover_script(
    items: list[Any],
    *,
    prefer_real: bool = False,
) -> tuple[list[dict[str, Any]], list[TimelineIssue]]:
    issues: list[TimelineIssue] = []
    out: list[dict[str, Any]] = []
    t = 0.0
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue
        text = raw.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        est = raw.get("est_sec")
        real = raw.get("real_sec") if prefer_real else None
        chosen = real if real is not None else est
        dur_sec = _as_float(chosen, default=estimate_zh_speech_seconds(text))
        if dur_sec <= 0:
            dur_sec = estimate_zh_speech_seconds(text)
            issues.append(TimelineIssue(kind="clip_fixup", message="voiceover duration fixed", index_a=i))
        t_start = float(t)
        t_end = float(t + dur_sec)
        out.append(
            {
                "text": text,
                "est_sec": float(dur_sec),
                "real_sec": float(dur_sec) if prefer_real else _as_float(raw.get("real_sec"), default=None),
                "audio_path": raw.get("audio_path") if isinstance(raw.get("audio_path"), str) else None,
                "t_start": t_start,
                "t_end": t_end,
            }
        )
        t = t_end
    return out, issues


def _order_clips(clips: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], TimelineIssue | None]:
    if all("t_start" in c for c in clips):
        starts = [_as_float(c.get("t_start"), default=None) for c in clips]
        if all(isinstance(x, float) for x in starts):
            sorted_clips = [c for _, c in sorted(zip(starts, clips, strict=False), key=lambda x: x[0])]
            if sorted_clips != clips:
                return sorted_clips, TimelineIssue(kind="reorder", message="clips reordered by t_start")
            return clips, None
    return clips, None


def _ensure_transitions(item: dict[str, Any]) -> None:
    if not isinstance(item.get("transition_in"), dict):
        item["transition_in"] = {"type": "none", "dur": 0.0}
    if not isinstance(item.get("transition_out"), dict):
        item["transition_out"] = {"type": "none", "dur": 0.0}

    for k in ("transition_in", "transition_out"):
        t = item[k]
        t_type = t.get("type", "none")
        if not isinstance(t_type, str):
            t_type = "none"
        t["type"] = t_type
        t["dur"] = float(max(0.0, _as_float(t.get("dur"), default=0.0)))


def _clip_duration(item: dict[str, Any]) -> float:
    clip = item.get("clip", {})
    if not isinstance(clip, dict):
        return 0.0
    clip_in = _as_float(clip.get("in"), default=0.0)
    clip_out = _as_float(clip.get("out"), default=clip_in)
    return float(max(0.0, clip_out - clip_in))


def _boundary_transition(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_out = a.get("transition_out", {})
    b_in = b.get("transition_in", {})
    if not isinstance(a_out, dict):
        a_out = {"type": "none", "dur": 0.0}
    if not isinstance(b_in, dict):
        b_in = {"type": "none", "dur": 0.0}

    type_a = str(a_out.get("type", "none"))
    type_b = str(b_in.get("type", "none"))
    dur_a = float(max(0.0, _as_float(a_out.get("dur"), default=0.0)))
    dur_b = float(max(0.0, _as_float(b_in.get("dur"), default=0.0)))
    dur = float(max(dur_a, dur_b))

    if type_a == "crossfade" or type_b == "crossfade":
        return {"type": "crossfade", "dur": dur}
    if type_a == "none" and type_b == "none":
        return {"type": "none", "dur": 0.0}
    t = type_a if type_a != "none" else type_b
    if t not in {"fade", "wipe", "none"}:
        t = "fade"
    return {"type": t, "dur": dur}


def _validate_transition(boundary: dict[str, Any], a_dur: float, b_dur: float) -> tuple[dict[str, Any], bool]:
    t_type = str(boundary.get("type", "none"))
    dur = float(max(0.0, _as_float(boundary.get("dur"), default=0.0)))
    max_dur = float(max(0.0, min(a_dur, b_dur)))
    new_dur = float(min(dur, max_dur))
    if new_dur < 0.05:
        new_type = "none"
        new_dur = 0.0
    else:
        new_type = t_type if t_type != "none" else "fade"
        if new_type not in {"fade", "crossfade", "wipe"}:
            new_type = "fade"
    changed = new_dur != dur or new_type != t_type
    return {"type": new_type, "dur": new_dur}, changed


def _apply_boundary_transition(a: dict[str, Any], b: dict[str, Any], boundary: dict[str, Any]) -> None:
    if not isinstance(a.get("transition_out"), dict):
        a["transition_out"] = {"type": "none", "dur": 0.0}
    if not isinstance(b.get("transition_in"), dict):
        b["transition_in"] = {"type": "none", "dur": 0.0}
    a["transition_out"]["type"] = boundary["type"]
    a["transition_out"]["dur"] = float(boundary["dur"])
    b["transition_in"]["type"] = boundary["type"]
    b["transition_in"]["dur"] = float(boundary["dur"])


def _count_speech_chars(text: str) -> int:
    cleaned = re.sub(r"\s+", "", text)
    cleaned = re.sub(r"[，。！？：；、“”‘’（）()\[\]{}<>《》…—\-~·.,!?;:'\"/\\]", "", cleaned)
    return int(len(cleaned))


def _effective_timeline_duration(clips: list[dict[str, Any]]) -> float:
    if not clips:
        return 0.0
    if all("t_end" in c for c in clips):
        ends = [_as_float(c.get("t_end"), default=None) for c in clips]
        if ends and all(isinstance(x, float) for x in ends):
            return float(max(ends))
    total = 0.0
    for c in clips:
        total += _clip_duration(c)
    for i in range(len(clips) - 1):
        t = _boundary_transition(clips[i], clips[i + 1])
        if t["type"] == "crossfade":
            total -= float(max(0.0, t["dur"]))
    return float(max(0.0, total))


def _transition_handle_sec(t: dict[str, Any]) -> float:
    if not isinstance(t, dict):
        return 0.0
    t_type = str(t.get("type", "none"))
    dur = float(max(0.0, _as_float(t.get("dur"), default=0.0)))
    if t_type in {"fade", "crossfade", "wipe"}:
        return dur
    return 0.0


def _enforce_transition_handle_budget(clips: list[dict[str, Any]], *, min_body_sec: float = 0.2) -> list[TimelineIssue]:
    issues: list[TimelineIssue] = []
    if len(clips) < 2:
        return issues

    boundaries: list[dict[str, Any]] = []
    for i in range(len(clips) - 1):
        boundaries.append(_boundary_transition(clips[i], clips[i + 1]))

    for sweep in range(2):
        indices = range(len(clips))
        if sweep == 1:
            indices = reversed(range(len(clips)))
        for i in indices:
            dur = _clip_duration(clips[i])
            allowed = float(max(0.0, dur - float(min_body_sec)))
            in_d = boundaries[i - 1]["dur"] if i - 1 >= 0 else 0.0
            out_d = boundaries[i]["dur"] if i < len(boundaries) else 0.0
            total = float(max(0.0, in_d) + max(0.0, out_d))
            if total <= allowed + 1e-6:
                continue
            if total <= 1e-9:
                continue
            factor = allowed / total if total > 0 else 0.0
            if i - 1 >= 0:
                boundaries[i - 1]["dur"] = float(boundaries[i - 1]["dur"] * factor)
            if i < len(boundaries):
                boundaries[i]["dur"] = float(boundaries[i]["dur"] * factor)
            issues.append(
                TimelineIssue(
                    kind="transition_shrink",
                    message="transition shortened to preserve handles",
                    index_a=max(0, i - 1),
                    index_b=min(len(clips) - 1, i),
                )
            )

    for i, b in enumerate(boundaries):
        b_type = str(b.get("type", "none"))
        b_dur = float(max(0.0, _as_float(b.get("dur"), default=0.0)))
        if b_dur < 0.05:
            b_type = "none"
            b_dur = 0.0
        _apply_boundary_transition(clips[i], clips[i + 1], {"type": b_type, "dur": b_dur})

    return issues


def _apply_desired_clip_durations(
    clips: list[dict[str, Any]],
    *,
    desired_durations: list[float],
    target_total_sec: float,
    video_duration_sec: float | None,
    min_body_sec: float,
) -> tuple[list[dict[str, Any]], list[TimelineIssue]]:
    issues: list[TimelineIssue] = []
    if len(desired_durations) != len(clips):
        raise ValueError("desired_durations length mismatch")

    mins: list[float] = []
    maxs: list[float] = []
    current: list[float] = []
    for i, item in enumerate(clips):
        clip = item.get("clip", {})
        if not isinstance(clip, dict):
            clip = {"src": "input.mp4", "in": 0.0, "out": 0.0}
            item["clip"] = clip
        clip_in = float(_as_float(clip.get("in"), default=0.0) or 0.0)
        clip_out = float(_as_float(clip.get("out"), default=clip_in) or clip_in)
        if clip_out < clip_in:
            clip_in, clip_out = clip_out, clip_in
        base_dur = float(max(0.0, clip_out - clip_in))
        current.append(base_dur)

        handle_in = _transition_handle_sec(item.get("transition_in", {}))
        handle_out = _transition_handle_sec(item.get("transition_out", {}))
        min_dur = float(max(0.2, handle_in + float(min_body_sec) + handle_out))
        mins.append(min_dur)

        if video_duration_sec is None:
            max_dur = float(max(min_dur, base_dur))
        else:
            max_dur = float(max(min_dur, max(0.0, float(video_duration_sec) - clip_in)))
        maxs.append(max_dur)

    targets = []
    for i, d in enumerate(desired_durations):
        targets.append(float(min(maxs[i], max(mins[i], float(d)))))

    effective_total = _effective_duration_from_durations(clips, targets)
    if effective_total <= 0:
        return clips, issues

    delta = float(target_total_sec - effective_total)
    if abs(delta) > 1e-3:
        slack = [maxs[i] - targets[i] for i in range(len(targets))]
        reducible = [targets[i] - mins[i] for i in range(len(targets))]
        if delta > 0:
            _distribute_delta(targets, slack, delta)
        else:
            _distribute_delta(targets, reducible, delta)

    for i, item in enumerate(clips):
        clip = item.get("clip", {})
        if not isinstance(clip, dict):
            continue
        clip_in = float(_as_float(clip.get("in"), default=0.0) or 0.0)
        clip["out"] = float(clip_in + targets[i])

    issues.append(TimelineIssue(kind="clip_trim", message="timeline aligned to voiceover real durations"))
    return clips, issues


def _effective_duration_from_durations(clips: list[dict[str, Any]], durations: list[float]) -> float:
    total = float(sum(max(0.0, d) for d in durations))
    for i in range(len(clips) - 1):
        t = _boundary_transition(clips[i], clips[i + 1])
        if t["type"] == "crossfade":
            total -= float(max(0.0, _as_float(t.get("dur"), default=0.0)))
    return float(max(0.0, total))


def _distribute_delta(targets: list[float], capacity: list[float], delta: float) -> None:
    remaining = float(delta)
    sign = 1.0 if remaining > 0 else -1.0
    remaining = abs(remaining)
    n = len(targets)
    for _ in range(3):
        total_cap = sum(max(0.0, c) for c in capacity)
        if total_cap <= 1e-9 or remaining <= 1e-6:
            break
        for i in range(n):
            cap_i = max(0.0, capacity[i])
            if cap_i <= 0:
                continue
            portion = remaining * (cap_i / total_cap)
            applied = min(cap_i, portion)
            targets[i] += sign * applied
            capacity[i] -= applied
            remaining -= applied
            if remaining <= 1e-6:
                break


def _as_float(value: Any, *, default: float | None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return default
        try:
            return float(v)
        except Exception:
            return default
    return default
