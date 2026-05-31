"""Compare baseline and diagnostic replay outputs without assuming cap-based tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return loaded


def _load_run_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if "average_latency_ms_by_stage" in payload:
        return {"summary": {}, "trace_summary": payload, "source_path": str(path)}
    trace_summary_path = payload.get("runtime_trace_summary_path")
    if not trace_summary_path:
        raise ValueError(
            f"{path} must be a runtime_trace_summary.json or a summary.json with runtime_trace_summary_path"
        )
    trace_summary = _load_json(Path(str(trace_summary_path)))
    return {"summary": payload, "trace_summary": trace_summary, "source_path": str(path)}


def _metric(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(payload: dict[str, Any], *keys: str, default: str = "unknown") -> str:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    text = str(value).strip()
    return text if text else default


def _bool_summary(trace_summary: dict[str, Any], key: str) -> str:
    return "yes" if trace_summary.get(key) else "no"


def _summarize_run(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    trace_summary = payload["trace_summary"]
    backend = trace_summary.get("backend_diagnostics", {})
    warmup = trace_summary.get("warmup_section", {})
    return {
        "source_path": payload["source_path"],
        "frames_submitted": int(summary.get("frames_submitted", trace_summary.get("total_frames_submitted", 0))),
        "frames_processed": int(summary.get("frames_processed", trace_summary.get("total_ips_processed", 0))),
        "frames_dropped": int(summary.get("frames_dropped", trace_summary.get("total_dropped_overflow", 0))),
        "stale_dropped": int(summary.get("stale_frames_dropped", trace_summary.get("total_stale_dropped", 0))),
        "warmup_frames_submitted": int(summary.get("warmup_frames_submitted", trace_summary.get("warmup_frames_submitted", 0))),
        "warmup_frames_processed": int(summary.get("warmup_frames_processed", trace_summary.get("warmup_frames_processed", 0))),
        "queue_wait_avg_ms": _metric(trace_summary, "average_latency_ms_by_stage", "queue_wait_ms"),
        "queue_wait_max_ms": _metric(trace_summary, "max_latency_ms_by_stage", "queue_wait_ms"),
        "worker_pre_rpm_delay_avg_ms": _metric(trace_summary, "average_latency_ms_by_stage", "worker_pre_rpm_delay_ms"),
        "worker_pre_rpm_delay_max_ms": _metric(trace_summary, "max_latency_ms_by_stage", "worker_pre_rpm_delay_ms"),
        "rpm_total_avg_ms": _metric(trace_summary, "average_latency_ms_by_stage", "rpm_total_ms"),
        "rpm_total_max_ms": _metric(trace_summary, "max_latency_ms_by_stage", "rpm_total_ms"),
        "od_model_avg_ms": _metric(trace_summary, "average_object_detection_model_ms_over_calls"),
        "od_model_max_ms": _metric(trace_summary, "max_latency_ms_by_stage", "object_detection_total_ms"),
        "fd_model_avg_ms": _metric(trace_summary, "average_face_detection_model_ms_over_calls"),
        "fd_model_max_ms": _metric(trace_summary, "face_detection_max_call_ms"),
        "fr_model_avg_ms": _metric(trace_summary, "average_face_recognition_model_ms_over_calls"),
        "fr_model_max_ms": _metric(trace_summary, "max_latency_ms_by_stage", "face_recognition_total_ms"),
        "motion_regions_dropped_by_cap": int(trace_summary.get("total_motion_regions_dropped_by_cap", 0)),
        "person_rois_dropped_by_cap": int(trace_summary.get("total_person_rois_dropped_by_cap", 0)),
        "face_rois_dropped_by_cap": int(trace_summary.get("total_face_rois_dropped_by_cap", 0)),
        "max_persons": int(trace_summary.get("max_person_rois_count", 0)),
        "four_person_detection": "yes" if trace_summary.get("frame_indexes_with_4_persons") else "no",
        "persons_per_od_call": _metric(trace_summary, "average_persons_returned_per_od_call"),
        "first_measured_rpm_total_ms": _metric(trace_summary, "first_measured_frame_rpm_total_ms"),
        "warmup_first_rpm_total_ms": _metric(warmup, "warmup_first_frame_rpm_total_ms"),
        "warmup_max_rpm_total_ms": _metric(warmup, "warmup_max_rpm_total_ms"),
        "bottleneck_stage": _text(trace_summary, "bottleneck_stage_guess"),
        "od_backend": _text(backend, "object_detection", "backend"),
        "od_device": _text(backend, "object_detection", "device_provider"),
        "fd_backend": _text(backend, "face_detection", "backend"),
        "fd_device": _text(backend, "face_detection", "device_provider"),
        "fr_backend": _text(backend, "face_recognition", "backend"),
        "fr_device": _text(backend, "face_recognition", "device_provider"),
        "motion_backend": _text(backend, "motion_detection", "backend"),
        "motion_device": _text(backend, "motion_detection", "device_provider"),
    }


def _diagnose(baseline: dict[str, Any], tuned: dict[str, Any]) -> dict[str, str]:
    tuned_model_total = tuned["od_model_avg_ms"] + tuned["fd_model_avg_ms"] + tuned["fr_model_avg_ms"]
    queue_bound = tuned["queue_wait_avg_ms"] > tuned_model_total
    cpu_backends = {
        tuned["od_device"].lower(),
        tuned["fd_device"].lower(),
        tuned["fr_device"].lower(),
        tuned["motion_device"].lower(),
    }
    cpu_bound = (not queue_bound) and all(
        any(token in device for token in ("cpu", "unknown", "executionprovider"))
        for device in cpu_backends
    )
    cold_start = tuned["warmup_first_rpm_total_ms"] > 0.0 and (
        tuned["warmup_first_rpm_total_ms"] > (tuned["first_measured_rpm_total_ms"] * 1.5)
        or tuned["warmup_max_rpm_total_ms"] > tuned["rpm_total_max_ms"]
    )
    coverage_preserved = (
        tuned["person_rois_dropped_by_cap"] == 0
        and tuned["face_rois_dropped_by_cap"] == 0
        and tuned["max_persons"] >= baseline["max_persons"]
    )
    motion_cap_beneficial = (
        tuned["rpm_total_avg_ms"] <= baseline["rpm_total_avg_ms"]
        and tuned["person_rois_dropped_by_cap"] == 0
        and tuned["face_rois_dropped_by_cap"] == 0
    )
    safe_without_caps = "yes" if coverage_preserved else "no"
    latency_source = "queue/backlog" if queue_bound else "model inference"
    bottleneck = tuned["bottleneck_stage"]
    if bottleneck in {"unknown", "rpm_total_ms"}:
        model_candidates = {
            "object_detection_model_ms": tuned["od_model_avg_ms"],
            "face_detection_model_ms": tuned["fd_model_avg_ms"],
            "face_recognition_model_ms": tuned["fr_model_avg_ms"],
        }
        bottleneck = max(model_candidates, key=lambda key: model_candidates[key])
    return {
        "is_cpu_bound": "yes" if cpu_bound else "no",
        "latency_source": latency_source,
        "true_bottleneck_after_warmup": bottleneck,
        "cold_start_lazy_load_causing_max_latency": "yes" if cold_start else "no",
        "motion_roi_cap_still_beneficial": "yes" if motion_cap_beneficial else "no",
        "safe_to_continue_without_person_face_caps": safe_without_caps,
        "recommended_next_action": (
            "Measure and optimize the bottleneck stage/backend identified here before any new threshold or ROI-cap tuning."
        ),
    }


def _print_run(label: str, run: dict[str, Any]) -> None:
    print(label)
    print(f"  source: {run['source_path']}")
    print(
        "  frames: "
        f"submitted={run['frames_submitted']} processed={run['frames_processed']} "
        f"dropped={run['frames_dropped']} stale={run['stale_dropped']}"
    )
    print(
        "  warmup: "
        f"submitted={run['warmup_frames_submitted']} processed={run['warmup_frames_processed']}"
    )
    print(
        "  queue_wait_ms avg/max: "
        f"{run['queue_wait_avg_ms']:.3f} / {run['queue_wait_max_ms']:.3f}"
    )
    print(
        "  rpm_total_ms avg/max: "
        f"{run['rpm_total_avg_ms']:.3f} / {run['rpm_total_max_ms']:.3f}"
    )
    print(
        "  OD model ms avg/max: "
        f"{run['od_model_avg_ms']:.3f} / {run['od_model_max_ms']:.3f}"
    )
    print(
        "  FD model ms avg/max: "
        f"{run['fd_model_avg_ms']:.3f} / {run['fd_model_max_ms']:.3f}"
    )
    print(
        "  FR model ms avg/max: "
        f"{run['fr_model_avg_ms']:.3f} / {run['fr_model_max_ms']:.3f}"
    )
    print(
        "  backends: "
        f"OD={run['od_backend']}:{run['od_device']} FD={run['fd_backend']}:{run['fd_device']} "
        f"FR={run['fr_backend']}:{run['fr_device']} Motion={run['motion_backend']}:{run['motion_device']}"
    )
    print(
        "  warmup impact: "
        f"first_measured={run['first_measured_rpm_total_ms']:.3f} "
        f"warmup_first={run['warmup_first_rpm_total_ms']:.3f} "
        f"warmup_max={run['warmup_max_rpm_total_ms']:.3f}"
    )
    print(
        "  coverage: "
        f"max_persons={run['max_persons']} four_person_detection={run['four_person_detection']} "
        f"motion_cap_drops={run['motion_regions_dropped_by_cap']} "
        f"person_cap_drops={run['person_rois_dropped_by_cap']} face_cap_drops={run['face_rois_dropped_by_cap']}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and diagnostic replay outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("baseline", type=Path, help="Path to baseline summary.json or runtime_trace_summary.json")
    parser.add_argument("tuned", type=Path, help="Path to tuned/warmup summary.json or runtime_trace_summary.json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    baseline = _summarize_run(_load_run_payload(args.baseline))
    tuned = _summarize_run(_load_run_payload(args.tuned))
    diagnosis = _diagnose(baseline, tuned)

    _print_run("Baseline", baseline)
    _print_run("Tuned/Warmup", tuned)
    print("Diagnosis")
    print(f"  CPU-bound: {diagnosis['is_cpu_bound']}")
    print(f"  Latency mostly from: {diagnosis['latency_source']}")
    print(f"  Bottleneck after warmup: {diagnosis['true_bottleneck_after_warmup']}")
    print(f"  Huge max latency from cold start/lazy load: {diagnosis['cold_start_lazy_load_causing_max_latency']}")
    print(f"  Motion ROI cap still beneficial: {diagnosis['motion_roi_cap_still_beneficial']}")
    print(f"  Safe without person/face caps: {diagnosis['safe_to_continue_without_person_face_caps']}")
    print(f"  Next recommended action: {diagnosis['recommended_next_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
