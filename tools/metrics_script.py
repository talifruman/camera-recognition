import json
import os
import glob
from collections import Counter

def get_data(base_path, od_cap, fd_cap, fr_cap):
    summary_path = os.path.join(base_path, "summary_all_frames.json")
    if not os.path.exists(summary_path):
        return None
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    metrics = {
        "frames": 0,
        "total_ftl": 0,
        "stages": [{"total": 0, "frames": 0} for _ in range(5)],
        "raw_counts": 0,
        "filter_counts": 0,
        "merge_counts": 0,
        "persistence_counts": 0,
        "od_requests": 0,
        "fd_requests": 0,
        "fr_requests": 0,
        "od_caps": 0,
        "fd_caps": 0,
        "fr_caps": 0,
        "temporal_suppressed": 0,
        "camera_shake": 0,
        "gmc_applied": 0,
        "persons_in_output": 0,
        "recognized_faces": 0,
        "frames_meaningful_od": 0,
        "empty_scene_fp": 0,
        "recognized_names": set(),
        "frame_details": []
    }

    for frame in summary:
        metrics["frames"] += 1
        rpm = frame.get("rpm_metrics", {})
        debug_dir = frame.get("debug_dir", "")
        video_name = frame.get("video_name", "")
        frame_id = frame.get("frame_id")
        
        ftl = rpm.get("total_ftl_calls", 0)
        metrics["total_ftl"] += ftl
        
        stages = rpm.get("stages", [])
        for i, s in enumerate(stages[:5]):
            metrics["stages"][i]["total"] += s
            metrics["stages"][i]["frames"] += 1
            
        # Parse 02_motion_debug.json
        motion_path = os.path.join(debug_dir, "02_motion_debug.json")
        if os.path.exists(motion_path):
            with open(motion_path, 'r') as f:
                motion = json.load(f)
                p_count = motion.get("motion_bboxes_after_persistence_count", 0)
                metrics["persistence_counts"] += p_count
                
                raw = motion.get("motion_bboxes_raw_count", 0)
                metrics["raw_counts"] += raw
                metrics["filter_counts"] += motion.get("motion_bboxes_after_filter_count", 0)
                metrics["merge_counts"] += motion.get("motion_bboxes_after_merge_count", 0)
                
                if motion.get("camera_shake_detected"): metrics["camera_shake"] += 1
                if motion.get("global_motion_compensation_applied"): metrics["gmc_applied"] += 1
                if raw > 0 and p_count == 0: metrics["temporal_suppressed"] += 1

        # Parse OD/FD
        od_req = rpm.get("od_requests", 0)
        fd_req = rpm.get("fd_requests", 0)
        fr_req = rpm.get("fr_requests", 0)
        
        metrics["od_requests"] += od_req
        metrics["fd_requests"] += fd_req
        metrics["fr_requests"] += fr_req
        
        if od_req >= od_cap: metrics["od_caps"] += 1
        if fd_req >= fd_cap: metrics["fd_caps"] += 1
        if fr_req >= fr_cap: metrics["fr_caps"] += 1

        # Parse 03_obj_det_output.json
        od_path = os.path.join(debug_dir, "03_obj_det_output.json")
        persons = 0
        if os.path.exists(od_path):
            with open(od_path, 'r') as f:
                od_data = json.load(f)
                # Estimate persons from detections
                dets = od_data.get("detections", [])
                persons = sum(1 for d in dets if d.get("class_name") == "person" or d.get("label") == "person")
                metrics["persons_in_output"] += persons
        
        if od_req > 0 and persons > 0:
            metrics["frames_meaningful_od"] += 1
        if "empty_scene" in video_name and persons > 0:
            metrics["empty_scene_fp"] += 1

        # Parse metadata.json
        meta_path = os.path.join(debug_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                rec_faces = meta.get("recognized_faces", [])
                metrics["recognized_faces"] += len(rec_faces)
                for face in rec_faces:
                    name = face.get("name")
                    if name and name.upper() != "UNKNOWN":
                        metrics["recognized_names"].add(name)

        metrics["frame_details"].append({
            "video": video_name,
            "frame_id": frame_id,
            "ftl": ftl,
            "debug_dir": debug_dir
        })

    return metrics

before_path = r"c:\Users\talif\Desktop\Camera-regogintion\debug_outputs\rpm_real_step_4_before_updated"
after_path = r"c:\Users\talif\Desktop\Camera-regogintion\debug_outputs\rpm_real_step_4_after_updated"

data_b = get_data(before_path, 1000, 1000, 1000)
data_a = get_data(after_path, 8, 16, 32)

def summarize(d):
    if not d: return {}
    n = d["frames"] or 1
    return {
        "frames": d["frames"],
        "mean_total_ftl_calls_per_frame": d["total_ftl"] / n,
        "stage_totals": [s["total"] for s in d["stages"]],
        "stage_means": [s["total"] / (s["frames"] or 1) for s in d["stages"]],
        "mean_raw_count": d["raw_counts"] / n,
        "mean_filter_count": d["filter_counts"] / n,
        "mean_merge_count": d["merge_counts"] / n,
        "mean_persistence_count": d["persistence_counts"] / n,
        "mean_od_requests": d["od_requests"] / n,
        "mean_fd_requests": d["fd_requests"] / n,
        "mean_fr_requests": d["fr_requests"] / n,
        "cap_triggers": {"od": d["od_caps"], "fd": d["fd_caps"], "fr": d["fr_caps"]},
        "temporal_suppressed_frames": d["temporal_suppressed"],
        "camera_shake_triggered_frames": d["camera_shake"],
        "gmc_applied_frames": d["gmc_applied"],
        "persons_in_output_total": d["persons_in_output"],
        "recognized_faces_total": d["recognized_faces"],
        "frames_with_meaningful_od": d["frames_meaningful_od"],
        "empty_scene_false_positive_frames": d["empty_scene_fp"],
        "names_non_unknown_count": len(d["recognized_names"]),
        "distinct_names": sorted(list(d["recognized_names"]))
    }

output = {
    "before": summarize(data_b),
    "after": summarize(data_a)
}

# Top 10 hotspots by FTL
if data_b:
    sorted_b = sorted(data_b["frame_details"], key=lambda x: x["ftl"], reverse=True)
    output["before"]["top10_hotspots_by_ftl"] = sorted_b[:10]
if data_a:
    sorted_a = sorted(data_a["frame_details"], key=lambda x: x["ftl"], reverse=True)
    output["after"]["top10_hotspots_by_ftl"] = sorted_a[:10]

# Top 5 overlay pairs
if data_b and data_a:
    lookup_b = {(f["video"], f["frame_id"]): f for f in data_b["frame_details"]}
    deltas = []
    for f_a in data_a["frame_details"]:
        key = (f_a["video"], f_a["frame_id"])
        if key in lookup_b:
            f_b = lookup_b[key]
            delta = abs(f_a["ftl"] - f_b["ftl"])
            deltas.append({
                "video": f_a["video"],
                "frame_id": f_a["frame_id"],
                "abs_ftl_delta": delta,
                "before_overlay": os.path.join(f_b["debug_dir"], "00_overlay.jpg"),
                "after_overlay": os.path.join(f_a["debug_dir"], "00_overlay.jpg")
            })
    output["top5_overlay_pairs_by_abs_ftl_delta"] = sorted(deltas, key=lambda x: x["abs_ftl_delta"], reverse=True)[:5]

print(json.dumps(output, indent=2))
