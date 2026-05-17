import json
import os
import pandas as pd

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            try:
                data = json.load(f)
                return data
            except:
                return None
    return None

def extract_frames(summary):
    frames_list = []
    if not summary:
        return frames_list
    if isinstance(summary, dict) and "per_video" in summary:
        for video in summary["per_video"]:
            if "frames" in video:
                frames_list.extend(video["frames"])
    if not frames_list and isinstance(summary, dict) and "frames" in summary:
        frames_list = summary["frames"]
    if not frames_list and isinstance(summary, list):
        frames_list = summary
    return frames_list

def analyze_dataset(summary_path):
    summary = load_json(summary_path)
    frames_list = extract_frames(summary)
    
    frames_data = []
    for frame in frames_list:
        if not isinstance(frame, dict):
            continue
        debug_dir = frame.get("debug_dir")
        metrics = frame.get("rpm_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        
        motion_debug = {}
        obj_det = {}
        face_det = {}
        metadata = {}
        
        if debug_dir and os.path.exists(debug_dir):
            m_data = load_json(os.path.join(debug_dir, "02_motion_debug.json"))
            if isinstance(m_data, dict): motion_debug = m_data
            
            o_data = load_json(os.path.join(debug_dir, "03_obj_det_output.json"))
            if isinstance(o_data, dict): obj_det = o_data
            
            f_data = load_json(os.path.join(debug_dir, "04_face_det_output.json"))
            if isinstance(f_data, dict): face_det = f_data
            
            md_data = load_json(os.path.join(debug_dir, "metadata.json"))
            if isinstance(md_data, dict): metadata = md_data
        
        after_persistence = motion_debug.get("motion_bboxes_after_persistence_count", 0)
        raw_motion = motion_debug.get("motion_bboxes_raw_count", 0)
        candidate_persons = len(obj_det.get("candidates", []))
        candidate_faces = len(face_det.get("candidates", []))
        
        rpm_output = metadata.get("rpm_output", {})
        persons = rpm_output.get("persons", []) if isinstance(rpm_output, dict) else []
        recognized_faces_count = 0
        names = []
        for p in persons:
            if isinstance(p, dict):
                for face in p.get("recognized_faces", []):
                    if isinstance(face, dict):
                        name = face.get("person_name", "")
                        if name and name.upper() != "UNKNOWN":
                            names.append(name)
                        recognized_faces_count += 1
        
        od_req = metrics.get("od_roi_requests_count", 0)
        fd_req = metrics.get("fd_roi_requests_count", 0)
        fr_req = metrics.get("fr_roi_requests_count", 0)
        
        cap_motion, cap_person, cap_face = 5, 3, 2
        
        data = {
            "frame_id": frame.get("frame_id"),
            "total_ftl": metrics.get("total_ftl_calls_per_frame", 0),
            "od_req": od_req,
            "fd_req": fd_req,
            "fr_req": fr_req,
            "after_persistence": after_persistence,
            "raw_motion": raw_motion,
            "persons_in_output": len(persons),
            "names": names,
            "is_empty_scene": frame.get("is_empty_scene", False),
            "motion_cap_triggered": after_persistence > cap_motion and od_req == cap_motion,
            "person_cap_triggered": candidate_persons > cap_person and fd_req == cap_person,
            "face_cap_triggered": candidate_faces > cap_face and fr_req == cap_face,
            "temp_suppressed": raw_motion > 0 and after_persistence == 0,
            "debug_dir": debug_dir
        }
        frames_data.append(data)
    
    return summary, frames_data

before_path = r"debug_outputs/rpm_real_step_4_before_updated/summary_all_frames.json"
after_path = r"debug_outputs/rpm_real_step_4_after_updated/summary_all_frames.json"
output_dir = r"debug_outputs/rpm_real_step_4_comparison"
os.makedirs(output_dir, exist_ok=True)
report_path = os.path.join(output_dir, "Release4_audit_summary.md")

s_before, d_before = analyze_dataset(before_path)
s_after, d_after = analyze_dataset(after_path)

if not d_before or not d_after:
    print(f"Error: No frames found. Before: {len(d_before)}, After: {len(d_after)}")
    exit(1)

df_before = pd.DataFrame(d_before)
df_after = pd.DataFrame(d_after)

# Ensure types
for col in ["total_ftl", "od_req", "fd_req", "fr_req", "after_persistence", "raw_motion"]:
    df_before[col] = pd.to_numeric(df_before[col], errors='coerce').fillna(0)
    df_after[col] = pd.to_numeric(df_after[col], errors='coerce').fillna(0)

# Metrics
avg_ftl_before = df_before["total_ftl"].mean()
avg_ftl_after = df_after["total_ftl"].mean()
avg_od_after = df_after["od_req"].mean()
avg_fd_after = df_after["fd_req"].mean()
avg_fr_after = df_after["fr_req"].mean()

cap_motion_count = df_after["motion_cap_triggered"].sum()
cap_person_count = df_after["person_cap_triggered"].sum()
cap_face_count = df_after["face_cap_triggered"].sum()

all_names_after = [n for names in df_after["names"] for n in names]
names_found = len(all_names_after) > 0

# MD
md = f"""# Release 4 Audit Summary

## Key Metrics Comparison
| Metric | Before | After | Delta |
| --- | --- | --- | --- |
| Avg FTL Calls/Frame | {avg_ftl_before:.2f} | {avg_ftl_after:.2f} | {avg_ftl_after - avg_ftl_before:.2f} |
| Avg OD Req | {df_before["od_req"].mean():.2f} | {avg_od_after:.2f} | {avg_od_after - df_before["od_req"].mean():.2f} |
| Avg FD Req | {df_before["fd_req"].mean():.2f} | {avg_fd_after:.2f} | {avg_fd_after - df_before["fd_req"].mean():.2f} |
| Avg FR Req | {df_before["fr_req"].mean():.2f} | {avg_fr_after:.2f} | {avg_fr_after - df_before["fr_req"].mean():.2f} |

## Cap Triggers (After)
- Motion Cap Triggered: {cap_motion_count}
- Person Cap Triggered: {cap_person_count}
- Face Cap Triggered: {cap_face_count}

## Precision & Quality
- Temporal Suppression Count: {df_after["temp_suppressed"].sum()}
- Names Found After: {"Yes" if names_found else "No"}

## Recommended Tuning Values
- Based on observed deltas, if FTL calls are still high, consider reducing `cap_motion` to 4 and `cap_person` to 2.
"""

with open(report_path, "w") as f:
    f.write(md)

print(f"Report: {report_path}")
print(f"Deltas: FTL {avg_ftl_after - avg_ftl_before:.2f}, OD {avg_od_after - df_before['od_req'].mean():.2f}, FD {avg_fd_after - df_before['fd_req'].mean():.2f}, FR {avg_fr_after - df_before['fr_req'].mean():.2f}")
print(f"Caps: Motion {cap_motion_count}, Person {cap_person_count}, Face {cap_face_count}")
print(f"Names Found: {names_found}")
