import json, pathlib, statistics; from collections import Counter, defaultdict
root=pathlib.Path(r'c:/Users/talif/Desktop/Camera-regogintion')
before=root/'debug_outputs/rpm_real_step_4_before_updated/summary_all_frames.json'
after=root/'debug_outputs/rpm_real_step_4_after_updated/summary_all_frames.json'
out_dir=root/'debug_outputs/rpm_real_step_4_comparison'; out_dir.mkdir(parents=True, exist_ok=True)
B=json.loads(before.read_text(encoding='utf-8')); A=json.loads(after.read_text(encoding='utf-8'))

def _read_json(p):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None

def collect(summary, caps):
    frames=[]
    stage_totals=Counter()
    motion_raw=[]; motion_filter=[]; motion_merge=[]; motion_persist=[]
    od=[]; fd=[]; fr=[]; total_ftl=[]
    shake=0; gmc=0; temporal_supp=0
    persons_total=0; rec_faces_total=0
    names=Counter()
    empty_scene_fp=0
    hotspots=[]
    cap_hits=Counter()
    per_video=defaultdict(lambda: {'frames':0,'persons':0,'faces':0,'ftl':0})
    frame_map={}
    for v in summary.get('per_video',[]):
        vname=v.get('video_name','unknown')
        for f in v.get('frames',[]):
            frames.append(f)
            debug_dir=pathlib.Path(f.get('debug_dir',''))
            metrics=f.get('rpm_metrics',{}) or {}
            tf=int(metrics.get('total_ftl_calls_per_frame',0)); total_ftl.append(tf)
            hotspots.append((tf,vname,f.get('frame_id',''),str(debug_dir)))
            for s,val in (metrics.get('ftl_calls_by_stage',{}) or {}).items():
                stage_totals[s]+=int(val)
            raw=int(metrics.get('motion_bboxes_raw_count',0)); af=int(metrics.get('motion_bboxes_after_filter_count',0)); am=int(metrics.get('motion_bboxes_after_merge_count',0))
            motion_raw.append(raw); motion_filter.append(af); motion_merge.append(am)
            odc=int(metrics.get('od_roi_requests_count',0)); fdc=int(metrics.get('fd_roi_requests_count',0)); frc=int(metrics.get('fr_roi_requests_count',0))
            od.append(odc); fd.append(fdc); fr.append(frc)
            md=_read_json(debug_dir/'02_motion_debug.json') or []
            ap=0; any_shake=False; any_gmc=False
            for entry in md:
                if isinstance(entry, dict):
                    ap=max(ap,int(entry.get('motion_bboxes_after_persistence_count',0) or 0))
                    any_shake=any_shake or bool(entry.get('camera_shake_detected',False))
                    any_gmc=any_gmc or bool(entry.get('global_motion_compensation_applied',False))
            motion_persist.append(ap)
            if raw>0 and ap==0: temporal_supp+=1
            if any_shake: shake+=1
            if any_gmc: gmc+=1
            obj=_read_json(debug_dir/'03_obj_det_output.json') or []
            face_det=_read_json(debug_dir/'04_face_det_output.json') or []
            cand_persons=0
            for o in obj:
                if isinstance(o, dict):
                    cand_persons += len(o.get('persons',[]) or [])
            cand_faces=0
            for d in face_det:
                if isinstance(d, dict):
                    cand_faces += len(d.get('detections',[]) or [])
            if ap>caps['motion'] and odc>=caps['motion']: cap_hits['motion']+=1
            if cand_persons>caps['person'] and fdc>=caps['person']: cap_hits['person']+=1
            if cand_faces>caps['face'] and frc>=caps['face']: cap_hits['face']+=1
            persons=int(f.get('persons_in_output',0)); persons_total+=persons
            meta=_read_json(debug_dir/'metadata.json') or {}
            rpm_out=(meta.get('rpm_output',{}) if isinstance(meta,dict) else {})
            rec_faces=0
            for p in rpm_out.get('persons',[]) or []:
                for rf in p.get('recognized_faces',[]) or []:
                    rec_faces += 1
                    name=(rf.get('person_name') or '').strip()
                    if name: names[name]+=1
            rec_faces_total += rec_faces
            if 'empty_scene' in vname and persons>0:
                empty_scene_fp += 1
            per_video[vname]['frames'] += 1
            per_video[vname]['persons'] += persons
            per_video[vname]['faces'] += rec_faces
            per_video[vname]['ftl'] += tf
            frame_map[(vname,f.get('frame_id',''))]=(tf,str(debug_dir))
    n=max(1,len(frames))
    def avg(lst): return round(sum(lst)/len(lst),3) if lst else 0.0
    return {
        'n':len(frames),'avg_total_ftl':avg(total_ftl),'stage_totals':dict(stage_totals),'stage_avg':{k:round(v/n,3) for k,v in stage_totals.items()},
        'avg_raw':avg(motion_raw),'avg_filter':avg(motion_filter),'avg_merge':avg(motion_merge),'avg_persist':avg(motion_persist),
        'avg_od':avg(od),'avg_fd':avg(fd),'avg_fr':avg(fr),
        'temporal_supp':temporal_supp,'shake':shake,'gmc':gmc,'cap_hits':dict(cap_hits),
        'persons_total':persons_total,'rec_faces_total':rec_faces_total,'names':dict(names),
        'empty_scene_fp':empty_scene_fp,'hotspots':sorted(hotspots, reverse=True)[:10],'per_video':per_video,'frame_map':frame_map
    }

bcaps={'motion':1000,'person':1000,'face':1000}
acaps={'motion':8,'person':16,'face':32}
R1=collect(B,bcaps); R2=collect(A,acaps)

pairs=[]
for k,(bftl,bdir) in R1['frame_map'].items():
    if k in R2['frame_map']:
        aftl,adir = R2['frame_map'][k]
        pairs.append((abs(aftl-bftl),k[0],k[1],bftl,aftl,bdir,adir))
pairs=sorted(pairs, reverse=True)[:5]

lines=[]
lines.append('# Release 4 Audit Summary (Updated)')
lines.append('')
lines.append('## Before/After Table')
lines.append('')
lines.append('| Metric | Before | After | Delta |')
lines.append('|---|---:|---:|---:|')
rows=[
 ('Frames',R1['n'],R2['n']),
 ('Avg total_ftl_calls_per_frame',R1['avg_total_ftl'],R2['avg_total_ftl']),
 ('Avg motion_bboxes_raw_count',R1['avg_raw'],R2['avg_raw']),
 ('Avg motion_bboxes_after_filter_count',R1['avg_filter'],R2['avg_filter']),
 ('Avg motion_bboxes_after_merge_count',R1['avg_merge'],R2['avg_merge']),
 ('Avg motion_bboxes_after_persistence_count',R1['avg_persist'],R2['avg_persist']),
 ('Avg od_roi_requests_count',R1['avg_od'],R2['avg_od']),
 ('Avg fd_roi_requests_count',R1['avg_fd'],R2['avg_fd']),
 ('Avg fr_roi_requests_count',R1['avg_fr'],R2['avg_fr']),
 ('Temporal suppression frames (raw>0 && persist=0)',R1['temporal_supp'],R2['temporal_supp']),
 ('Camera shake triggered frames',R1['shake'],R2['shake']),
 ('Global motion compensation applied frames',R1['gmc'],R2['gmc']),
 ('persons_in_output total',R1['persons_total'],R2['persons_total']),
 ('recognized_faces total',R1['rec_faces_total'],R2['rec_faces_total']),
 ('empty_scene frames with persons>0 (false-positive proxy)',R1['empty_scene_fp'],R2['empty_scene_fp']),
]
for name,bv,av in rows:
    d=round(float(av)-float(bv),3)
    lines.append(f'| {name} | {bv} | {av} | {d} |')
lines.append('')
lines.append('## FTL Calls By Stage')
lines.append('')
lines.append('| Stage | Before total | After total | Delta | Before avg/frame | After avg/frame |')
lines.append('|---|---:|---:|---:|---:|---:|')
stages=sorted(set(R1['stage_totals'])|set(R2['stage_totals']))
for s in stages:
    bt=R1['stage_totals'].get(s,0); at=R2['stage_totals'].get(s,0)
    ba=R1['stage_avg'].get(s,0.0); aa=R2['stage_avg'].get(s,0.0)
    lines.append(f'| {s} | {bt} | {at} | {at-bt} | {ba} | {aa} |')
lines.append('')
lines.append('## ROI Caps Triggered')
lines.append('')
lines.append(f'- Before (caps 1000/1000/1000): {R1["cap_hits"]}')
lines.append(f'- After  (caps 8/16/32): {R2["cap_hits"]}')
lines.append('')
lines.append('## Detection Quality Checks')
lines.append('')
lines.append(f'- PersonDirectory names seen before: {sorted(R1["names"].items())[:15]}')
lines.append(f'- PersonDirectory names seen after: {sorted(R2["names"].items())[:15]}')
lines.append("- Per-video totals (after):")
for v,stats in sorted(R2['per_video'].items()):
    lines.append(f"  - {v}: frames={stats['frames']} persons={stats['persons']} faces={stats['faces']} avg_ftl={round(stats['ftl']/max(1,stats['frames']),3)}")
lines.append('')
lines.append('## Remaining FTL Hotspots (After)')
for tf,v,fid,dbg in R2['hotspots']:
    lines.append(f'- {v} / {fid}: total_ftl_calls_per_frame={tf} -> {dbg}')
lines.append('')
lines.append('## Visual Comparison (Top 5 FTL Delta Pairs)')
for d,v,fid,bftl,aftl,bdir,adir in pairs:
    lines.append(f'- {v}/{fid}: before_ftl={bftl}, after_ftl={aftl}, delta={d}')
    lines.append(f'  - Before overlay: {bdir}/06_final_annotated.jpg')
    lines.append(f'  - After overlay:  {adir}/06_final_annotated.jpg')
report=out_dir/'Release4_audit_summary.md'
report.write_text('\n'.join(lines), encoding='utf-8')
print('REPORT', report)
print('KEY_DELTA avg_total_ftl', R1['avg_total_ftl'], '->', R2['avg_total_ftl'])
print('KEY_DELTA avg_od', R1['avg_od'], '->', R2['avg_od'])
print('KEY_DELTA avg_fd', R1['avg_fd'], '->', R2['avg_fd'])
print('KEY_DELTA avg_fr', R1['avg_fr'], '->', R2['avg_fr'])
print('FP_EMPTY_SCENE', R1['empty_scene_fp'], '->', R2['empty_scene_fp'])
print('NAMES_AFTER_COUNT', len(R2['names']))
