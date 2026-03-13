# Media Service Merge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge Clip Recording and Storage into a unified Media Service in documentation, reducing streaming overhead while preserving future extraction seams.

**Architecture:** Replace the external Clip Recording -> Storage hop with a single Media Service that contains internal clip pipeline and storage adapter modules. Keep Frame Buffer, Event Service, and Image processing service boundaries unchanged, and retain event topic compatibility.

**Tech Stack:** Markdown documentation, Mermaid sequence diagrams, architecture decision records in system.md.

---

### Task 1: Add Plan and Scope Baseline

**Files:**
- Create: `docs/plans/2026-03-12-media-service-merge.md`
- Modify: `doc/system.md`

**Step 1: Confirm baseline architecture references**

Search for references to Clip Recording and Storage boundaries in `doc/system.md`.

**Step 2: Record merge decision in system document**

Add a concise architecture decision note in `doc/system.md` stating that Clip Recording and Storage are merged into Media Service.

**Step 3: Verify references remain consistent**

Check that new wording does not conflict with remaining sections.

**Step 4: Commit checkpoint**

```bash
git add docs/plans/2026-03-12-media-service-merge.md doc/system.md
git commit -m "docs: add media service merge implementation plan"
```

### Task 2: Update Architecture and Data Flow

**Files:**
- Modify: `doc/system.md`

**Step 1: Update Full System Architecture section**

Replace service boundary statements so Media Service owns clip assembly, mp4 muxing, and media write operations.

**Step 2: Update Service Diagram section**

Replace Clip Recording and Storage boxes with a single Media Service box, including internal module bullets.

**Step 3: Update Data Flow and sequence diagrams**

Edit event-triggered flow so Event Service triggers Media Service, which fetches frames from Frame Buffer and emits `event.clip.ready` after persistence.

**Step 4: Run consistency verification**

Confirm all major architecture and flow sections reference Media Service consistently.

**Step 5: Commit checkpoint**

```bash
git add doc/system.md
git commit -m "docs: merge clip recording and storage into media service"
```

### Task 3: Update API and Executable Surface

**Files:**
- Modify: `doc/system.md`
- Modify: `README.md`

**Step 1: Update Service APIs section**

Rename Clip Recording API responsibilities to Media Service responsibilities and keep Frame Buffer API references intact.

**Step 2: Update executable list**

Replace `clip-recording-service.exe` and `storage-service.exe` with `media-service.exe`.

**Step 3: Update README summary**

Add a short architecture summary noting merged Media Service and current event-triggered clip flow.

**Step 4: Verify doc links and terminology**

Ensure names are stable and no stale executable names remain.

**Step 5: Commit checkpoint**

```bash
git add doc/system.md README.md
git commit -m "docs: refresh topology summary for media service"
```

### Task 4: Verification and Handoff

**Files:**
- Modify: `doc/system.md` (if needed)

**Step 1: Run final text checks**

Search for stale references to old service split and confirm intended mentions remain only in historical comparisons.

**Step 2: Review full diff for clarity**

Inspect markdown diff to ensure concise, non-duplicated wording and consistent terminology.

**Step 3: Record what remains open**

List any unresolved operational decisions (idempotency policy, storage consistency mode, rollback flag design).

**Step 4: Commit final docs update**

```bash
git add doc/system.md README.md docs/plans/2026-03-12-media-service-merge.md
git commit -m "docs: finalize media service merge architecture updates"
```
