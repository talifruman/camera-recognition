# Image Processing Service Runtime Ownership Review

## Purpose

This is a review/discussion document to validate runtime ownership definitions before implementation.

Scope:
- documentation/spec alignment only
- no code changes
- no runtime behavior changes

---

## Intended Runtime Ownership Model (Target)

- Image Processing Service is the only top-level runtime lifecycle owner.
- Image Processing Service owns and manages queues.
- Image Processing Service starts, stops, and supervises two managed runtime flows per camera:
  - ingestion flow (Frame Ingestion Gateway behavior executed under Image Processing Service lifecycle ownership)
  - RPM processing flow (RecognitionPipelineManager execution under Image Processing Service lifecycle ownership)
- Frame Ingestion Gateway owns ingestion logic only.
- Frame Ingestion Gateway does not independently own runtime threads.
- Image Processing Service owns and manages ingestion worker lifecycle, including startup, shutdown, supervision, and per-camera execution management.
- RecognitionPipelineManager owns recognition pipeline logic only.
- RecognitionPipelineManager does not own top-level lifecycle, queue ownership, queue pulling ownership, or worker lifecycle ownership.
- Queues live between ingestion and RPM processing and are owned by Image Processing Service.
- Per-camera lanes (ingestion path + queue + processing path) are owned by Image Processing Service.
- Shutdown is initiated and coordinated by Image Processing Service: stop ingestion, close enqueue acceptance, drain/drop queues by policy, then stop processing workers.

---

## Current Definitions by Document

## 1. Image Processing Service Spec

Source: `doc/image_processing_service/image_processing_service.md`

Currently defines:
- Image Processing Service as lifecycle owner and orchestrator.
- service-owned bounded per-camera queues.
- service-owned sink that receives enqueue from Gateway.
- service-owned processing worker scheduling/lifecycle.
- startup order: start Gateway, then processing workers.
- shutdown order: stop Gateway first, stop accepting frames, stop workers, release queues.

Aligned with target:
- top-level lifecycle ownership is mostly aligned.
- queue ownership is aligned.
- per-camera queue + worker model is aligned.

Gaps/ambiguities:
- sink section still states unknown-camera validation/rejection inside service sink, which overlaps Gateway language.
- service wording should explicitly state ingestion runtime workers are Image Processing Service-managed runtime execution, not Gateway-independent threads.
- shutdown details need stronger single-owner semantics for enqueue gate and timeout governance.

## 2. Frame Ingestion Gateway Spec

Source: `doc/image_processing_service/frame_ingestion_gateway.md`

Currently defines:
- Gateway receives/validates/normalizes/builds FramePacket and publishes to sink.
- Gateway in-scope includes managing transport lifecycle and ingestion workers.
- start() binds transport and starts per-camera ingestion workers.
- stop() stops transport, waits for workers, honors stop_timeout_ms.
- Gateway out-of-scope excludes queue ownership and downstream scheduling.

Aligned with target:
- queue ownership is explicitly outside Gateway.
- downstream orchestration is outside Gateway.

Conflicting/overreaching implications:
- wording implies Gateway owns ingestion worker runtime lifecycle directly.
- wording can be read as Gateway independently owning runtime threads.
- unknown-camera rejection ownership appears duplicated with service sink language.

## 3. RecognitionPipelineManager Spec

Source: `doc/image_processing_service/RecognitionPipelineManager.md`

Currently defines:
- RPM owns internal pipeline orchestration logic only.
- RPM explicitly does not own transport, ingestion, system threads, external lifecycle.
- RPM does not pull frames; caller supplies FramePacket.
- RPM does not own queue/workers/top-level runtime.

Alignment:
- strongly aligned with target architecture.

Potential doc-hardening opportunity:
- explicitly name Image Processing Service as canonical runtime caller/owner in system integration sections to remove remaining interpretation gaps.

## 4. Shared Contracts

Source: `doc/image_processing_service/shared_contracts.md`

Currently defines:
- FramePacket and FramePacketSink boundary contracts.
- enqueue interface as publication boundary without exposing queue internals.

Alignment:
- compatible with Image Processing Service queue ownership model.

Gap:
- does not define ownership semantics directly; ownership must remain explicit in module specs.

## 5. Runtime/Orchestration Docs

Sources:
- `doc/system.md`
- `FrameIngressTransport.java` (Gateway spec duplicate copy)
- existing discussion docs under `docs/discussions/`

Current state:
- system-level language generally supports unified service ownership direction.
- duplicate Gateway spec copy increases drift risk.
- earlier discussion docs contain mixed readiness/finalization conclusions while ownership wording remains partially ambiguous.

---

## Conflict Review Against Intended Architecture

## Mismatch 1: Gateway independent runtime lifecycle implication

Current wording / implied behavior:
- Gateway spec says Gateway manages ingestion workers and starts/stops them.
- can be read as independent runtime lifecycle ownership.

Why wrong or risky:
- conflicts with single top-level lifecycle owner model.
- creates split ownership for startup/shutdown/supervision.
- increases risk of inconsistent stop behavior and unclear incident ownership.

Required documentation change:
- reword Gateway ownership to ingestion behavior/logic + transport adapter behavior only.
- explicitly state ingestion runtime worker lifecycle is Image Processing Service-owned.

Fix before implementation:
- Yes.

## Mismatch 2: Gateway-managed threads outside Image Processing Service ownership implication

Current wording / implied behavior:
- Gateway start()/stop() wording suggests Gateway owns runtime threads directly.

Why wrong or risky:
- contradicts target that Gateway does not independently own runtime threads.
- weakens per-camera lane ownership under Image Processing Service.

Required documentation change:
- add explicit sentence in Gateway spec:
  - Frame Ingestion Gateway does not independently own runtime threads.
  - It defines ingestion behavior/logic.
  - Image Processing Service owns actual runtime execution lifecycle.

Fix before implementation:
- Yes.

## Mismatch 3: RPM owns queue pulling or worker lifecycle implication

Current wording / implied behavior:
- RPM spec mostly rejects this; no major direct conflict found.

Why wrong or risky (if left implicit elsewhere):
- if integration docs are loose, teams may bypass service ownership boundary.

Required documentation change:
- keep RPM spec as-is for core ownership.
- add cross-reference in service/runtime docs that RPM is a managed processing component invoked under Image Processing Service lifecycle ownership.

Fix before implementation:
- Yes (documentation clarity gate).

## Mismatch 4: Image Processing Service as sink-only, not ingestion execution manager

Current wording / implied behavior:
- some language emphasizes sink boundary but does not always state ingestion execution management clearly.

Why wrong or risky:
- can be interpreted as service only exposing enqueue sink while Gateway independently runs ingestion runtime.

Required documentation change:
- add explicit ownership statement in Image Processing Service spec:
  - Image Processing Service owns and manages ingestion worker lifecycle, startup, shutdown, supervision, and per-camera execution.

Fix before implementation:
- Yes.

## Mismatch 5: Queue ownership outside Image Processing Service implication

Current wording / implied behavior:
- queue ownership is mostly clear in service spec.
- overlap remains through duplicated unknown-camera rejection path between Gateway and service sink.

Why wrong or risky:
- duplicated reject path obscures boundary and metric ownership.

Required documentation change:
- define single primary owner for unknown-camera rejection.
- keep service-side unknown-camera path as defensive boundary violation handling only (or remove in normal path wording).

Fix before implementation:
- Yes.

## Mismatch 6: Shutdown ownership split across components

Current wording / implied behavior:
- service shutdown order exists but ownership details for enqueue closure, drain/drop enforcement, and timeout authority are not fully explicit.

Why wrong or risky:
- can produce race conditions and inconsistent shutdown states.
- unclear final authority over degraded/error transition.

Required documentation change:
- specify Image Processing Service as single shutdown coordinator.
- define ordered shutdown contract:
  1. stop ingestion execution
  2. close enqueue acceptance
  3. drain/drop queues by policy
  4. stop RPM workers
- define timeout governance ownership under service lifecycle.

Fix before implementation:
- Yes.

---

## Correct Target Wording

Use the following wording consistently across specs:

- Image Processing Service is the only top-level lifecycle owner.
- Frame Ingestion Gateway is a managed ingestion component inside the Image Processing Service runtime.
- Frame Ingestion Gateway owns ingestion logic only, while Image Processing Service owns and manages the ingestion worker lifecycle, including startup, shutdown, supervision, and per-camera execution management.
- Frame Ingestion Gateway does not independently own runtime threads. It defines ingestion behavior/logic, while Image Processing Service owns the actual runtime execution lifecycle.
- RecognitionPipelineManager is a managed processing component inside the Image Processing Service runtime.
- RecognitionPipelineManager owns pipeline logic only and does not own top-level lifecycle or queue lifecycle.
- Queues between ingestion and RPM processing are owned by Image Processing Service.
- Per-camera lanes are owned by Image Processing Service (ingestion execution path + queue + RPM processing execution path).
- Shutdown is coordinated by Image Processing Service in this order:
  1. stop ingestion execution
  2. close enqueue acceptance
  3. drain/drop queues per policy
  4. stop RPM workers

---

## Required Spec Changes Before Implementation

## A. `doc/image_processing_service/frame_ingestion_gateway.md`

Required updates:
- replace/clarify worker lifecycle wording to avoid independent runtime ownership implication.
- keep Gateway ownership at ingestion behavior/logic + transport behavior boundary.
- add explicit statement that runtime thread lifecycle is service-owned.
- align startup/shutdown language as managed under Image Processing Service lifecycle authority.

## B. `doc/image_processing_service/image_processing_service.md`

Required updates:
- add explicit ownership statement that Image Processing Service manages ingestion runtime worker lifecycle.
- strengthen single-owner shutdown sequence and enqueue acceptance closure wording.
- clarify per-camera execution lane ownership end-to-end.
- resolve overlap with Gateway unknown-camera rejection wording.

## C. `doc/image_processing_service/RecognitionPipelineManager.md`

Required updates:
- keep current ownership boundaries.
- add integration note that RPM runtime execution is service-managed in deployed runtime.

## D. `doc/system.md`

Required updates:
- ensure top-level service orchestration text reflects service-owned per-camera lane runtime lifecycle.

## E. `FrameIngressTransport.java` (duplicate spec copy)

Required updates:
- either align wording to canonical Gateway spec immediately or deprecate as duplicate source.
- avoid conflicting ownership language across two Gateway spec artifacts.

## F. `doc/image_processing_service/shared_contracts.md`

Required updates:
- no contract shape change required.
- optionally add note that runtime ownership is defined by module specs, not sink contract type.

---

## Open Questions Before Implementation

1. Unknown-camera rejection single-owner rule:
   - Gateway-only primary enforcement, with service defensive metric path only?

2. Shutdown timeout governance:
   - one service-level timeout authority coordinating component-local timeouts?

3. Queue drain/drop policy defaults during shutdown:
   - default drain vs default drop by mode (realtime vs non-realtime)?

4. Supervision semantics for ingestion workers:
   - restart policy, escalation thresholds, degraded/error transition ownership under service lifecycle.

5. Duplicate Gateway spec governance:
   - which file is canonical for implementation and review acceptance?

---

## Do Not Implement Yet

This document is review/discussion-only.

Do not implement code changes yet.
Do not modify runtime behavior yet.
Do not refactor runtime ownership behavior yet.

Implementation work should start only after the above spec wording changes are agreed and merged.