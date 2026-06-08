# Image Processing Service Pre-Implementation Runtime Review

## Purpose

This discussion reviews the current IPS design as a realtime multi-camera runtime before implementation begins.

Scope:
- documentation review only
- no code changes
- no spec rewrite
- no architecture change unless a critical issue requires clarification

Reviewed sources:
- `doc/image_processing_service/image_processing_service.md`
- `doc/image_processing_service/frame_ingestion_gateway.md`
- `doc/image_processing_service/RecognitionPipelineManager.md`
- `doc/image_processing_service/shared_contracts.md`
- `doc/image_processing_service/frame_transformation_layer.md`
- `doc/system.md`

## Already Clear in the Specs

- IPS is defined as the sole top-level runtime and lifecycle owner.
- Queue ownership is IPS-only; queues are internal and not Gateway- or RPM-owned.
- Gateway is specified as ingestion logic only, not a runtime-thread owner.
- RPM is specified as processing logic only, not a queue or runtime owner.
- Per-camera execution lanes are the intended runtime shape.
- Realtime freshness is explicitly favored over completeness through bounded queues, stale dropping, and DROP_OLDEST support.
- `queue_wait_ms` is explicitly defined as queue residence time only and excludes processing time.

## Approved Refinements Captured in Specs

The following runtime refinements are now documented in the module specs and system overview:

- Atomic STOPPING is the authoritative enqueue gate.
- `stop(drain=false)` drops queued frames immediately, while `stop(drain=true)` drains fresh queued frames until timeout with stale policy still active.
- IPS is the sole owner of service-wide runtime health escalation; Gateway and RPM expose local symptoms only.
- Deterministic queue fairness now includes per-camera limits first, reserved per-camera queue bytes, and global queued-byte limits second.
- Queue memory is bounded, queue accounting excludes derived processing buffers, and deep frame history retention is prohibited.
- Per-frame ROI fan-out is capped and excess ROIs are dropped deterministically.
- Freshness checks now cover dequeue-time, post-dequeue pre-processing, and shutdown drain validation.
- The canonical rejection taxonomy is shared across Gateway and IPS.
- Cold-start observability, non-blocking metrics, rate-limited logging, and lane-local failure containment are documented.

# Shutdown Gate Is Not Operationally Atomic
Severity: CRITICAL

Problem:
The shutdown order says IPS stops ingestion execution first and closes enqueue acceptance second, but it does not define a single atomic transition that prevents in-flight `sink.enqueue(...)` calls from racing with queue shutdown and worker stop sequencing.

Why it matters for realtime/runtime:
This leaves enqueue-after-stop behavior nondeterministic. Frames can be accepted, rejected, or partially accounted for depending on timing, which is unsafe for shutdown correctness and incident debugging.

Suggested spec adjustment:
Define one authoritative STOPPING transition point, state exactly when `ServiceFramePacketSink.enqueue(...)` must begin rejecting, and define how in-flight enqueue calls behave during the transition.

# Drain Versus Drop Shutdown Behavior Is Undefined
Severity: CRITICAL

Problem:
`stop(drain)` exists, and the shutdown order says to "drain/drop queues according to policy," but the specs do not define what `drain=true` or `drain=false` means in operational terms. The interaction with stale-frame dropping, overflow policy, in-flight frame processing, and shutdown deadlines is not specified.

Why it matters for realtime/runtime:
The service cannot implement deterministic shutdown without a precise drain contract. Under load, one implementation may continue processing stale frames while another drops everything immediately.

Suggested spec adjustment:
Define `drain=true` and `drain=false` behavior explicitly, including treatment of queued stale frames, in-flight frames, maximum drain deadline, and what happens when the deadline expires.

# Timeout And Degraded-State Authority Is Split
Severity: MAJOR

Problem:
IPS says long processing duration in a service-managed worker moves the service to DEGRADED, while Gateway says stop-timeout behavior for IPS-managed ingestion workers moves Gateway health to DEGRADED or ERROR. The authoritative owner for timeout detection, escalation, and health reporting is not explicit.

Why it matters for realtime/runtime:
Operational health becomes ambiguous. A single lane failure can produce conflicting degraded signals, duplicate alerts, or disagreement about whether the service or only ingestion is unhealthy.

Suggested spec adjustment:
Make IPS the authoritative owner of runtime timeout and degraded-state escalation. Define Gateway health as local symptom reporting only, or explicitly define the mapping from Gateway-local failure to IPS service health.

# Global Fairness Rule Has No Deterministic Enforcement Model
Severity: MAJOR

Problem:
The specs say global queued-byte limits must not allow one camera to starve others, but they do not define how fairness is enforced when `max_total_queued_bytes` is hit across cameras with different frame sizes and arrival rates.

Why it matters for realtime/runtime:
The system claims deterministic queueing and fairness, but the behavior under mixed-camera pressure is still implementation-defined. This is a direct multi-camera runtime risk.

Suggested spec adjustment:
Define a deterministic global-memory arbitration rule such as per-camera reserve, quota, or explicit cross-camera rejection precedence when global byte pressure is reached.

# Queue Memory Limits Do Not Bound Total Runtime Memory
Severity: MAJOR

Problem:
Queue limits bound queued `FramePacket` residency, but the FTL always converts accepted frames to `np.ndarray`, retains CURRENT and PREVIOUS per camera, and RPM may create multiple derived ROI images for a single dequeued frame. The specs do not define total IPS memory budgeting across queues, FTL state, and derived processing buffers.

Why it matters for realtime/runtime:
IPS can remain "queue bounded" while overall process memory still grows with camera count, resolution, and per-frame fan-out. That is a bounded-queue design, not yet a bounded-memory design.

Suggested spec adjustment:
Add an explicit service-level memory model. Either define total memory accounting across queue residency, FTL residency, and derived buffers, or explicitly declare a non-goal with required deployment sizing constraints.

# RPM Per-Frame Fan-Out Can Amplify Backlog
Severity: MAJOR

Problem:
One dequeued frame can trigger FTL ingest, two full-frame lookups for motion, and then a variable number of ROI lookups for motion regions, persons, and faces. The specs do not define any per-frame work budget, admission cap, or overload degradation rule for long frames.

Why it matters for realtime/runtime:
This creates backlog amplification. A single expensive frame can consume far more than one worker timeslice, which undermines realtime freshness even if queue policy is otherwise correct.

Suggested spec adjustment:
Define per-frame processing budget rules, maximum ROI fan-out safeguards, and the degradation or drop behavior when a frame exceeds its runtime budget.

# Freshness Checks Are Too Narrow For Overload Conditions
Severity: MAJOR

Problem:
Stale-frame dropping is specified at dequeue time, but not for shutdown drain, not for long-running in-flight processing, and not for frames that become stale after dequeue but before RPM completes.

Why it matters for realtime/runtime:
The service claims freshness-first behavior, yet it may still spend CPU on work that is no longer realtime-relevant once overload or shutdown starts.

Suggested spec adjustment:
Define all freshness checkpoints explicitly: enqueue, dequeue, post-dequeue pre-processing, and shutdown drain. State whether stale in-flight work may continue, be abandoned, or must finish.

# Shared Resource Contention Is Acknowledged But Not Controlled
Severity: MAJOR

Problem:
The specs say shared downstream components should avoid coarse global locks, but they do not define stronger runtime constraints for FTL storage access, model runtime contention, result handling, or metrics emission in a multi-camera deployment.

Why it matters for realtime/runtime:
Per-camera workers alone do not guarantee lane isolation if shared components serialize work internally. One slow or noisy camera can still degrade others through hidden shared-resource bottlenecks.

Suggested spec adjustment:
Add explicit non-functional constraints that shared components must avoid cross-camera blocking on the hot path, and identify which serialization points are allowed versus prohibited.

# Enqueue Rejection Semantics Are Not Operationally Precise Enough
Severity: MEDIUM

Problem:
Gateway increments `sink_enqueue_rejected_total` for STOPPING rejection, queue backpressure, sink unavailability, and other rejection cases. IPS has more detailed internal reasons, but the end-to-end rejection taxonomy is not required to stay aligned across Gateway and IPS.

Why it matters for realtime/runtime:
Operators cannot reliably distinguish normal shutdown, overload, boundary violation, or sink failure from one another. That makes realtime incidents harder to diagnose and tune.

Suggested spec adjustment:
Define a stable rejection reason taxonomy shared across Gateway and IPS metrics, with explicit codes for STOPPING, capacity rejection, backpressure, sink unavailable, and boundary violation.

# Cold Start Is Operationally Silent
Severity: MEDIUM

Problem:
RPM treats missing PREVIOUS as a normal cold start and returns an empty `PipelineResult` immediately, but IPS metrics do not define whether this counts as processed, skipped, or warmup-only work. No explicit warmup visibility is required per camera.

Why it matters for realtime/runtime:
The first-frame behavior per camera becomes hard to interpret. Throughput, dequeue, and detection counters can look inconsistent during startup or camera recovery.

Suggested spec adjustment:
Add explicit cold-start counters and define whether cold-start frames count as processed, skipped, or a separate warmup category.

# Metrics And Logging Constraints Are Incomplete
Severity: MAJOR

Problem:
The specs define useful queue and worker metrics, but they do not require metrics collection to be non-blocking, do not require logging to be rate-limited, and do not constrain observability overhead on the realtime hot path.

Why it matters for realtime/runtime:
Under overload or repeated failures, metrics and logging can become part of the latency problem. A realtime design needs observability guarantees that do not destabilize the runtime it is trying to diagnose.

Suggested spec adjustment:
Add explicit requirements that metrics emission is non-blocking, degraded-path logging is rate-limited, and health/metrics collection must not serialize enqueue or dequeue hot paths.

# Per-Camera Failure Containment Is Not Fully Defined
Severity: MAJOR

Problem:
The specs clearly isolate per-camera queues and workers, but they do not define how repeated per-camera failures are contained once a lane becomes chronically slow, repeatedly stale, or repeatedly timing out against shared processing resources.

Why it matters for realtime/runtime:
Without explicit containment rules, one bad camera can still destabilize the service through repeated retries, repeated degraded transitions, or shared-resource pressure, even though the lane model looks isolated on paper.

Suggested spec adjustment:
Define lane-local degradation behavior, backoff or quarantine behavior for repeated failures, and the criteria for escalating from camera-local degradation to service-wide degradation.

## Runtime-Readiness Summary

The IPS specs are directionally strong on ownership boundaries, queue ownership, and per-camera lane intent. They are not yet strong enough on shutdown determinism, overload control, total-memory bounding, and operational observability to support a confident realtime implementation.

## Remaining Blocking Risks

- non-atomic STOPPING and enqueue race behavior
- undefined drain-versus-drop shutdown contract
- missing deterministic fairness rule under global byte pressure
- no service-level memory bound across queues, FTL state, and derived buffers
- no explicit per-frame processing budget for RPM fan-out
- incomplete degraded-state and rejection observability

## Recommended Spec Fixes Before Implementation

- Define one atomic shutdown gate and exact enqueue rejection timing.
- Specify `stop(drain)` semantics, deadlines, and stale/in-flight frame handling.
- Define global-memory fairness and total IPS memory accounting expectations.
- Add explicit per-frame budget and overload degradation rules for RPM processing.
- Make degraded-state ownership and timeout authority unambiguous.
- Add non-blocking metrics, rate-limited logging, and camera-local failure containment requirements.

## Final Assessment

IPS is not blocked by architectural direction. The approved runtime-detail gaps have been captured in the module specs and system overview, leaving only deployment-specific sizing choices such as exact timeout and capacity values.

APPROVED REFINEMENTS CAPTURED