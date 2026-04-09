# Module Spec Prompt Toolkit

A collection of reusable prompts for creating, editing, reviewing, and refining module specification documents using the Markdown Module Spec Skill.

---

## 1. Create Full Module Specification

Generate a complete module spec from scratch.

```text
Write a full module specification using the Markdown Module Spec Skill.

Module name: {MODULE_NAME}

Requirements:
- Follow all mandatory sections
- Keep strict module isolation
- Ensure API is engine-agnostic
- Use processing engine abstraction (AI or algorithmic)
- Include acceptance/filtering logic if relevant
- Include all diagrams (class + sequence + data flow)
- Follow conciseness rules for Error Handling, Metrics, Lifecycle, Internal Data Structures

Context:
{SHORT DESCRIPTION OF MODULE}
```

---

## 2. Write Specific Sections

### 2.1 Input Section

Generate only the Input section.

```text
Write the Input section for this module according to the Markdown Module Spec Skill.

Module: {MODULE_NAME}

Context:
{WHAT THE MODULE RECEIVES}

Requirements:
- Include Responsibility Boundary, Structure, Contract, Validation Rules, Semantics
- Use strict typing
- Do not include preprocessing steps that belong upstream
```

### 2.2 Processing Engine

Describe engine abstraction and replaceability.

```text
Write the Processing Engine section.

Module: {MODULE_NAME}

Context:
{MODEL OR ALGORITHM USED}

Requirements:
- Define abstraction interface
- Specify if AI-based or algorithmic
- Emphasize replaceability
- Do not leak engine-specific details to public API
```

### 2.3 Acceptance / Filtering Logic

Define filtering rules and thresholds.

```text
Write the Acceptance / Filtering Logic section.

Module: {MODULE_NAME}

Context:
{HOW RESULTS SHOULD BE FILTERED}

Requirements:
- Define thresholds and rules
- Specify where filtering happens
- Ensure only accepted results reach output
```

---

## 3. Rewrite Existing Document

Improve structure and enforce the standard.

```text
Rewrite the following module specification to fully comply with the Markdown Module Spec Skill.

Requirements:
- Preserve meaning, improve structure
- Enforce all mandatory sections
- Fix API to be engine-agnostic
- Ensure no internal data leaks
- Add missing sections if needed
- Improve clarity and precision

Document:
{PASTE YOUR MD FILE}
```

---

## 4. Review Module Specification

Find issues and provide concrete fixes.

```text
Review this module specification against the Markdown Module Spec Skill.

Focus on:
- Missing sections
- Violations of module isolation
- API stability issues
- Internal data leakage
- Missing acceptance/filtering logic
- Incorrect use of processing engine abstraction
- Non-functional requirements

Provide:
- List of issues
- Specific fixes (not general advice)

Document:
{PASTE FILE}
```

---

## 5. Targeted Improvements (Non-Destructive)

Improve only what is broken.

```text
Update this module specification according to the Markdown Module Spec Skill.

Make ONLY targeted improvements:
- Fix violations
- Add missing sections
- Improve clarity

Do NOT rewrite everything.
Do NOT change correct parts.

Document:
{PASTE}
```
