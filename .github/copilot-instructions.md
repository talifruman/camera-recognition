# Copilot Instructions

This repository contains design and workflow documentation for the Camera-regogintion project.

## Priorities

- Keep changes small, focused, and easy to review.
- Preserve existing folder structure and naming unless explicitly asked to refactor.
- Update docs when behavior, architecture, or workflow changes.
- Prefer practical, incremental solutions over broad rewrites.

## Documentation Rules

- Put architecture updates in `doc/system.md`.
- Keep markdown concise and scannable.
- Include concrete steps and assumptions when proposing implementation plans.

## Coding Rules

- **Always follow Python Code Conventions** — All Python code creation, updates, and refactoring must comply with `docs/PythonCode/PYTHON_CODE_CONVENTIONS.md`. This is mandatory for every code change.
- Follow existing code style in touched files.
- Avoid introducing new dependencies unless necessary.
- Add brief comments only where logic is not obvious.
- Do not modify unrelated files.

## Python Code Standards

**Required for all Python code:**
- Function length: ≤ 50 lines (excluding docstrings)
- Function/class docstrings: Required for all functions and classes
- Naming: snake_case for functions/vars, PascalCase for classes
- Constants: Define named constants instead of magic numbers (section 12)
- Type hints: Required for all function signatures
- Module docstring: Every Python file must start with module docstring

**Reference:** See `docs/PythonCode/PYTHON_CODE_CONVENTIONS.md` for complete details on all 17 sections covering function design, naming conventions, avoiding magic numbers, documentation standards, and best practices.

## Validation

- Run relevant checks/tests for changed code when possible.
- If checks cannot be run, state what was skipped and why.
