# PROJECT_RULES

## Purpose
This document defines the permanent engineering rules and development contract for the entire project.

These rules are the default standard for all future work, including design decisions, code changes, new files, architecture proposals, and implementation guidance.

## Rule Precedence
When guidance overlaps with other project guidance files, this document is the primary source of truth for engineering and development behavior.

## Conversation Startup Requirement
- At the start of every new development conversation, the AI assistant must read this file before proposing design, code, architecture, or implementation steps.
- If this file is updated, the AI assistant must use the latest version as the active contract.
- No implementation action should begin until this file has been read for the current conversation.

## 1. Development Workflow
- Before starting implementation, define the design first.
- First understand the requirement, then define architecture and module boundaries, and only then implement.
- Do not jump directly into code before the structure is clear.
- For every non-trivial feature, briefly describe the design before implementation.

## 2. Simplicity and Readability
- Code must be simple, readable, and maintainable.
- Avoid unnecessary complexity, cleverness, and over-engineering.
- Prefer explicit and clear code over compact but hard-to-read code.
- Keep logic easy to follow.
- Use meaningful names for classes, methods, variables, and modules.

## 3. Small Focused Modules
- Each module should have one clear responsibility.
- Do not create large modules that handle unrelated concerns.
- Follow the Single Responsibility Principle.
- Split logic into small, focused, cohesive components.
- Keep boundaries between modules explicit and clean.

## 4. Small Focused Functions
- Functions should be short and focused.
- A function should ideally not exceed 50 lines.
- If a function becomes too large, split it into smaller helper functions.
- Each function should do one thing well.
- Avoid deeply nested logic when possible.

## 5. Clear Interfaces and Loose Coupling
- Define clear interfaces between components.
- Prefer dependency on abstractions rather than concrete implementations.
- Avoid tight coupling between modules.
- Design components so they are replaceable with minimal impact on other components.

## 6. Dependency Injection and Testability
- Design modules using Dependency Injection.
- Dependencies should be injected, not hardcoded inside classes.
- Make every important module testable in isolation.
- Ensure the architecture supports mocking and stubbing dependencies in unit tests.
- Prefer constructor injection unless there is a strong reason not to.
- Avoid hidden global dependencies.

## 7. Unit Testing Mindset
- Write code in a way that supports unit testing.
- Separate business logic from infrastructure, UI, and framework concerns.
- Keep modules independently testable.
- Design for mocks, fakes, and test doubles where needed.
- Avoid designs that make isolated testing difficult.

## 8. Separation of Concerns
- Separate business logic, infrastructure, UI, persistence, transport, and configuration concerns.
- Do not mix orchestration, business rules, and low-level implementation in the same place.
- Keep layers clean and responsibilities clearly separated.

## 9. Maintainability Rules
- Prefer extensible design, but do not over-engineer for hypothetical future needs.
- Refactor when code becomes too large, too coupled, or hard to test.
- Avoid duplication when practical, but do not force abstractions too early.
- Keep files and classes at a manageable size.

## 10. Implementation Behavior for the AI Assistant
- At the beginning of every conversation, read this file first and align all actions to it.
- Before implementing, always align with the rules in this file.
- If a requested implementation would violate these rules, propose a cleaner design first.
- When generating code, briefly explain how the design follows these rules.
- When appropriate, suggest refactoring if existing code violates these rules.
- Do not produce code that is overly large, tightly coupled, or difficult to test unless explicitly requested.

## 11. Output Expectations
- All new code should follow these rules by default.
- All architectural suggestions should be evaluated against simplicity, modularity, readability, and testability.
- Prefer designs that support mocking and isolated unit tests.

## Pre-Implementation Checklist
- Did I define the design first?
- Is each module focused on one responsibility?
- Are functions short and readable?
- Are dependencies injected?
- Can this module be unit tested in isolation?
- Are interfaces clear and boundaries clean?
- Is the code simple rather than clever?
