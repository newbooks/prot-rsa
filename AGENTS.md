# Repository Ground Rules

These instructions apply to the entire repository.

## Branch safety

- Do not create, modify, delete, or commit files while the checked-out branch is `main` unless the user explicitly permits edits on `main` for the current task.
- Read-only inspection on `main` is allowed.
- If an edit is requested without permission to edit `main`, stop and ask the user to switch to or authorize work on a non-`main` branch.

## Implementation and testing

- Keep the application in a single Python script that can both run independently as a command-line program and be imported as a module. Expose its reusable functions and constants directly without triggering command-line execution or other side effects on import.
- For every new function or module, add or update relevant `pytest` tests.
- Before finishing the task, run the focused tests for the changed behavior and, when practical, the full test suite.
- Do not report the work as complete unless the tests pass. If a test cannot be run or an existing failure remains, report it clearly with the reason and scope.

## Compute resources

- Use multiprocessing whenever the workload can be safely and meaningfully parallelized, with a default of 4 worker processes. Allow callers and command-line users to override the worker count.
- Use GPU acceleration when the required operation and available dependencies support it. Detect support safely and retain a correct CPU fallback for environments without a compatible GPU.
- Keep results consistent across serial, multiprocessing, CPU, and GPU execution within the scientifically appropriate numerical tolerance.

## Issue resolution and scientific ambiguity

- Resolve issues discovered while performing the requested work when they are clearly related, reproducible, and within scope.
- Prefer fixing root causes over suppressing symptoms, and add regression tests for corrected defects when practical.
- If a decision depends on scientifically ambiguous assumptions, definitions, reference values, models, or validation criteria, stop before encoding the assumption and ask the user for direction. State the ambiguity, the available evidence, and the consequences of the plausible choices.

## Public PyPI release quality

- Treat this repository as a public PyPI package: preserve a stable public API where practical and avoid unnecessary breaking changes.
- Keep package metadata, dependencies, supported Python versions, documentation, type information, and user-facing errors accurate when affected by a change.
- Do not introduce secrets, private paths, unpublished data, environment-specific assumptions, or dependencies that cannot be installed from public package sources.
- Favor portable behavior and ensure source distributions and wheels can be built cleanly when packaging is changed.
- Record user-visible changes and clearly flag any unavoidable compatibility or release implications.
