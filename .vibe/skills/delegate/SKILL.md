---
name: delegate
description: Load this skill when you need to delegate work to a subagent — code review, implementation, exploration, or a quick consultation. Covers all available subagents for NonEquilibriumGreenFunction.jl, their capabilities, when to use each, and the exact task() call syntax. Also load when deciding which subagent to spawn for a given task.
---

# Subagent delegation for NonEquilibriumGreenFunction.jl

This skill documents the subagents available via the `task` tool and how to
call them. Subagents run independently, return a result to you, and cannot
interact with the user. Use them to parallelize work, get a second
perspective, or hand off specialized tasks.

## Available subagents

### explore (built-in, read-only)

Codebase exploration. Read-only: `read_file`, `grep`, and `skill` only.

```
task(agent="explore", task="Find every use of solve_dyson in src/ and test/, and summarize which kernel combinations each call site uses")
```

Use for: searching the codebase, understanding structure, reading files
without polluting your context. Cannot edit anything.

### code-review (Mistral Medium, read-only)

Diff review focused on correctness, style, and test coverage.

```
task(agent="code-review", task="Review the unstaged diff: check that the new kernel product in src/Kernels/kernel_algebra.jl preserves the causality classes, and verify the @testitem coverage in test/test_kernels.jl")
```

Use for: a structured code review pass on a diff or commit range. Returns
findings grouped by blocking / suggestion / nitpick. Cannot edit files.

### implement-mechanical (Mistral Small, write)

Boilerplate, simple edits, test additions, mechanical refactors.

```
task(agent="implement-mechanical", task="Add a @testitem to test/test_kernels.jl exercising blockindex/blockrange consistency for bs=1, 2, 4")
```

Use for: straightforward edits where the approach is already decided —
adding tests, formatting, renaming, simple function additions, updating the
export list in src/NonEquilibriumGreenFunction.jl. Escalates to
implement-challenging when it hits algorithmic judgment.

### implement-challenging (Mistral Medium, write)

Algorithm changes, multi-file refactors, kernel/compression/solver work,
performance-sensitive edits.

```
task(agent="implement-challenging", task="Add an HSS recompression step to solve_dyson after every Volterra iteration, keeping the causality structure intact; update test_kernels.jl and run the full test suite")
```

Use for: non-trivial implementation that requires algorithmic judgment,
multi-file changes, or numerical kernel work. Can delegate exploration to
`explore`.

### large-consult (GLM 5.3, read-only)

Single-step consultation: one focused question, one direct answer. No
multi-step exploration, no bash, no nested delegation. Read-only
(`read_file`, `grep` only).

```
task(agent="large-consult", task="Is the product of an advanced and a retarded kernel acausal, and does kernel_algebra.jl implement that correctly for stationary vs two-time operands?")
```

Use for: a quick independent gut-check on a specific mathematical claim,
design decision, or correctness question. Do not use for multi-step
analysis or exploration.

## How to choose

| You need to... | Use |
|---|---|
| Search or read code without bloating context | `explore` |
| Get a structured code review of a diff | `code-review` |
| Hand off a simple, well-specified edit | `implement-mechanical` |
| Hand off algorithmic or multi-file work | `implement-challenging` |
| Get a quick second opinion on one question | `large-consult` |

## Calling syntax

All subagents are spawned via the `task` tool:

```
task(agent="<name>", task="<description of the work>")
```

The `task` parameter is a self-contained description — the subagent sees only
this text, not your conversation history. Include everything it needs: file
paths, the diff range to look at, the specific question, any constraints.

## Guidelines

- Give the subagent a complete, self-contained task description. It does not
  share your context.
- Do not delegate if the task is trivial enough to do yourself in one step.
- Do not delegate to multiple write-capable subagents on the same files
  concurrently — they may conflict.
- `large-consult` is read-only and single-step: give it one question, not a
  multi-part task.
- When you delegate implementation, check the result afterward with
  `git diff` or by reading the changed files, and re-run the test suite
  (`julia --project=@. -e 'using Pkg; Pkg.test()'`) if the change touches
  logic.
