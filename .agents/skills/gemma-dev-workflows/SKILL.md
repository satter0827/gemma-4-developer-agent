---
name: gemma-dev-workflows
description: Prepare Gemma comparison candidates, diagnose saved patches, analyze evaluation evidence, or prepare candidate registration in this repository. Use for these development workflows, including recovery from a saved workflow record.
---

# Gemma development workflows

Read the repository AGENTS.md and [workflow operations](../../../docs/operations/workflows.md). Load the linked detailed guide only for the selected task.

Choose the matching `gemma-dev workflow` action: `prepare` obtains CPU evidence for a frozen experiment; `diagnose` replays saved patches and controls; `report` validates saved results, optionally from one fixed Notebook version; `candidate` verifies CPU/GPU evidence and the saved archive to prepare registration.

Use `--json`. Read `status`, `stage`, `artifacts`, `reason_codes`, and `next_actions`; pass returned paths directly to subsequent operations. CPU verification does not establish model accuracy, comparison completion, or submission acceptance.

For `prepare` and `diagnose`, use `--resume` with the saved workflow record and unchanged inputs. `status --run` validates retained evidence. Changed evidence requires investigation; do not replace hashes or search for a convenient recent output.

The services own checks, budgets, and artifact formats. Do not duplicate their logic in shell scripts. A stale development package requires `make setup`; source changes require the repository's quality checks. Do not run tests concurrently against one checkout's shared temporary directories.

GPU startup, formal submission, state registration, and scheduled submission remain separate explicit actions. Respect existing user authorization without repeated confirmation. Do not infer authorization from successful CPU preparation or retry remote operations with unknown receipts.

Keep competition data, patches, logs, archives, and credentials under private artifacts. Public records contain validated metrics and explicit review decisions. Preserve incomplete and unknown outcomes.
