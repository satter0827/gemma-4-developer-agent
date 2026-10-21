---
name: python-verification-v1
description: Run focused Python tests and check that implementation changes exist, with explicit exit codes, collected counts and bounded failure summaries.
---

# Verify a Python implementation

Read the original issue and identify the public entry point and expected behavior. Inspect the real source diff. Choose the nearest existing tests and an assertion-based public API reproduction; passing unrelated tests does not establish resolution.

Get the absolute repository directory using `run_command` with `pwd`. Skill scripts execute in a temporary directory, so always pass that repository as `--workspace`. Run `scripts/check.py` with a list of arguments:

`--workspace /absolute/repository --test tests/test_module.py --source-root package --timeout-seconds SECONDS`

Repeat `--test` and `--source-root` as needed. Use `.` for a repository whose package is at the root, or the actual source directory for a src layout. Choose SECONDS below the remaining shared time shown by `get_status` and leave time to inspect failures and submit.

Read the JSON `status`, `test_exit_code`, `collected`, `failed`, `errors`, `source_changed`, and `failures`. A zero-test run, timeout, infrastructure error, or absent implementation change is not successful verification. The surrounding tool may label nonzero script exits as a warning; use the script's JSON and exit information.

Repair failures and rerun the relevant check. Never weaken protected tests, install dependencies, change runner settings or hide exceptions. The script checks execution mechanics; independently verify that the tests cover the reported public behavior and boundary cases. Use `submit_patch` through the normal tool only after inspecting the final implementation.

Execute through `run_skill_script`. The SDK sandbox owns the process group and enforces the shared deadline. Do not detach the helper or its test process from that group.
