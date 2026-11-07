# Verify the public behavior

Choose tests covering the reported API, inputs, return values, exceptions and boundary conditions. Get the absolute repository directory with `pwd`. Invoke `scripts/issue_verification.py` through `run_skill_script`:

`--workspace /absolute/repository --test tests/test_module.py --source-root package --module package --timeout-seconds SECONDS`

Repeat `--test`, `--source-root` and `--module` as needed. `--module` names the Python public API package or module the selected tests must actually import. For a src layout use the real implementation directory and import name, not an installed package elsewhere. The script observes imports inside the same pytest process and does not redirect the tested API. Choose SECONDS within the remaining shared time, retaining the configured submission reserve. Never repeat the same check manually.

Inspect JSON `status`, `test_exit_code`, test counts, `failures`, `source_paths`, `origins`, `missing_modules` and `unexpected_origins`. A timeout, zero tests, skipped-only tests, absent implementation diff, missing import or import outside the selected implementation roots is not successful verification. Source changes can include templates and other non-Python implementation files; test-only and reproduction-only changes do not count. Pass narrow implementation roots to exclude unrelated files.

For a missing suitable test, create a temporary pytest file containing assertions against the public API. Run it with the same script and report the coverage gap. A successful script establishes execution mechanics and provenance; decide separately whether the checks cover the issue. Do not weaken tests, install dependencies, alter runner settings or convert caught exceptions into success.
