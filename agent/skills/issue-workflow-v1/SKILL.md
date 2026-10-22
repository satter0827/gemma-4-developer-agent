---
name: issue-workflow-v1
description: Resolve a repository issue with bounded exploration, public API checks, focused verification and a retained submitted patch.
---

# Resolve the issue

Use `load_skill_resource` to read `assets/policy.json`. Use its exploration, verification reserve and submission reserve as planning guidance against the remaining time from `get_status`; these are not independent deadlines.

1. Read the original issue, identify the public API and inspect its implementation and nearest tests. Trace inputs, return values, exceptions and relevant boundaries. After exploration call `get_status`. Once the cause and edit location are clear, edit. Stop searches that add no information.
2. Apply the smallest implementation fix. Keep uncertain expectations explicit. Do not infer a return value from a print-only reproduction or hide exceptions. Do not edit protected tests or runner settings.
3. Call `get_status` before verification. Use `load_skill_resource` to read `references/verification.md` and execute the selected tests once through `run_skill_script`. If there is no suitable test, write an assertion-based reproduction using the public API and report what remains unverified. Inspect failures and repair only when supported by the issue and source.
4. Call `get_status` before submitting. Inspect the implementation diff and use `submit_patch`. After a successful submission retain the patch and finish. Do not resubmit to match a file count, stage files, or reset/checkout/revert the changes.
