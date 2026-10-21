Independently verify and complete the implementation within the remaining shared session budget. The preceding agent's claims are unverified evidence.

Original issue:
{problem_description}

Repair handoff:
{repair_summary}

Inspect the actual working-tree diff with run_command, then read the affected public interface. Confirm that source implementation changed; a scratch reproduction alone is insufficient. Check the reported input and expected return value, exact messages, and boundary cases through the public entry point. Run focused tests or an assertion-based reproduction that fails with a nonzero exit code when behavior is wrong. Inspect failures, repair the source if necessary, and rerun the relevant check. Do not treat printed output or zero collected tests as success.

Use tool arguments exactly as declared. Do not install packages, access the network, weaken tests, or alter protected tests or runner settings. Remove scratch files and unrelated edits. Use get_status to reserve enough time to call submit_patch once with the final implementation. State actual checks and remaining uncertainty honestly. Do not transfer control to another agent.
