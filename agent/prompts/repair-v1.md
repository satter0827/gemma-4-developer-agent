Resolve the original issue within the shared session budget. Another agent will independently verify the real diff before submitting it.

Original issue:
{problem_description}

Identify the public interface, concrete input, expected result, and exact boundary behavior. Read the implementation and its nearest tests together. Locate the cause with bounded searches and implement a small source change early. Use the current tool schema exactly; after a tool error, inspect its required arguments and reread the target before retrying. Scratch scripts do not repair the implementation.

Run a focused existing test and a reproduction with assertions at the public interface. A failed assertion must produce a nonzero exit code. Never catch an exception merely to print it and report success. Do not install packages, use the network, weaken tests, change test runner settings, or modify protected tests.

Use get_status to track the shared time and tool budget. Finish with a short handoff: cause, implementation paths changed, checks and their actual outcomes, remaining uncertainty. Leave enough budget for verification. Do not submit the patch or delegate to another agent.
