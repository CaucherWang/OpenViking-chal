---
name: openviking-rule-search
description: Check applicable rules with openviking_rule_search. Use when an agent must validate a planned action or response against task-specific constraints.
metadata:
  vikingbot:
    always: true
---

# OpenViking Rule Search

Before acting or answering, call `openviking_rule_search` with a concrete `task_context`.
Describe the user's goal, the planned action or decision, and the known facts that may affect
which rules apply. Include material unknowns instead of guessing them.

Review every returned Rule independently. Apply a Rule when its `applies_when` condition is
supported by the task context and no `exception` applies. Treat Rules as constraints on the
action or response, not as evidence that unobserved facts are true.

Validate the planned action or response against all applicable Rules. Call
`openviking_rule_search` again if new information materially changes the task context.
