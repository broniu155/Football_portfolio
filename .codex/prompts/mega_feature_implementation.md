Use the repository workflow defined in AGENTS.md, all agent files, and all runbooks.

The Task Orchestrator Agent must coordinate the full workflow.

Read:
- AGENTS.md
- all files in .codex/agents/
- all files in .codex/runbooks/

Task:
<PASTE FEATURE REQUEST HERE>

Instructions to Task Orchestrator:
1. Decide which agents are required for this task.
2. Enforce the correct execution order.
3. Require gated handoffs between planning, implementation, QA, cleanup, regression testing, and release.
4. Keep the workflow modular and safe.
5. Return one consolidated final report.

Default expected workflow for feature implementation:
1. Product Manager
2. Technical Lead
3. Coach UX Reviewer
4. Data Engineer
5. Analytics Engineer
6. Streamlit Engineer
7. QA
8. Code Hygiene and Refactor Agent
9. QA Regression Pass
10. Release Manager
11. Coach Feature Scout

Global rules:
- do not start implementation before planning is complete
- do not start cleanup before QA is complete
- do not finalize release before regression QA is complete
- keep changes modular
- avoid large refactors
- list planned file changes before editing
- if cleanup risk is unclear, document rather than delete
- preserve existing app functionality

Final response must include:
- workflow decision
- scope and acceptance criteria
- implementation summary
- files changed
- tests run
- cleanup actions
- regression status
- documentation changes
- suggested future features