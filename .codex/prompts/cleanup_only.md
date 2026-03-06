Use the repository workflow defined in AGENTS.md, all agent files, and all runbooks.

The Task Orchestrator Agent must coordinate this cleanup-only workflow.

Read:
- AGENTS.md
- all files in .codex/agents/
- all files in .codex/runbooks/

Task:
Review recently changed modules and safely remove redundant, obsolete, duplicated, or overly complex code while preserving all existing functionality.

Required workflow:
1. Technical Lead
2. QA
3. Code Hygiene and Refactor Agent
4. QA Regression Pass
5. Release Manager

Rules:
- no new features
- no broad rewrites
- preserve behavior
- document risky areas instead of changing them
- final output must be one consolidated report

Final output:
- files reviewed
- cleanup actions taken
- risky areas left untouched
- regression results
- documentation updates if needed
- recommendations for future refactor work