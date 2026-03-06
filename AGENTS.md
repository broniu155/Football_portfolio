# Football Portfolio – Repository Workflow and Engineering Rules

## Project Purpose
This repository contains a football analytics portfolio application built in Streamlit.
Its purpose is to provide clear, tactically useful, analyst-friendly workflows for match reporting, event analysis, and football insight generation.

The application should support:
- reliable football data processing
- interpretable analytics
- coach-friendly and analyst-friendly visualizations
- maintainable and modular development
- safe iteration without breaking existing functionality

## Core Working Principles
- Prioritize football usefulness over technical novelty.
- Prefer small, modular, reviewable changes over broad rewrites.
- Reuse existing utilities, loaders, and plotting patterns where possible.
- Keep football logic interpretable, especially in first-release implementations.
- Preserve existing app behavior unless a change is explicitly requested.
- Make the most important tactical information visible quickly.
- Prefer fewer, clearer visuals over cluttered screens.
- Handle empty, sparse, or imperfect data gracefully.

## Workflow Governance
- The Task Orchestrator Agent must coordinate all non-trivial work.
- Planning must complete before implementation begins.
- Implementation must complete before QA begins.
- QA must validate the feature before any cleanup starts.
- Cleanup must be followed by regression QA.
- Final output must be one consolidated report, not disconnected fragments.
- Only involve agents relevant to the task.

## Standard Agent Order for Feature Work
Use this order unless the task is explicitly review-only or cleanup-only:

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

## Review-Only Workflow
Use this order for feature review or planning-only tasks:

1. Product Manager
2. Technical Lead
3. Coach UX Reviewer
4. QA
5. Coach Feature Scout

No implementation should happen in this workflow.

## Cleanup-Only Workflow
Use this order for maintenance, redundancy removal, or post-feature cleanup:

1. Technical Lead
2. QA
3. Code Hygiene and Refactor Agent
4. QA Regression Pass
5. Release Manager

No new feature logic should be added in this workflow.

## Planning Requirements
Before any implementation:
- define the problem being solved
- define the user story
- define clear acceptance criteria
- define minimum viable scope
- identify out-of-scope items for the first release
- identify risks, dependencies, and verification steps
- list planned files to inspect and likely files to change

## Implementation Requirements
Before editing files:
- list the intended files to modify
- explain why each file needs to change
- reuse existing modules where possible
- avoid unnecessary refactors
- preserve stable public interfaces unless explicitly approved
- keep new logic modular and testable

During implementation:
- maintain compatibility with existing app behavior
- document assumptions clearly
- keep football classifications and metrics understandable
- prefer rule-based logic for initial versions unless a stable reusable abstraction already exists

## QA and Validation Requirements
Every implemented feature must include:
- feature validation against acceptance criteria
- edge case testing
- empty-state testing
- regression checks for impacted existing features
- clear pass/fail reporting
- reproducible test or validation steps where possible

## Cleanup and Refactor Rules
Cleanup is allowed only after implementation QA is complete.

Safe cleanup includes:
- removal of unused imports
- removal of duplicated logic
- removal of obsolete helper functions
- removal of deprecated code paths replaced by validated logic
- simplification of unnecessary branching where behavior is unchanged

Cleanup must not:
- change functional behavior intentionally
- rewrite large modules without explicit instruction
- remove code when dependency impact is unclear
- proceed when test coverage or behavior is uncertain

If risk is unclear, leave the code in place and document the concern instead.

## UX and Football Review Rules
Features should be reviewed from a football coach and analyst perspective.

A strong feature should:
- answer a real football question
- reduce effort needed to find insight
- use football language that feels natural
- support match analysis, opponent analysis, or training preparation
- make tactical patterns easier to see, not harder

## Documentation Requirements
When relevant, update:
- README
- CHANGELOG
- manual verification notes

Documentation should explain:
- what changed
- how to use it
- how to verify it
- any known limitations

## Definition of Done
Work is considered complete only when:
- scope and acceptance criteria were defined
- implementation is finished
- QA has validated the feature
- cleanup was completed or explicitly deferred
- regression QA passed after cleanup
- documentation was updated where needed
- final output includes one consolidated summary

## Final Response Format
The final response for non-trivial work must include:
- workflow decision
- scope and acceptance criteria
- implementation summary
- files changed
- tests and validation run
- cleanup actions performed
- regression status
- documentation updates
- manual verification steps
- suggested future features where relevant