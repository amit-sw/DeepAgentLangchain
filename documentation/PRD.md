# Product Requirements Document (PRD)

## Title
Migrate `app_autogen.py` from LangChain Agents to Microsoft AutoGen

## Objective
Replace the LangChain agent implementation in `app_autogen.py` with Microsoft AutoGen while maintaining feature parity and the existing Streamlit UX.

## Background
The project currently has two entry points:
- `app.py`: LangChain-based agents
- `app_autogen.py`: Intended alternative implementation
This PRD focuses exclusively on `app_autogen.py` to leverage AutoGen’s multi-agent orchestration.

## In Scope
- Use Microsoft AutoGen to implement a research assistant that:
  - Accepts a research query
  - Uses tools (e.g., price lookup via `yfinance`, simple web query stub, summarization via OpenAI models)
  - Streams intermediate updates to the UI
  - Produces a final report string
- Secrets: continue reading from `.streamlit/secrets.toml` and env vars (no breaking changes)
- Logging: structured, leveled, configurable
- Minimal new deps; document any additions and versions

## Out of Scope
- Major UI redesign in `app.py`
- Cloud deployment changes beyond documenting run instructions

## Users and Use Cases
- Researcher inputs a stock/company query and gets a synthesized report.
- Wants visibility into intermediate steps (tools used, brief summaries) while the agent runs.

## UX Requirements
- Input: text area for query (reuse current `app_autogen.py` UI pattern)
- Feedback: progress container showing agent/tool messages
- Output: final report text area
- Clear error messages when failures occur

## Functional Requirements
- Run button triggers AutoGen conversation
- Stream intermediate events to `progress_area` as they occur
- Stop on error with user-readable message

## Non-Functional Requirements
- Configurable logging level/destination
- Works locally via `streamlit run app_autogen.py`
- Reasonable performance (end-to-end < 2–3 min typical query)

## Success Metrics
- Parity with existing behavior (same inputs, similar outputs)
- No secret/config regressions
- Tests of basic flows pass manually

## Dependencies
- Microsoft AutoGen (pyautogen)
- OpenAI-compatible client (already present); confirm compatibility
- `yfinance`, `streamlit`, `toml`

## Acceptance Criteria
- Demonstrated run of `streamlit run app_autogen.py` with working AutoGen multi-agent flow
- Intermediate streaming and final report visible
- Updated docs: Requirements, PRD, EDD, Deployment, User Guide
