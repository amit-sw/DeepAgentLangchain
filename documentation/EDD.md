# Engineering Design Document (EDD)

## Title
AutoGen-based Multi-Agent Migration for `app_autogen.py`

## Overview
Replace LangChain agent orchestration in `app_autogen.py` with Microsoft AutoGen to drive a stock research workflow with tool usage, intermediate streaming to Streamlit, and a final report.

## Architecture
- Agents (AutoGen):
  - Assistant agent: main reasoning/planning agent using OpenAI-compatible chat model.
  - Tool agent: executes Python tools: price lookup (`yfinance`), simple web/info fetch stubs, summarization helper.
  - (Optional) User proxy: to mediate between Streamlit UI and the assistant for message injection and streaming.
- UI (`app_autogen.py`):
  - Streamlit components: input text area, progress container, final report area.
  - Streams intermediate messages from AutoGen via callbacks.
- Config/Secrets:
  - `load_secrets()` reads `.streamlit/secrets.toml` and environment.
  - Keys: `OPENAI_API_KEY` (and any others) populated into env.
- Logging:
  - Python `logging` with level from env var `LOG_LEVEL` (default INFO).
  - Format includes timestamp, level, module.

## Data Flow
1. User enters query in Streamlit.
2. UI creates AutoGen agents and kicks off a conversation.
3. Assistant plans; delegates tool calls to Tool agent via function/tool calling.
4. Each message/tool result is streamed to `progress_area.write(...)` (escaping `$` for Streamlit Markdown if needed).
5. Assistant compiles final report.
6. UI displays final report.

## Tools
- `get_stock_price(symbol: str)` using `yfinance.Ticker(symbol).info` with safe access.
- `search_news(query: str)` placeholder returning stub or using a safe public endpoint (optional for MVP).
- `summarize(text: str, model)` uses chat model with short prompt.

## Error Handling
- Wrap agent run in try/except.
- On exceptions, log with `logging.exception` and show user-friendly error in UI.
- Timeouts for tool calls (e.g., network) with fallbacks.

## Streaming Strategy
- Use AutoGen event hooks or conversation loop to capture each message sent/received.
- Immediately write to `progress_area` with minimal formatting; escape `$` to avoid LaTeX.

## Configuration
- Environment variables loaded in `load_secrets()`.
- Model name from env: `OPENAI_MODEL` (default `gpt-4o-mini` or project default).

## Dependencies
- `pyautogen` (Microsoft AutoGen)
- `streamlit`, `toml`, `yfinance`, `openai` (or `openai>=1.0` compatible client used by AutoGen)

## Testing Plan
- Manual run: `streamlit run app_autogen.py` and exercise a few queries.
- Validate: intermediate streaming, final report, error paths (no key, network fail).

## Rollout / Backout
- This change is isolated to `app_autogen.py`.
- Backout by reverting to prior version if issues arise.

## Open Questions
- Whether to include a web search tool initially or keep MVP to prices + summarization.
- Exact model names matching existing project constraints.
