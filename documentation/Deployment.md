# Deployment Guide

## Prerequisites
- Python 3.10+
- Virtual environment activated
- Secrets available via `.streamlit/secrets.toml` or environment variables
  - `OPENAI_API_KEY`
  - Optional: `OPENAI_API_BASE`, `OPENAI_MODEL`

## Install Dependencies
After code lands, ensure requirements include AutoGen and related libs. For now:

```
pip install -r requirements.txt
# If AutoGen not present yet (will be added with code change):
pip install pyautogen
```

## Run Locally
```
streamlit run app_autogen.py
```

## Configuration
- `.streamlit/secrets.toml` example:
```
[default]
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
```
- Or set env vars before running.

## Logging
- Control via env var `LOG_LEVEL` (DEBUG, INFO, WARNING, ERROR).
- Logs emit to console; redirect as needed with standard tooling.

## Troubleshooting
- Missing API key: ensure it’s set in secrets or env.
- Network errors: retry or verify connectivity.
- `$` rendered as LaTeX: ensure UI escaping or use `st.text`/`st.code`.
