# User Guide

## Overview
`app_autogen.py` provides a Streamlit UI to run an AutoGen-powered stock research workflow.

## Steps
1. Set your API key in `.streamlit/secrets.toml` or env.
2. Start the app:
   ```
   streamlit run app_autogen.py
   ```
3. Enter a research query (e.g., "Comprehensive analysis on AAPL").
4. Click Run. Watch intermediate steps in the progress panel.
5. Read the final report in the output area.

## Tips
- If `$` appears incorrectly, it’s Streamlit’s Markdown LaTeX; use escaping in text.
- Longer queries may take 1–3 minutes.

## Error Messages
- Authentication errors: verify `OPENAI_API_KEY`.
- Tool errors: shown in progress area; try again or simplify the query.
