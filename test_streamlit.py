import streamlit as st

text1="The assistant fetched  stock prices: NVIDIA (NVDA) at $180.45 and Cisco (CSCO) at $66.20. It also updated its todo list to mark the price fetches as completed and the summary as in progress."
text2=text1.replace("$", "\\$")

st.code(f"{text1=}")
st.write(f"{text2=}")
obj2="""
{'agent': {'messages': [AIMessage(content='NVIDIA (NVDA)\n- Price: $180.45\n- Market cap: $4.4007T\n- P/E: 58.21\n- 52-week range: 
$86.62 – $184.48\n\nCisco Systems (CSCO)\n- Price: $66.20\n- Market cap: $262.15B\n- P/E: 25.36\n- 52-week range: 
$47.85 – $72.55\n\nWould you like intraday charts, recent news, analyst ratings, or comparisons (e.g., 
% change from 52-week low/high)?', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 
'model_name': 'gpt-5-mini-2025-08-07', 'service_tier': 'default'}, id='run--4dae24e0-0a26-4efa-9ab6-a98ef0f92138')]}}")
"""

