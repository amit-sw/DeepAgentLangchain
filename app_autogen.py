import os
from pathlib import Path
import toml
import logging
import streamlit as st
import yfinance as yf
import json
import asyncio
import threading
import time
from typing import Dict, List, Tuple

from autogen_agentchat.agents import AssistantAgent

# tools (instead of register_function)
from autogen_core.tools import FunctionTool  # optional; you can also pass plain functions
from autogen_ext.models.openai import OpenAIChatCompletionClient

def load_secrets():
    """Load secrets from .streamlit/secrets.toml and set as environment variables."""
    secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    
    if not secrets_path.exists():
        print(f"Warning: {secrets_path} not found. Using environment variables only.")
        return
    
    try:
        secrets = toml.load(secrets_path)
        set_keys = []
        for key, value in secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ[key.upper()] = str(value)
                set_keys.append(key.upper())
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    env_key = f"{key.upper()}_{subkey.upper()}"
                    os.environ[env_key] = str(subvalue)
                    set_keys.append(env_key)
        #if set_keys:
        #    print(f"Loaded secrets for: {', '.join(sorted(set_keys))}")
    except Exception as e:
        print(f"Error loading secrets: {e}")


load_secrets()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def get_stock_price(symbol: str) -> str:
    """Get current stock price and basic information."""
    logging.info(f"[TOOL] Fetching stock price for: {symbol}")
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1d")
        if hist.empty:
            logging.error("No historical data found")
            return json.dumps({"error": f"Could not retrieve data for {symbol}"})
            
        current_price = hist['Close'].iloc[-1]
        result = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "company_name": info.get('longName', symbol),
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "52_week_high": info.get('fiftyTwoWeekHigh', 0),
            "52_week_low": info.get('fiftyTwoWeekLow', 0)
        }
        logging.info(f"[TOOL RESULT] {result}")
        return json.dumps(result, indent=2)

    except Exception as e:
        logging.exception("Exception in get_stock_price")
        return json.dumps({"error": str(e)})

def get_financial_statements(symbol: str) -> str:
    """Retrieve key financial statement data."""
    try:
        stock = yf.Ticker(symbol)
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        
        latest_year = financials.columns[0]
        
        return json.dumps({
            "symbol": symbol,
            "period": str(latest_year.year),
            "revenue": float(financials.loc['Total Revenue', latest_year]) if 'Total Revenue' in financials.index else 'N/A',
            "net_income": float(financials.loc['Net Income', latest_year]) if 'Net Income' in financials.index else 'N/A',
            "total_assets": float(balance_sheet.loc['Total Assets', latest_year]) if 'Total Assets' in balance_sheet.index else 'N/A',
            "total_debt": float(balance_sheet.loc['Total Debt', latest_year]) if 'Total Debt' in balance_sheet.index else 'N/A'
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

def get_technical_indicators(symbol: str, period: str = "3mo") -> str:
    """Calculate key technical indicators."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return f"Error: No historical data for {symbol}"
        
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        latest = hist.iloc[-1]
        latest_rsi = rsi.iloc[-1]
        
        return json.dumps({
            "symbol": symbol,
            "current_price": round(latest['Close'], 2),
            "sma_20": round(latest['SMA_20'], 2),
            "sma_50": round(latest['SMA_50'], 2),
            "rsi": round(latest_rsi, 2),
            "volume": int(latest['Volume']),
            "trend_signal": "bullish" if latest['Close'] > latest['SMA_20'] > latest['SMA_50'] else "bearish"
        }, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


# Sub-agent configurations
fundamental_analyst = {
    "name": "fundamental-analyst",
    "description": "Performs deep fundamental analysis of companies including financial ratios, growth metrics, and valuation",
    "prompt": """You are an expert fundamental analyst with 15+ years of experience. 
    Focus on:
    - Financial statement analysis
    - Ratio analysis (P/E, P/B, ROE, ROA, Debt-to-Equity)
    - Growth metrics and trends
    - Industry comparisons
    - Intrinsic value calculations
    Always provide specific numbers and cite your sources."""
}

technical_analyst = {
    "name": "technical-analyst", 
    "description": "Analyzes price patterns, technical indicators, and trading signals",
    "prompt": """You are a professional technical analyst specializing in chart analysis and trading signals.
    Focus on:
    - Price action and trend analysis
    - Technical indicators (RSI, MACD, Moving Averages)
    - Support and resistance levels
    - Volume analysis
    - Entry/exit recommendations
    Provide specific price levels and timeframes for your recommendations."""
}

risk_analyst = {
    "name": "risk-analyst",
    "description": "Evaluates investment risks and provides risk assessment",
    "prompt": """You are a risk management specialist focused on identifying and quantifying investment risks.
    Focus on:
    - Market risk analysis
    - Company-specific risks
    - Sector and industry risks
    - Liquidity and credit risks
    - Regulatory and compliance risks
    Always quantify risks where possible and suggest mitigation strategies."""
}

subagents = [fundamental_analyst, technical_analyst, risk_analyst]


# Main research instructions
DEFAULT_RESEARCH_INSTRUCTIONS = """You are an elite stock research analyst with access to multiple specialized tools and sub-agents. 

Your research process should be systematic and comprehensive:

1. **Initial Data Gathering**: Start by collecting basic stock information, price data, and recent news
2. **Fundamental Analysis**: Deep dive into financial statements, ratios, and company fundamentals
3. **Technical Analysis**: Analyze price patterns, trends, and technical indicators
4. **Risk Assessment**: Identify and evaluate potential risks
5. **Competitive Analysis**: Compare with industry peers when relevant
6. **Synthesis**: Combine all findings into a coherent investment thesis
7. **Recommendation**: Provide clear buy/sell/hold recommendation with price targets

Always:
- Use specific data and numbers to support your analysis
- Cite your sources and methodology
- Consider multiple perspectives and potential scenarios
- Provide actionable insights and concrete recommendations
- Structure your final report professionally

When using sub-agents, provide them with specific, focused tasks and incorporate their specialized insights into your overall analysis."""


xxx_DEFAULT_RESEARCH_INSTRUCTIONS = """
This is a demo. Just say that per-share price is 100.
""" 



def run_stock_research(query: str, progress_area, instructions: str):
    """Run AutoGen agents and stream intermediate messages to the UI."""

    system_prompt = instructions.strip()
    print(f"[run_stock_research] OPENAI_MODEL: {OPENAI_MODEL}, System prompt: {system_prompt}")

    model_client = OpenAIChatCompletionClient(
        model=OPENAI_MODEL,
        api_key=os.getenv("OPENAI_API_KEY", "")
    )

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message=system_prompt,
    )

    # Attach tools directly (v0.7 API)
    assistant.tools = [
        get_stock_price,
        get_financial_statements,
        get_technical_indicators,
    ]

    # Execute the task with the new run API
    async def _do_run():
        return await assistant.run(task=query)

    try:
        result = asyncio.run(_do_run())
    except RuntimeError:
        # In case there's already an event loop (e.g., certain Streamlit configs), fall back
        result = asyncio.get_event_loop().run_until_complete(_do_run())

    final_text = []
    # Try to extract a text payload from the result; fall back to string
    try:
        content = getattr(result, "content", None)
        if not content and isinstance(result, dict):
            content = result.get("content")
        if not content:
            content = str(result)
        final_text.append(content)
    except Exception:
        final_text.append(str(result))

    return "\n\n".join([t for t in final_text if t])

st.set_page_config(page_title="Stock Research Agent (Autogen)", page_icon="📊", layout="centered")
st.title("Stock Research Agent (Autogen)")
research_instructions = st.text_area("Research Instructions", height=160, value=DEFAULT_RESEARCH_INSTRUCTIONS)    

query = st.text_area("Research Query", height=160, placeholder="Example: Comprehensive analysis on Apple Inc. (AAPL)")
if st.button("Run Analysis"):
    with st.spinner("Running analysis...", show_time=True):
        progress_area = st.container()
        output = run_stock_research(query, progress_area, research_instructions)
        st.text_area("# Research Report", value=output, height=600)
