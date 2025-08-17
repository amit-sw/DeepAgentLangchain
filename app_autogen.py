import os
from pathlib import Path
import toml
import logging
import streamlit as st

from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
import yfinance as yf
import logging
from langchain_core.tools import tool
from typing import Dict
import json
import asyncio

# --- Autogen imports ---
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool  # optional; plain functions also work

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

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ArrowStreet"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# 1. Create an OpenAI model
openai_model = ChatOpenAI(
    model="gpt-5-mini", 
    temperature=0,
)


@tool
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

@tool
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

@tool
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

# --- Autogen tool wrappers (plain functions) ---
# The `@tool` decorator above creates LangChain Tool objects. Autogen v0.7 can take
# simple callables, so we wrap to ensure a direct (args -> str) signature.

def stock_price(symbol: str) -> str:
    try:
        # If using LangChain Tool object, call .invoke; else call function directly
        if hasattr(get_stock_price, "invoke"):
            return get_stock_price.invoke({"symbol": symbol})
        return get_stock_price(symbol)  # type: ignore[misc]
    except Exception as e:
        return json.dumps({"error": f"stock_price failed: {e}"})


def financial_statements(symbol: str) -> str:
    try:
        if hasattr(get_financial_statements, "invoke"):
            return get_financial_statements.invoke({"symbol": symbol})
        return get_financial_statements(symbol)  # type: ignore[misc]
    except Exception as e:
        return json.dumps({"error": f"financial_statements failed: {e}"})


def technical_indicators(symbol: str, period: str = "3mo") -> str:
    try:
        if hasattr(get_technical_indicators, "invoke"):
            return get_technical_indicators.invoke({"symbol": symbol, "period": period})
        return get_technical_indicators(symbol, period=period)  # type: ignore[misc]
    except Exception as e:
        return json.dumps({"error": f"technical_indicators failed: {e}"})



def run_stock_research_autogen(query: str, progress_area, instructions: str):
    """Run an Assistant that consults three sub-agents, then synthesizes a report."""

    system_prompt = instructions.strip()
    print(f"[run_stock_research_autogen] OPENAI_MODEL: {OPENAI_MODEL}, System prompt: {system_prompt}")

    # Build a single model client reused by all agents
    model_client = OpenAIChatCompletionClient(
        model=OPENAI_MODEL,
        api_key=os.getenv("OPENAI_API_KEY", "")
    )

    # Main synthesizer assistant
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message=system_prompt,
    )
    assistant.tools = [stock_price, financial_statements, technical_indicators]

    # Helper to construct a sub-agent from a config dict
    def make_subagent(cfg: Dict[str, str]) -> AssistantAgent:
        agent = AssistantAgent(
            name=cfg["name"],
            model_client=model_client,
            system_message=cfg["prompt"].strip(),
        )
        # Attach tools per specialization
        if cfg["name"].startswith("fundamental"):
            agent.tools = [stock_price, financial_statements]
        elif cfg["name"].startswith("technical"):
            agent.tools = [stock_price, technical_indicators]
        else:  # risk
            agent.tools = [stock_price, financial_statements, technical_indicators]
        return agent

    # Instantiate sub-agents from the provided configs
    sub_agents = [make_subagent(cfg) for cfg in subagents]

    # A small wrapper to run a sub-agent with a focused instruction
    async def run_sub(agent: AssistantAgent, subrole_desc: str) -> str:
        task = (
            f"You are the {agent.name}. Work on the following research query: '{query}'.\n"
            f"Deliver a concise section focused on your specialty: {subrole_desc}.\n"
            f"Return a markdown section with a clear heading '## {agent.name.replace('-', ' ').title()}',\n"
            f"specific numbers, brief rationale, and cite sources inline (e.g., [source]).\n"
        )
        result = await agent.run(task=task)
        content = getattr(result, "content", None)
        if not content:
            content = str(result)
        try:
            progress_area.write(f"✅ {agent.name} completed.")
        except Exception:
            pass
        return content

    # Execute sub-agents (sequentially to stream progress reliably in Streamlit)
    import asyncio as _asyncio
    fundamental_out = _asyncio.run(run_sub(sub_agents[0], fundamental_analyst["description"]))
    technical_out = _asyncio.run(run_sub(sub_agents[1], technical_analyst["description"]))
    risk_out = _asyncio.run(run_sub(sub_agents[2], risk_analyst["description"]))

    # Now synthesize a final report using the main assistant
    synthesis_task = (
        "Synthesize a cohesive professional equity research report based on the three sections below.\n"
        "Unify terminology, remove duplicate facts, and resolve any conflicts explicitly.\n"
        "Structure with: Executive Summary, Fundamentals, Technicals, Risks, and Final Recommendation.\n"
        "Keep it concise (700-1000 words) and include a clear price target range if possible.\n\n"
        "--- Fundamental Analysis ---\n" + fundamental_out + "\n\n"
        "--- Technical Analysis ---\n" + technical_out + "\n\n"
        "--- Risk Analysis ---\n" + risk_out + "\n\n"
        "End with 'Sources' aggregating citations from all sections."
    )

    synth_result = _asyncio.run(assistant.run(task=synthesis_task))
    final_text = getattr(synth_result, "content", None) or str(synth_result)

    return final_text

# --- Streamlit UI ---
try:
    import streamlit as st  # safe to import here for Streamlit runs

    st.set_page_config(page_title="Stock Research Agent (Autogen)", page_icon="📊", layout="centered")
    st.title("Stock Research Agent (Autogen)")

    default_instructions = globals().get(
        "DEFAULT_RESEARCH_INSTRUCTIONS",
        (
            "You are an elite stock research analyst with access to multiple specialized tools and sub-agents.\n\n"
            "1) Gather data, 2) Fundamentals, 3) Technicals, 4) Risks, 5) Synthesis & Recommendation.\n"
            "Provide specific numbers, cite sources, and write a concise professional report."
        ),
    )

    research_instructions = st.text_area(
        "Research Instructions", height=160, value=default_instructions
    )

    query = st.text_area(
        "Research Query", height=160,
        placeholder="Example: Comprehensive analysis on Apple Inc. (AAPL)"
    )

    if st.button("Run Analysis"):
        print(f"[run_stock_research] Running analysis. {query=}, {research_instructions=}...")
        with st.spinner("Running analysis..."):
            progress_area = st.container()
            output = run_stock_research_autogen(query, progress_area, research_instructions)
            st.text_area("# Research Report", value=output, height=600)
except Exception as _e:
    # If Streamlit isn't the runner, ignore UI setup
    pass
