# Your code here
import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_chroma import Chroma
import textwrap
from langchain_aws.chat_models import ChatBedrockConverse
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor

config="""
[default]
aws_access_key_id=[add_here]
aws_secret_access_key=[add_here]
aws_session_token=[add_here]
"""

# Parse credentials from string
access_key_id=config.split("aws_access_key_id=")[1].split("\n")[0]
secret_access_key=config.split("aws_secret_access_key=")[1].split("\n")[0]
session_token=config.split("aws_session_token=")[1].split("\n")[0]

# Once the keys are generated you do the following to create your credentials file
os.environ["AWS_ACCESS_KEY_ID"] = access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = secret_access_key
os.environ["AWS_SESSION_TOKEN"] = session_token
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

# Page and Model Set up
st.set_page_config(page_title = "Enron Email Assistant", layout = "wide")

CHATMODEL = "us.meta.llama3-1-70b-instruct-v1:0"
REGION = "us-east-1"
PERSIST_DIR = "./chroma_db"
STOCK_EXCEL_PATH = "assets/enron_stock_prices.xlsx"

@st.cache_resource
def load_llm_and_embeddings():
    embeddings = HuggingFaceEmbeddings(encode_kwargs = {"batch_size": 64})
    llm = ChatBedrockConverse(model = CHATMODEL, region_name = REGION)
    return embeddings, llm

embeddings, llm = load_llm_and_embeddings()

# Tools
@tool("enron_email_rag")
def enron_email_rag(query: str) -> str: 
    """
    Searches and retrieves information from the Enron corporation email corpus vector database.
    Use this tool when anwering questions about emails, employees (e.g. Jeffrey Skilling),
    communications, meetings, or potential illegal activities.
    """
    try: 
        chroma_db = Chroma(
            persist_directory = PERSIST_DIR, embedding_function = embeddings
        )
        retriever = chroma_db.as_retriever(search_kwargs = {"k": 5})
        docs = retriever.invoke(query)

        context = "\n\n".join(
            [
                f"Subject: {d.metadata.get('subject', 'N/A')}\nContent: {d.page_content}"
                for d in docs
            ]
        )
        return context if context else "No relevant email context found."
    except Exception as e:
        return f"Error querying email database: {str(e)}"

@tool("enron_stock_analyzer")
def enron_stock_analyzer(analysis_type: str) -> str:
    """
    Analyzes overall Enron stock price history, statistics (mean, std, min, max),
    peak/low dates, and trading volume spikes.
    Pass 'overall' or 'volume' or specific stock questions as the input query.
    """
    try: 
        df = pd.read_excel(STOCK_EXCEL_PATH, header = 4, sheet_name = "Sheet1")
        df["Date"] = pd.to_datetime(df["Date"])

        desc = df["Close"].describe()
        mean_val = desc["mean"]
        std_val = desc["std"]
        count_val = desc["count"]

        max_row = df.loc[df["Close"].idxmax()]
        min_row = df.loc[df["Close"].idxmin()]

        max_vol_row = df.loc[df["Volume"].idxmax()]

        summary = (
            f"Enron Stock Price Analysis:\n\n"
            f"Basic Statistics of Closing Price:\n"
            f"count     {count_val:.6f}\n"
            f"mean      {mean_val:.6f}\n"
            f"std       {std_val:.6f}\n"
            f"Peak Price: {max_row["Close"]:.2f} USD on {max_row["Date"].strftime("%B %d %Y")}\n"
            f"Lowest Price: {min_row["Close"]:.2f} USD on {min_row["Date"].strftime("%B %d %Y")}\n"
            f"Highest Volume Day: {max_vol_row["Date"].strftime("%B %d %Y")}\n"
            f"with volume of {(max_vol_row["Volume"]):,} and closing price of {max_vol_row["Close"]:2f} USD."
        )
        return summary
    except Exception as e:
        return f"Error reading stock data: {str(e)}"

@tool("stock_over_time")
def stock_over_time(start_date: str, end_date: str) -> str:
    """
    Calculates the stock prices gain/loss over start_date - end_date (inclusive, YYYY-MM-DD).
    """
    try: 
        df = pd.read_excel(STOCK_EXCEL_PATH, header = 4, sheet_name = "Sheet1")
        df["Date"] = pd.to_datetime(df["Date"])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        start_price = float(df[df["Date"] == start_dt]["Open"].iloc[0])
        end_price = float(df[df["Date"] == end_dt]["Open"].iloc[0])
        diff = round(end_price - start_price, 3)
        return f"{diff} USD"
    except Exception as e:
        return f"Error calculating stock price difference: {str(e)}"

tools = [enron_email_rag, enron_stock_analyzer, stock_over_time]


# Agent & Chain

system_prompt = """You are a helpful data analysis assistant/agent with access to corporate email inbox data and stock price data for Enron.
If the user asks for information or metrics that are NOT present in the given data/context, do NOT attempt to call tools or hallucinate values.
Instead of stating that there is no available function, CLEARLY explain that the dataset does not contain that specific information and/or state
that you cannot make the requested calculation.
Always cite reference email subject line(s) when answering questions using email data."""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name = "agent_scratchpad")
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(
    agent = agent, tools = tools, return_intermediate_steps = True, verbose = True
)

# Streamlit Set Up
st.title("Enron Email Assistant")
st.caption(
    "An AI assistant to help you analyze the Enron email corpus and stock price data."
)
            
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_response" not in st.session_state:
    st.session_state.latest_response = ""
if "latest_steps" not in st.session_state:
    st.session_state.latest_steps = []
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

question_val = st.text_area(
    "Question",
    key = "question_box",
    value = st.session_state.question_input,
    height = 100
)

col_submit, _ = st.columns([1, 5])
with col_submit: 
    submit_clicked = st.button("Submit", use_container_width = True)

st.write("⋮☰ **Examples**")
col_ex1, col_ex2, _ = st.columns([1.5, 1.8, 3])

with col_ex1:
    if st.button("What was the day with the highest trading volume?"):
        st.session_state.question_input = (
            "What was the day with the highest trading volume?"
        )
        st.rerun()
with col_ex2: 
    if st.button("What was the overall performance of Enron's stock price?"):
        st.session_state.question_input = (
            "What was the overall performance of Enron's stock price?"
        )
        st.rerun()

if submit_clicked and question_val.strip():
    user_query = question_val.strip()

    formatted_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content = msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(AIMessage(content = msg["content"]))

    with st.spinner("Analyzing emails and stock data..."):
        result = agent_executor.invoke(
            {"input": user_query, "chat_history": formatted_history}
        )

        final_ans = result.get("output", "")
        steps = result.get("intermediate_steps", [])

        formatted_steps = []
        for step_action, step_observation in steps:
            tool_name = step_action.tool
            tool_input = step_action.tool_input
            log_str = step_action.log
            formatted_steps.append(
                f"Tool: {tool_name}\n"
                f"Input: {tool_input}\n"
                f"Log: Thought: Do I need to use a tool? Yes\n"
                f"Action: {tool_name}\n"
                f"Action Input: {tool_input}\n"
                f"Result:\n{step_observation}"
            )

        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": final_ans}
        )
        st.session_state.latest_response = final_ans
        st.session_state.latest_steps = "\n\n".join(formatted_steps)

tab_final, tab_steps, tab_history = st.tabs(
    ["Final Response", "Intermediate Steps", "Chat History"]
)

with tab_final:
    st.subheader("Final Response")
    if st.session_state.latest_response:
        st.info(st.session_state.latest_response)
    else:
        st.write("Submit a query to view the final response.")
with tab_steps:
    st.subheader("Intermediate Steps")
    if st.session_state.latest_steps:
        st.text_area(
            label = "Agent Thought Process and Execution",
            value = st.session_state.latest_steps,
            height = 300,
            disabled = True
        )
    else: 
        st.write("No intermediate thought steps recorded yet.")
with tab_history: 
    st.subheader("Chat History")
    if st.session_state.messages:
        history_text = ""
        for msg in st.session_state.messages:
            role_prefix = "User: " if msg["role"] == "user" else "AI: "
            history_text += f"{role_prefix}{msg["content"]}\n"

        st.text_area(
            label = "Full Conversation History",
            value = history_text,
            height = 400,
            disabled = True
        )
    else:
        st.write("No chat history available.")