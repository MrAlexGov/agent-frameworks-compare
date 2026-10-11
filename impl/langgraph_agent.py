"""LangGraph: ReAct-агент из prebuilt + structured output через response_format."""
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from common import SYSTEM, Answer, calc, model_cfg, search_kb


def build(model: str):
    llm = ChatOpenAI(model=model, temperature=0, **model_cfg())
    return create_react_agent(llm, [tool(search_kb), tool(calc)], prompt=SYSTEM, response_format=Answer)


async def run(agent, question: str):
    state = await agent.ainvoke({"messages": [("user", question)]}, {"recursion_limit": 12})
    calls = sum(len(getattr(m, "tool_calls", None) or []) for m in state["messages"])
    return state["structured_response"], calls
