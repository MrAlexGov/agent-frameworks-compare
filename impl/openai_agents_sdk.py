"""OpenAI Agents SDK поверх любого OpenAI-совместимого endpoint (Chat Completions)."""
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from agents.items import ToolCallItem
from openai import AsyncOpenAI

from common import SYSTEM, Answer, calc, model_cfg, search_kb

set_tracing_disabled(True)  # трейсы по умолчанию уходят в OpenAI — нам не нужно


def build(model: str):
    m = OpenAIChatCompletionsModel(model=model, openai_client=AsyncOpenAI(**model_cfg()))
    return Agent(name="oblachko", instructions=SYSTEM, model=m, output_type=Answer,
                 tools=[function_tool(search_kb), function_tool(calc)], model_settings=ModelSettings(temperature=0))


async def run(agent, question: str):
    r = await Runner.run(agent, question, max_turns=10)
    return r.final_output, sum(isinstance(i, ToolCallItem) for i in r.new_items)
