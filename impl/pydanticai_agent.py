"""PydanticAI: Agent с типизированным output_type и инструментами-функциями."""
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from common import SYSTEM, Answer, calc, model_cfg, search_kb


def build(model: str):
    m = OpenAIChatModel(model, provider=OpenAIProvider(**model_cfg()))
    return Agent(m, output_type=Answer, system_prompt=SYSTEM, tools=[search_kb, calc],
                 model_settings={"temperature": 0})


async def run(agent, question: str):
    r = await agent.run(question)
    calls = sum(1 for msg in r.all_messages() for p in msg.parts
                if isinstance(p, ToolCallPart) and p.tool_name in ("search_kb", "calc"))
    return r.output, calls
