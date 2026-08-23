import json
from pydoc import text

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.llm.models import get_chat_model


@tool
def get_weather(location: str) -> str:
    """获取某个位置的天气。"""
    return f"{location} 天气晴朗。"

model = get_chat_model()

model_with_tools = model.bind_tools([get_weather])
full = None
for chunk in model_with_tools.stream("今天南京天气怎么样？"):
    for block in chunk.content_blocks:
        full  = chunk if full is None else full + chunk
        if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
            print(f"推理：{reasoning}")
        elif block["type"] == "tool_call_chunk":
            print(f"工具调用块：{block}")
        elif block["type"] == "text":
            print(block["text"])
        else:
            ...

print("===============================")

messages = [{"role": "user", "content": "波士顿的天气怎么样？"}]

response = model_with_tools.invoke(messages)
for tool in response.tool_calls:
    tool_result = get_weather.invoke(tool)
    messages.append(tool_result)

final_response = model_with_tools.invoke(messages)
print(final_response.text)

print("===============================")

async def start():
    async for event in model.astream_events("你好"):

        if event["event"] == "on_chat_model_start":
            print(f"输入：{event['data']['input']}")

        elif event["event"] == "on_chat_model_stream":
            print(f"令牌：{event['data']['chunk'].text}")

        elif event["event"] == "on_chat_model_end":
            print(f"完整消息：{event['data']['output'].text}")

        else:
            pass

print('=======================')
class Movie(BaseModel):
    """一部带有详细信息的电影。"""
    title: str = Field(..., description="电影标题")
    year: int = Field(..., description="电影上映年份")
    director: str = Field(..., description="电影导演")
    rating: float = Field(..., description="电影评分，满分 10 分")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("提供关于电影《盗梦空间》的详细信息")
if isinstance(response,Movie):
    print("hjhj")
print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)