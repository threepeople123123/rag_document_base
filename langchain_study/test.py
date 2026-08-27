import base64

from langchain.agents import create_agent
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import Field, BaseModel

from app.llm.models import get_chat_model


@tool
def get_weather(location: str,runtime: ToolRuntime) -> str:
    """获取某个位置的天气。"""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")
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

# model_with_structure = model.with_structured_output(Movie)
# response = model_with_structure.invoke("提供关于电影《盗梦空间》的详细信息")
# if isinstance(response,Movie):
#     print("hjhj")
# data = response.model_dump()
# json_str = json.dumps(data, ensure_ascii=False, indent=4)
# print(json_str)

print("================================")

for chunk in model.stream("为什么鹦鹉有五颜六色的羽毛？"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    print(reasoning_steps if reasoning_steps else chunk.text)

print("========================")

callback = UsageMetadataCallbackHandler()
result_1 = model.invoke("你好", config={"callbacks": [callback]})
model_name, usage = next(iter(callback.usage_metadata.items()))

print(model_name)
print(usage["input_tokens"])
print(usage["output_tokens"])
print(usage["total_tokens"])

print("===================================")



ai_message = []
init_message = HumanMessage("北京的天气怎么样")
while(True):
    for chunk in  model_with_tools.stream(messages):
        print(f"{chunk}")
    break

with open("test.jpeg", "rb") as f:
    image_data = f.read()

# 编码为 base64 字符串
base64_str = base64.b64encode(image_data).decode("utf-8")


message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"},
        },
    ]
)


image_response = model.invoke([message])
print(image_response.content)

@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

