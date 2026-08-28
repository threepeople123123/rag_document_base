from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from app.core.config import settings

_chat_model:BaseChatModel | None = None

_chat_mini_model:BaseChatModel | None = None

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 每 10 秒 1 个请求
    check_every_n_seconds=0.1,  # 每 100 毫秒检查是否允许发出请求
    max_bucket_size=10,  # 控制最大突发大小。
)


class MyCallbackHandler(BaseCallbackHandler):
    """示例回调处理器：打印 LLM 调用的开始/结束。"""

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        print("LLM 调用开始")

    def on_llm_new_token(self, token, **kwargs) -> None:
        print(f"新令牌: {token}")

    def on_llm_end(self, response, **kwargs) -> None:
        print("LLM 调用结束")


def get_chat_model(mini_chat_model:bool=False)->BaseChatModel:

    if not mini_chat_model:
        global _chat_mini_model
        if _chat_mini_model is not None:
            return _chat_mini_model
        _chat_mini_model = ChatOpenAI(
            base_url=settings.CHAT_BASE_URL,
            model=settings.CHAT_MODEL,
            api_key=settings.CHAT_API_KEY
            ,temperature=0
            ,streaming=True,
            # 注意：阿里云 MaaS compatible-mode 是 OpenAI 兼容端点，不是 OpenAI 官方
            # reasoning 模型。配 reasoning 会触发 langchain-openai 走 Responses API
            # (/responses)，而该端点对 structured outputs(parsed 字段)支持不完整，
            # 导致 with_structured_output 解析失败。统一强制走 chat completions。
            use_responses_api=False,
            rate_limiter=rate_limiter,
            # 注意：config 不是 ChatOpenAI 的构造参数（它只属于 invoke/stream 的调用期配置），
            # 传进去会被塞进 model_kwargs 导致 "unexpected keyword argument 'config'"。
            # tags / metadata / callbacks 是模型本身支持的字段，可以直接在构造时设置；
            # run_name 只能在调用时通过 config 传，例如:
            #   model.invoke("...", config={"run_name": "joke_generation"})
            tags=["humor", "demo"],          # 用于分类的标签
            metadata={"user_id": "123"},     # 自定义元数据
            max_tokens=10000,
            # callbacks=[MyCallbackHandler()], # 回调处理程序（必须是 BaseCallbackHandler 实例）
        )
        return _chat_mini_model
    else:
        global _chat_model
        if _chat_model is not None:
            return _chat_model
        _chat_model = ChatOpenAI(
            base_url=settings.CHAT_BASE_URL,
            model=settings.MINI_CHAT_MODEL,
            api_key=settings.CHAT_API_KEY
            ,temperature=0
            ,streaming=True,
            use_responses_api=False,
            rate_limiter=rate_limiter,
            # 注意：config 不是 ChatOpenAI 的构造参数（它只属于 invoke/stream 的调用期配置），
            # 传进去会被塞进 model_kwargs 导致 "unexpected keyword argument 'config'"。
            # tags / metadata / callbacks 是模型本身支持的字段，可以直接在构造时设置；
            # run_name 只能在调用时通过 config 传，例如:
            #   model.invoke("...", config={"run_name": "joke_generation"})
            tags=["humor", "demo"],          # 用于分类的标签
            metadata={"user_id": "123"},     # 自定义元数据
            # callbacks=[MyCallbackHandler()], # 回调处理程序（必须是 BaseCallbackHandler 实例）
        )
        return _chat_model



system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")

# 与聊天模型一起使用
messages = [system_msg, human_msg]
