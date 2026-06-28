"""DeepSeek 的 LangChain 聊天模型访问器。

项目通过 LangChain 的 OpenAI 兼容聊天模型来使用 DeepSeek。为了让应用导入更
健壮，真正的客户端在首次使用时才懒创建，而不是在模块导入时创建。这意味着
FastAPI 应用可以正常启动，并在后续缺少 API key 时给出有用的错误，而不会
在启动阶段就崩溃。
"""

from __future__ import annotations

from threading import Lock
from typing import Any

from langchain_openai import ChatOpenAI

from app.config import get_settings

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

settings = get_settings()


class LazyChatModel:
    """ChatOpenAI 的代理，把客户端构造推迟到真正需要时。"""

    def __init__(self, streaming: bool):
        self.streaming = streaming
        self._client: ChatOpenAI | None = None
        self._lock = Lock()

    def _build_client(self) -> ChatOpenAI:
        """构造底层的 DeepSeek 聊天客户端。"""
        api_key = settings.DEEPSEEK_API_KEY.strip()
        if not api_key:
            raise RuntimeError(
                "缺少 DEEPSEEK_API_KEY。请将其放入项目根目录的 .env 文件中，"
                "或在启动应用前导出该环境变量。"
            )

        return ChatOpenAI(
            model=settings.MODEL_NAME,
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            streaming=self.streaming,
            temperature=0.2,
            timeout=60,
            max_retries=2,
        )

    def _get_client(self) -> ChatOpenAI:
        """返回缓存的 ChatOpenAI 实例，按需创建一次。"""
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                self._client = self._build_client()
        return self._client

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """把 invoke 调用委托给底层模型。"""
        return self._get_client().invoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        """把流式调用委托给底层模型。"""
        return self._get_client().stream(*args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """把结构化输出绑定委托给底层模型。"""
        return self._get_client().with_structured_output(*args, **kwargs)

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        """把工具绑定委托给底层模型。"""
        return self._get_client().bind_tools(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        """按需代理 ChatOpenAI 的其它属性。"""
        return getattr(self._get_client(), item)


llm = LazyChatModel(streaming=False)
"""标准的非流式 DeepSeek 聊天模型代理。"""

streaming_llm = LazyChatModel(streaming=True)
"""流式 DeepSeek 聊天模型代理。"""
