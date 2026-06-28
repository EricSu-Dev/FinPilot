"""聊天会话与消息的 ORM 模型。

把对话落到 MySQL，使多会话可保存、刷新不丢、可切换。每条会话还带上
active_report_id——上传财报后绑到对应会话，之后该会话的财报提问默认走这份语料。
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class ChatConversation(Base):
    """一个用户的一次对话（多轮）。"""

    __tablename__ = "chat_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="新对话")
    # 当前会话绑定的财报语料库标识（如 300308_2025_4）。None 表示未上传财报。
    active_report_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    create_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ChatMessage(Base):
    """会话内的一条消息（user 或 assistant）。"""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # 只持久化 user / assistant 文本；工具调用过程不存（agent 每轮重新决策）。
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    create_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
