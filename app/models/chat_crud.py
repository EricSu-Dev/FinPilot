"""聊天会话与消息的 CRUD 辅助函数。

风格与 ``portfolio_crud`` 一致：短生命周期的 session，关闭 expire_on_commit
以便在 session 生命周期外读取属性；所有查询/修改都以 ``user_id`` 做归属校验，
避免一个用户读到/改到另一用户的会话。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chat import ChatConversation, ChatMessage
from app.models.database import SessionLocal


@contextmanager
def _session_scope() -> Iterator[Session]:
    """短生命周期 session，处理提交/回滚。

    关闭 expire_on_commit：函数会把 ORM 对象返回到 session 生命周期之外，
    默认的 expire_on_commit=True 会在 commit 后标记属性过期，导致关闭后访问
    触发 DetachedInstanceError。
    """
    db = SessionLocal(expire_on_commit=False)
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_conversation(user_id: int, title: str = "新对话") -> ChatConversation:
    """为用户创建一条新会话。"""
    with _session_scope() as db:
        conv = ChatConversation(user_id=user_id, title=title)
        db.add(conv)
        db.flush()
        db.refresh(conv)
        return conv


def get_conversation(user_id: int, conv_id: int) -> Optional[ChatConversation]:
    """按 id 取会话，带归属校验。不属于该用户返回 None。"""
    with _session_scope() as db:
        return db.scalar(
            select(ChatConversation).where(
                ChatConversation.id == conv_id, ChatConversation.user_id == user_id
            )
        )


def list_conversations(user_id: int) -> list[ChatConversation]:
    """返回该用户的全部会话，按最近更新倒序。"""
    with _session_scope() as db:
        return list(
            db.scalars(
                select(ChatConversation)
                .where(ChatConversation.user_id == user_id)
                .order_by(ChatConversation.update_time.desc())
            ).all()
        )


def delete_conversation(user_id: int, conv_id: int) -> bool:
    """删除一条会话及其全部消息。不属于该用户返回 False。"""
    with _session_scope() as db:
        conv = db.scalar(
            select(ChatConversation).where(
                ChatConversation.id == conv_id, ChatConversation.user_id == user_id
            )
        )
        if conv is None:
            return False
        db.delete(conv)
        return True


def rename_conversation(user_id: int, conv_id: int, title: str) -> Optional[ChatConversation]:
    """重命名会话。带归属校验。"""
    with _session_scope() as db:
        conv = db.scalar(
            select(ChatConversation).where(
                ChatConversation.id == conv_id, ChatConversation.user_id == user_id
            )
        )
        if conv is None:
            return None
        conv.title = title.strip()[:200] or "新对话"
        db.flush()
        db.refresh(conv)
        return conv


def add_message(conv_id: int, role: str, content: str) -> ChatMessage:
    """追加一条消息。"""
    with _session_scope() as db:
        msg = ChatMessage(conversation_id=conv_id, role=role, content=content)
        db.add(msg)
        db.flush()
        db.refresh(msg)
        return msg


def get_messages(conv_id: int) -> list[ChatMessage]:
    """返回会话的全部消息，按 id 升序（即时间顺序）。"""
    with _session_scope() as db:
        return list(
            db.scalars(
                select(ChatMessage)
                .where(ChatMessage.conversation_id == conv_id)
                .order_by(ChatMessage.id.asc())
            ).all()
        )


def set_active_report(user_id: int, conv_id: int, report_id: str) -> bool:
    """把 report_id 绑到会话。带归属校验。"""
    with _session_scope() as db:
        conv = db.scalar(
            select(ChatConversation).where(
                ChatConversation.id == conv_id, ChatConversation.user_id == user_id
            )
        )
        if conv is None:
            return False
        conv.active_report_id = report_id
        return True


def get_active_report(user_id: int, conv_id: int) -> Optional[str]:
    """读取会话绑定的 report_id。带归属校验。"""
    with _session_scope() as db:
        conv = db.scalar(
            select(ChatConversation).where(
                ChatConversation.id == conv_id, ChatConversation.user_id == user_id
            )
        )
        return conv.active_report_id if conv is not None else None
