"""聊天会话的持久化访问层（MySQL 后端）。

早期版本是内存字典，重启即丢、也无法保存多个会话。现在改为全部走
``chat_crud`` 落 MySQL：会话与消息持久化，刷新不丢，可列出/切换/删除。

对外仍保留原先的几个函数签名（``get_or_create`` / ``append_messages`` /
``set_active_report`` / ``get_active_report``），让 chat 路由与 chat_agent
工具层不必改动。``get_or_create`` 返回的 ``Conversation`` 是一个本次请求用的
瞬时容器——其 ``messages`` 是从 DB 载入的历史，供 agent 作为多轮上下文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.models import chat_crud

# 给 agent 的上下文窗口：每个会话只取最近这么多条消息喂给 LLM。
# 全量历史仍在 MySQL（侧边栏/回看/刷新恢复不受影响），这里只控制上下文长度，
# 防止长对话 token 无限膨胀撞模型上下文上限。想保留更多上下文就调大这个数。
MAX_CONTEXT_MESSAGES = 20


@dataclass
class Conversation:
    """本次请求用的瞬时会话视图：历史消息 + 当前绑定的财报 report_id。"""

    user_id: int
    messages: list[BaseMessage] = field(default_factory=list)
    active_report_id: Optional[str] = None


def get_or_create(user_id: int, conversation_id: Optional[int]) -> tuple[int, Conversation]:
    """按 conversation_id 取会话；不存在或归属不符则新建。

    Returns
    -------
    tuple[int, Conversation]
        实际使用的 conversation_id（int）与本次请求的会话视图（含历史消息）。
    """
    if conversation_id:
        conv = chat_crud.get_conversation(user_id, int(conversation_id))
        if conv is None:
            # 前端传了一个不存在的 id（或属于别的用户）：另开一个新会话，不泄露他人历史。
            conv = chat_crud.create_conversation(user_id)
    else:
        conv = chat_crud.create_conversation(user_id)

    db_msgs = chat_crud.get_messages(conv.id)
    # 滑动窗口：只取最近 N 条作 agent 上下文。db_msgs 已按 id 升序（时间序），
    # 取尾部即最近的消息。全量历史仍在 DB，这里只影响喂给 LLM 的长度。
    if len(db_msgs) > MAX_CONTEXT_MESSAGES:
        db_msgs = db_msgs[-MAX_CONTEXT_MESSAGES:]
    history: list[BaseMessage] = [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in db_msgs
    ]
    return conv.id, Conversation(
        user_id=user_id,
        messages=history,
        active_report_id=conv.active_report_id,
    )


def append_messages(conversation_id: int, messages: list[BaseMessage]) -> None:
    """把消息落库。role 由消息类型推断：HumanMessage→user，其余→assistant。

    带重试：远程 MySQL 下，长 agent 运行期间连接池里的连接可能被 NAT 掐断，
    导致落库时撞上半开连接（2013 Lost connection）。每次 add_message 都开自己的
    session，pool_pre_ping 会在重试时换一条新连接，所以重试通常第二次就成功。
    """
    import time

    from sqlalchemy.exc import OperationalError

    cid = int(conversation_id)
    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        content = m.content or ""
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                chat_crud.add_message(cid, role, content)
                last_exc = None
                break
            except OperationalError as exc:
                last_exc = exc
                # 短退避后重试，让 pool_pre_ping 回收坏连接、换一条新的。
                time.sleep(0.5 * (attempt + 1))
        if last_exc is not None:
            raise last_exc


def set_active_report(user_id: int, conversation_id: int, report_id: str) -> None:
    """把上传得到的 report_id 绑到会话。带归属校验。"""
    chat_crud.set_active_report(user_id, int(conversation_id), report_id)


def get_active_report(user_id: int, conversation_id: int) -> Optional[str]:
    """读取会话当前绑定的 report_id。"""
    return chat_crud.get_active_report(user_id, int(conversation_id))
