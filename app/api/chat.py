"""对话首页的 FastAPI 路由：流式 agent + 财报上传绑定会话。

``POST /api/chat`` 跑一个工具调用型 agent，用 SSE 逐 token 下发回答，
并在工具调用前后发出 ``tool_start`` / ``tool_end`` 进度事件。所有结果由
agent 用自然语言口语化输出，前端只渲染 markdown 文本（不做结构化卡片）。

``POST /api/chat/upload`` 把财报 PDF 入库（复用 RAG 的 ``ingest_uploaded_report``）
并把得到的 report_id 绑定到当前会话，之后该会话里的财报提问默认走这份语料。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from app.api.common import api_error, api_ok, sse_event
from app.api.deps import get_current_user
from app.api.rag import build_report_corpus_background, ingest_uploaded_report
from app.agents import chat_memory
from app.agents.chat_agent import build_chat_agent
from app.models import chat_crud
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# 工具名中英映射：前端显示用。SSE 的 tool_start/tool_end 事件里发英文函数名，
# 前端显示时查这个表转成中文。新增工具时在这里加映射即可。
_TOOL_NAME_ZH = {
    "diagnose_stock": "股票诊断",
    "diagnose_fund": "基金诊断",
    "analyze_my_portfolio": "持仓分析",
    "query_stock_report": "财报问答",
    "analyze_financial_report": "财报分析",
    "query_uploaded_report": "上传财报问答",
    "market_hotspots_tool": "市场热点数据",
    "web_search_tool": "联网搜索",
}

# 工具结果回发给前端时的摘要长度上限。完整工具结果已经作为 ToolMessage
# 喂回给 agent 内部，前端只需一个简短片段做"已执行"提示即可，避免把大段
# JSON 刷屏到聊天界面。
_TOOL_SUMMARY_LIMIT = 200

# 首条消息自动生成的标题长度上限。
_TITLE_LIMIT = 30


class ChatRequest(BaseModel):
    """一轮对话的请求载荷。"""

    message: str = Field(description="用户本轮发言。")
    conversation_id: Optional[int] = Field(
        default=None,
        description="会话 id。首次对话不传，由服务端生成并在 start 事件里返回；"
        "后续轮次带上以保留多轮上下文。",
    )


async def _chat_stream(user_id: int, message: str, conversation_id: Optional[int]):
    """生成 SSE 事件流：start → (token|tool_start|tool_end)* → done/error。"""
    cid, conv = chat_memory.get_or_create(user_id, conversation_id)
    yield sse_event("start", {"conversation_id": cid})

    # 先把用户发言写进会话历史，agent 调用时即带上完整多轮上下文。
    chat_memory.append_messages(cid, [HumanMessage(content=message)])

    # 首条消息：用其内容片段作为会话标题，方便侧边栏区分。
    if not conv.messages:
        chat_crud.rename_conversation(user_id, cid, message[:_TITLE_LIMIT])

    try:
        agent = build_chat_agent(user_id, cid)
        full_text = ""
        pending_tool: Optional[str] = None  # 已发 tool_start 但还没收到 ToolMessage 的工具名

        # conv.messages 是 DB 载入的历史；本轮用户消息已落库但不在该列表里，
        # 这里手动拼上，让 agent 拿到 完整历史 + 本轮发言。
        agent_messages = [*conv.messages, HumanMessage(content=message)]

        # 关键设计：LangGraph 的 ToolNode 调用同步工具函数（baostock / akshare
        # 均为同步 I/O）时直接阻塞当前线程。若在主 asyncio 线程里跑 agent.astream()，
        # 工具执行期间事件循环被卡住，任何 asyncio task（包括心跳）都无法调度。
        #
        # 解决方案：把 agent.astream() 扔到独立线程里跑（自带 event loop），主线程
        # 通过线程安全的 queue.Queue 收 chunk，同时用 run_in_executor 做 5s 超时轮询
        # ——队列空超 5s 即发心跳 SSE，确保前端 / Nginx 在工具执行期间不会静默超时。
        import queue as sync_queue

        chunk_q: sync_queue.Queue[tuple[str, Any]] = sync_queue.Queue()

        def _run_agent_in_thread():
            """在独立线程里跑 agent.astream()，产出推入线程安全队列。"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def _stream():
                    # recursion_limit=18 → 最多约 8 轮工具调用（每轮约 2 superstep:
                    # agent 节点 + tools 节点），用作显式熔断以防止 agent 陷入循环。
                    async for chunk, meta in agent.astream(
                        {"messages": agent_messages},
                        stream_mode="messages",
                        config={"recursion_limit": 18},
                    ):
                        chunk_q.put(("chunk", (chunk, meta)))
                    chunk_q.put(("agent_done", None))
                loop.run_until_complete(_stream())
            except Exception as exc:
                chunk_q.put(("agent_error", exc))
            finally:
                loop.close()

        agent_thread = threading.Thread(target=_run_agent_in_thread, daemon=True)
        agent_thread.start()

        try:
            while True:
                # 用 run_in_executor 做带超时的队列轮询：不阻塞主事件循环，
                # 超时 5s 无消息则发心跳 SSE。
                try:
                    msg_type, payload = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: chunk_q.get(timeout=5.0)
                    )
                except sync_queue.Empty:
                    # 5 秒内 agent 线程没产出任何 chunk → 工具执行中，发心跳保活
                    yield sse_event("heartbeat", {"status": "processing"})
                    continue

                if msg_type == "agent_done":
                    break

                if msg_type == "agent_error":
                    raise payload  # type: ignore[misc]

                # msg_type == "chunk"
                chunk, meta = payload

                # ToolMessage = 工具执行完毕，发 tool_end（带可靠工具名 + 简短摘要）。
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", None) or pending_tool or "工具"
                    tool_name_zh = _TOOL_NAME_ZH.get(tool_name, tool_name)
                    content = chunk.content
                    summary = content if isinstance(content, str) else str(content)
                    yield sse_event(
                        "tool_end",
                        {"tool": tool_name_zh, "summary": summary[:_TOOL_SUMMARY_LIMIT]},
                    )
                    pending_tool = None
                    continue

                # AIMessageChunk：可能是文本 token，也可能是工具调用参数分片。
                tool_call_chunks = getattr(chunk, "tool_call_chunks", None) or []
                content = getattr(chunk, "content", "")

                if tool_call_chunks:
                    # 工具调用参数在流式分片里（content 通常为空）。首个带 name 的
                    # 分片触发 tool_start；DeepSeek 有时不分片给 name，则用 ToolMessage
                    # 的 name 在 tool_end 时补，这里先用兜底名占位。
                    if pending_tool is None:
                        name = None
                        for tcc in tool_call_chunks:
                            if tcc.get("name"):
                                name = tcc["name"]
                                break
                        pending_tool = name or "工具"
                        pending_tool_zh = _TOOL_NAME_ZH.get(pending_tool, pending_tool)
                        yield sse_event("tool_start", {"tool": pending_tool_zh})
                    # 工具参数分片不作为 token 下发给前端。
                    continue

                if isinstance(content, str) and content:
                    full_text += content
                    yield sse_event("token", {"content": content})
        finally:
            # agent_thread 是 daemon，进程退出时会自动清理；但还是 join 一下
            # 避免 agent 还没跑完就被上层 cancel 时残留。
            agent_thread.join(timeout=2.0)

        # 把本轮最终回答写回会话历史，供下一轮多轮上下文使用。
        # 此时回答已经流式发给用户了，落库即使失败也不应再给用户弹错误（会让人
        # 困惑：明明看到回答了却又报错）。append_messages 自带重试，真失败就只记日志。
        if full_text:
            try:
                chat_memory.append_messages(cid, [AIMessage(content=full_text)])
            except Exception:  # noqa: BLE001
                logger.exception(
                    "保存 assistant 消息失败 cid=%s（用户已收到回答，不影响本次；"
                    "该轮回答不会进入历史，下一轮 agent 看不到它）", cid
                )
        yield sse_event("done", {"conversation_id": cid})
    except Exception as exc:  # noqa: BLE001  流式过程中的异常转成 error 事件
        logger.exception("chat 流式失败 user_id=%s cid=%s", user_id, cid)
        yield sse_event("error", {"message": str(exc)})


@router.post("")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
):
    """流式对话端点。返回 text/event-stream。"""
    message = (request.message or "").strip()
    if not message:
        return api_error("message 不能为空", status_code=422)

    return StreamingResponse(
        _chat_stream(current_user.id, message, request.conversation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations")
async def list_conversations(current_user: User = Depends(get_current_user)):
    """列出当前用户的全部会话（按最近更新倒序）。"""
    convs = chat_crud.list_conversations(current_user.id)
    return api_ok(
        [
            {
                "id": c.id,
                "title": c.title,
                "active_report_id": c.active_report_id,
                "create_time": c.create_time.isoformat() if c.create_time else None,
                "update_time": c.update_time.isoformat() if c.update_time else None,
            }
            for c in convs
        ]
    )


@router.get("/conversations/{conv_id}/messages")
async def get_conversation_messages(
    conv_id: int,
    current_user: User = Depends(get_current_user),
):
    """取某个会话的全部消息历史。带归属校验。"""
    conv = chat_crud.get_conversation(current_user.id, conv_id)
    if conv is None:
        return api_error("会话不存在或无权访问", status_code=404)
    msgs = chat_crud.get_messages(conv_id)
    return api_ok(
        [
            {
                "role": m.role,
                "content": m.content,
                "create_time": m.create_time.isoformat() if m.create_time else None,
            }
            for m in msgs
        ]
    )


class RenameRequest(BaseModel):
    """重命名会话的载荷。"""

    title: str


@router.patch("/conversations/{conv_id}")
async def rename_conversation(
    conv_id: int,
    request: RenameRequest,
    current_user: User = Depends(get_current_user),
):
    """重命名会话。带归属校验。"""
    conv = chat_crud.rename_conversation(current_user.id, conv_id, request.title)
    if conv is None:
        return api_error("会话不存在或无权访问", status_code=404)
    return api_ok({"id": conv.id, "title": conv.title})


@router.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: int,
    current_user: User = Depends(get_current_user),
):
    """删除会话及其全部消息。带归属校验。"""
    deleted = chat_crud.delete_conversation(current_user.id, conv_id)
    return api_ok({"deleted": deleted})


@router.post("/upload")
async def upload_report_for_chat(
    conversation_id: str = Form(""),
    file: UploadFile = File(...),
    code: str = Form(...),
    year: int | None = Form(None),
    quarter: int | None = Form(None),
    current_user: User = Depends(get_current_user),
):
    """上传财报 PDF 并绑定到当前会话。

    入库逻辑复用 ``app.api.rag.ingest_uploaded_report``；成功后把 report_id 写到
    会话的 active_report_id，之后该会话里的财报提问默认走这份语料库。
    """
    # 前端校验：文件类型与大小
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        return api_error("只支持上传 PDF 文件", status_code=422)
    try:
        logger.info("[上传端点] 收到文件 filename=%s code=%s year=%s quarter=%s conversation_id=%s",
                     filename, code, year, quarter, conversation_id)
        result = ingest_uploaded_report(file, code, year, quarter)
        report_id = result["report_id"]
        logger.info("[上传端点] 文件已保存 report_id=%s", report_id)

        # 绑定会话（立即返回给前端，不等向量库构建）
        cid_in: Optional[int] = (
            int(conversation_id) if conversation_id and conversation_id.strip().isdigit() else None
        )
        logger.info("[上传端点] 绑定会话 cid_in=%s", cid_in)
        cid, _conv = chat_memory.get_or_create(current_user.id, cid_in)
        chat_memory.set_active_report(current_user.id, cid, report_id)

        # 向量库构建在后台线程执行（4056 chunks × batch_size=10 ≈ 406 次 API 调用，
        # 同步跑会超时；后台构建期间首次问答会等它完成）
        threading.Thread(
            target=build_report_corpus_background,
            args=(report_id,),
            daemon=True,
        ).start()
        logger.info("[上传端点] 全部完成 cid=%d report_id=%s（向量库后台构建中）", cid, report_id)
        return api_ok({"report_id": report_id, "status": "processing", "conversation_id": cid})
    except ValueError as exc:
        logger.warning("[上传端点] ValueError: %s", exc)
        return api_error(str(exc), status_code=422)
    except Exception as exc:
        logger.exception("[上传端点] 上传财报失败 user_id=%s code=%s filename=%s", current_user.id, code, filename)
        return api_error(str(exc))
