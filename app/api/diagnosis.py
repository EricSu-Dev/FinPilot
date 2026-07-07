"""诊断 Agent 的 FastAPI 路由。"""

from __future__ import annotations

import json
import logging
import queue as sync_queue
import threading
import time
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.agents.diagnosis_agent import build_diagnosis_graph
from app.api.common import api_error, api_ok, sse_event
from app.data_source.code_validation import verify_security_code
from app.observability import elapsed_ms, log_agent_event, new_trace_id

router = APIRouter(prefix="/diagnosis", tags=["diagnosis"])


class DiagnosisRequest(BaseModel):
    """传入的诊断请求载荷。"""

    # pattern 兜底：即使绕过前端校验直接打 /api/diagnosis，非 6 位数字也会被
    # pydantic 拦下返回 422，不会进诊断图白跑一遍。
    code: str = Field(pattern=r"^\d{6}$", description="Security code such as 600519 or 110010.")
    type: Literal["stock", "fund"] = Field(description="Target asset type.")


@router.get("/validate")
async def validate_diagnosis_code(code: str, type: str):
    """校验诊断代码真实性，股票顺带返回名称。

    与 /api/portfolio/validate 行为一致，但不要求登录——诊断页是公开的。
    供诊断页输入框失焦 / 提交前调用，避免拿不存在的代码跑完整条诊断流水线。
    """
    try:
        if not code.isdigit() or len(code) != 6:
            return api_ok({"valid": False, "name": None, "message": "证券代码必须是 6 位数字"})
        resolved_name, err = await run_in_threadpool(verify_security_code, code, type)
        if err:
            return api_ok({"valid": False, "name": None, "message": err})
        return api_ok({"valid": True, "name": resolved_name, "message": None})
    except Exception as exc:
        return api_error(str(exc))


def _diagnosis_stream_payload(code: str, target_type: Literal["stock", "fund"]):
    """在每个诊断节点完成时产出对应的 SSE 事件。

    和对话首页一样的问题：基金诊断的 fetch_data_node 同步阻塞 15~50s
    （akshare 多次串行调用），期间 graph.stream() 不产出任何输出，
    前端/Nginx 静默超时。

    解决：graph.stream() 跑在独立线程里，通过线程安全队列传递 chunk；主线程
    每 5s 轮询队列，空超则发心跳 SSE 保活。
    """
    trace_id = new_trace_id("diagnosis")
    request_started_at = time.perf_counter()
    last_node_finished_at = request_started_at
    heartbeat_count = 0
    log_agent_event(
        "diagnosis.request.start",
        trace_id=trace_id,
        target_code=code,
        target_type=target_type,
    )
    graph = build_diagnosis_graph(debug=False)
    initial_state = {
        "target_code": code,
        "target_type": target_type,
        "messages": [],
        "observability_trace_id": trace_id,
    }

    q: sync_queue.Queue[tuple[str, Any]] = sync_queue.Queue()

    def _run_graph():
        try:
            for chunk in graph.stream(initial_state, stream_mode="updates"):
                q.put(("chunk", chunk))
            q.put(("done", None))
        except Exception as exc:
            q.put(("error", exc))

    t = threading.Thread(target=_run_graph, daemon=True)
    t.start()

    try:
        while True:
            try:
                msg_type, payload = q.get(timeout=5.0)
            except sync_queue.Empty:
                heartbeat_count += 1
                log_agent_event(
                    "diagnosis.heartbeat",
                    level=logging.DEBUG,
                    trace_id=trace_id,
                    target_code=code,
                    target_type=target_type,
                    count=heartbeat_count,
                )
                yield sse_event("heartbeat", {"status": "processing"})
                continue

            if msg_type == "done":
                log_agent_event(
                    "diagnosis.request.done",
                    trace_id=trace_id,
                    target_code=code,
                    target_type=target_type,
                    duration_ms=elapsed_ms(request_started_at),
                    heartbeats=heartbeat_count,
                )
                break

            if msg_type == "error":
                log_agent_event(
                    "diagnosis.request.error",
                    level=logging.ERROR,
                    trace_id=trace_id,
                    target_code=code,
                    target_type=target_type,
                    duration_ms=elapsed_ms(request_started_at),
                    error=str(payload),
                )
                yield sse_event("error", {"node": "stream", "message": str(payload)})
                return

            # msg_type == "chunk"
            chunk = payload
            if not isinstance(chunk, dict):
                continue

            for node_name, node_update in chunk.items():
                node_error = node_update.get("error") if isinstance(node_update, dict) else None
                now = time.perf_counter()
                node_duration_ms = int((now - last_node_finished_at) * 1000)
                last_node_finished_at = now
                log_agent_event(
                    "diagnosis.node.done",
                    trace_id=trace_id,
                    target_code=code,
                    target_type=target_type,
                    node=node_name,
                    status="error" if node_error else "done",
                    duration_ms=node_duration_ms,
                    total_duration_ms=elapsed_ms(request_started_at),
                )

                yield sse_event("node_complete", {
                    "node": node_name,
                    "status": "error" if node_error else "done",
                })

                if node_error:
                    log_agent_event(
                        "diagnosis.node.error",
                        level=logging.ERROR,
                        trace_id=trace_id,
                        target_code=code,
                        target_type=target_type,
                        node=node_name,
                        error=str(node_error),
                    )
                    yield sse_event("error", {
                        "node": node_name,
                        "message": f"节点 [{node_name}] 执行失败: {node_error}",
                    })
                    return

                if node_name == "risk_check" and isinstance(node_update, dict):
                    analysis_result = node_update.get("analysis_result")
                    if analysis_result is None:
                        yield sse_event("error", {
                            "node": node_name,
                            "message": "risk_check 未获取到分析结果，analyze 阶段可能已静默失败。",
                        })
                        return
                    if hasattr(analysis_result, "model_dump"):
                        payload_out = analysis_result.model_dump()
                    elif isinstance(analysis_result, dict):
                        payload_out = analysis_result
                    else:
                        payload_out = json.loads(json.dumps(analysis_result, default=str))
                    yield sse_event("analysis_result", payload_out)
    finally:
        t.join(timeout=2.0)


@router.post("")
async def diagnose(request: DiagnosisRequest):
    """运行诊断 Agent，并以节点级别进度流式推送给客户端。"""

    def event_generator():
        yield from _diagnosis_stream_payload(request.code, request.type)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
