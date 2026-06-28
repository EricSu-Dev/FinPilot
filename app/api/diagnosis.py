"""诊断 Agent 的 FastAPI 路由。"""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.agents.diagnosis_agent import build_diagnosis_graph
from app.api.common import api_error, api_ok, sse_event
from app.data_source.code_validation import verify_security_code

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
    """在每个诊断节点完成时产出对应的 SSE 事件。"""
    graph = build_diagnosis_graph(debug=False)
    initial_state = {"target_code": code, "target_type": target_type, "messages": []}

    try:
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue

            for node_name, node_update in chunk.items():
                # 检查节点是否返回了错误
                node_error = node_update.get("error") if isinstance(node_update, dict) else None

                yield sse_event("node_complete", {
                    "node": node_name,
                    "status": "error" if node_error else "done",
                })

                # 如果节点失败，立刻发送 error 事件并终止
                if node_error:
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
                        payload = analysis_result.model_dump()
                    elif isinstance(analysis_result, dict):
                        payload = analysis_result
                    else:
                        payload = json.loads(json.dumps(analysis_result, default=str))
                    yield sse_event("analysis_result", payload)
    except Exception as exc:
        yield sse_event("error", {"node": "stream", "message": str(exc)})


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
