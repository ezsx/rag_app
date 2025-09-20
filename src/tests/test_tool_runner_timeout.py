import time
from schemas.agent import ToolRequest
from services.tools.tool_runner import ToolRunner


def _slow_tool(delay_ms: int) -> dict:
    time.sleep(delay_ms / 1000.0)
    return {"ok": True}


def test_tool_runner_timeout():
    runner = ToolRunner(default_timeout_sec=0.05)  # 50ms
    runner.register("slow", lambda delay_ms: _slow_tool(delay_ms))
    action = runner.run("req", 1, ToolRequest(tool="slow", input={"delay_ms": 200}))
    assert action.output.ok is False
    assert isinstance(action.output.meta.took_ms, int)
    assert action.output.meta.error and action.output.meta.error.startswith("timeout>")
