import json
from typing import Any, Dict, List, Optional

from pydantic import Field

from app.logger import logger
from app.tool.base import BaseTool


class PlanningMonitor(BaseTool):
    """监控和展示Planning-Act执行进度的工具"""

    name: str = "planning_monitor"
    description: str = "监控Planning-Act执行进度"

    # 转换为Pydantic字段
    current_plan: Optional[List[Dict[str, Any]]] = Field(default=None)
    current_step: int = Field(default=0)
    total_steps: int = Field(default=0)
    step_results: Dict[int, str] = Field(default_factory=dict)
    is_planning: bool = Field(default=False)
    is_executing: bool = Field(default=False)

    def reset(self) -> None:
        """重置监控状态"""
        self.current_plan = None
        self.current_step = 0
        self.total_steps = 0
        self.step_results = {}
        self.is_planning = False
        self.is_executing = False

    def start_planning(self) -> None:
        """开始规划阶段"""
        self.reset()
        self.is_planning = True
        print("\n🧠 正在规划执行步骤...\n")

    def end_planning(self, plan: List[Dict[str, Any]]) -> None:
        """结束规划阶段"""
        self.is_planning = False
        self.current_plan = plan
        self.total_steps = len(plan)

        print("\n📑 规划完成！执行计划如下:\n")
        for step in sorted(plan, key=lambda x: x["step"]):
            step_num = step["step"]
            service = step["service"]
            desc = step.get("description", step.get("action", "未知操作"))
            print(f"  步骤 {step_num}: [{service}] {desc}")
        print("\n")

    def start_execution(self) -> None:
        """开始执行阶段"""
        self.is_executing = True
        print("\n🔄 开始按计划执行...\n")

    def update_step(
        self, step_number: int, status: str, result: Optional[str] = None
    ) -> None:
        """更新步骤状态"""
        self.current_step = step_number

        if status == "start":
            if self.current_plan:
                step_data = next(
                    (s for s in self.current_plan if s["step"] == step_number), None
                )
                if step_data:
                    service = step_data["service"]
                    desc = step_data.get(
                        "description", step_data.get("action", "未知操作")
                    )
                    print(
                        f"\n⏳ 执行步骤 {step_number}/{self.total_steps}: [{service}] {desc}"
                    )

        elif status == "complete":
            if result:
                self.step_results[step_number] = result
                result_preview = result[:100] + "..." if len(result) > 100 else result
                print(f"✅ 步骤 {step_number} 完成! 结果: {result_preview}")

        elif status == "error":
            print(f"❌ 步骤 {step_number} 执行失败! 错误: {result}")

    def end_execution(self) -> None:
        """结束执行阶段"""
        self.is_executing = False
        print("\n🏁 所有步骤执行完毕!\n")

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具功能"""
        action = kwargs.get("action", "status")

        if action == "status":
            return {
                "is_planning": self.is_planning,
                "is_executing": self.is_executing,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "progress": (
                    f"{self.current_step}/{self.total_steps}"
                    if self.total_steps > 0
                    else "0/0"
                ),
            }

        elif action == "reset":
            self.reset()
            return {"status": "reset"}

        elif action == "start_planning":
            self.start_planning()
            return {"status": "planning_started"}

        elif action == "end_planning":
            plan = kwargs.get("plan", [])
            self.end_planning(plan)
            return {"status": "planning_completed", "steps": len(plan)}

        elif action == "start_execution":
            self.start_execution()
            return {"status": "execution_started"}

        elif action == "update_step":
            step_number = kwargs.get("step", 0)
            status = kwargs.get("status", "")
            result = kwargs.get("result", "")
            self.update_step(step_number, status, result)
            return {"status": "step_updated", "step": step_number}

        elif action == "end_execution":
            self.end_execution()
            return {"status": "execution_completed"}

        else:
            return {"error": f"未知操作: {action}"}
