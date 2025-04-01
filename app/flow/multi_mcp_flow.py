import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from app.agent.base import BaseAgent
from app.agent.manus import Manus
from app.agent.mcp import MCPAgent
from app.agent.mcp_manager import MCPAgentManager
from app.flow.base import BaseFlow
from app.logger import logger
from app.tool.planning_monitor import PlanningMonitor


class MultiMCPFlow(BaseFlow):
    """多MCP服务工作流，支持动态路由请求到不同的MCP服务，实现planning-act模式"""

    mcp_manager: MCPAgentManager = Field(default_factory=MCPAgentManager)
    planner: Optional[Manus] = None
    monitor: PlanningMonitor = Field(default_factory=PlanningMonitor)

    def __init__(self, agents: Dict[str, BaseAgent]):
        super().__init__(agents)
        self.mcp_manager = MCPAgentManager()
        self.planner = agents.get("manus")  # 使用Manus智能体作为规划器
        self.monitor = PlanningMonitor()  # 添加进度监控工具

    async def initialize(self) -> None:
        """初始化工作流和所有MCP代理"""
        await self.mcp_manager.initialize_agents()

        # 将所有MCP代理添加到agents字典中
        for name, agent in self.mcp_manager.get_all_agents().items():
            self.agents[name] = agent

        logger.info(f"多MCP服务工作流初始化完成，可用服务: {list(self.agents.keys())}")

    async def cleanup(self) -> None:
        """清理所有资源"""
        await self.mcp_manager.cleanup()

        # 清理其他代理
        for name, agent in self.agents.items():
            if (
                hasattr(agent, "cleanup")
                and name != "manus"
                and not name in self.mcp_manager.get_all_agents()
            ):
                try:
                    await agent.cleanup()
                except Exception as e:
                    logger.error(f"清理代理 {name} 时出错: {str(e)}")

    async def route_request(self, prompt: str) -> Tuple[str, Optional[BaseAgent]]:
        """根据提示内容路由到最合适的MCP服务"""
        # 检查显式路由指令 (@service_name query)
        explicit_match = re.match(r"@(\w+)\s+(.*)", prompt)
        if explicit_match:
            service_name = explicit_match.group(1)
            actual_prompt = explicit_match.group(2)

            if service_name in self.agents:
                return actual_prompt, self.agents[service_name]
            else:
                logger.warning(f"未找到指定的服务: {service_name}，将使用默认规划")

        # 使用规划器确定最佳服务
        if self.planner and isinstance(self.planner, Manus):
            planning_prompt = (
                f"分析以下请求，并确定最适合处理的服务。可用服务: "
                f"{list(self.agents.keys())}。\n回答格式: '服务名: 理由'\n\n请求: {prompt}"
            )

            planning_result = await self.planner.run(planning_prompt)
            service_match = re.match(r"(\w+):", planning_result)

            if service_match:
                service_name = service_match.group(1)
                if service_name in self.agents:
                    logger.info(f"规划器选择了服务: {service_name}")
                    return prompt, self.agents[service_name]

        # 默认返回原始提示和None，表示使用所有服务的组合结果
        return prompt, None

    async def planning_phase(self, prompt: str) -> List[Dict[str, Any]]:
        """规划阶段：制定详细执行计划"""
        # 开始规划阶段监控
        self.monitor.start_planning()

        if not self.planner or not isinstance(self.planner, Manus):
            raise ValueError("Planning-Act模式需要Manus智能体作为规划器")

        # 获取可用的MCP服务列表
        available_services = list(self.mcp_manager.get_all_agents().keys())
        if not available_services:
            logger.warning("没有可用的MCP服务，只能使用manus")
            available_services = ["manus"]

        # 构建规划提示词
        planning_prompt = f"""
请为以下任务制定详细的执行计划，按照Planning-Act模式，先思考再执行。
可用服务: {', '.join(['manus'] + available_services)}

您的计划应该包含以下要素:
1. 分析任务需求
2. 设计详细的执行步骤
3. 每个步骤应明确指定使用哪个服务执行
4. 步骤之间的依赖关系

以JSON格式返回，格式如下:
```json
[
  {{
    "step": 1,
    "service": "服务名",
    "action": "具体行动描述",
    "input": "输入内容或前一步骤的输出引用",
    "description": "步骤说明"
  }},
  ...
]
```

用户任务: {prompt}
"""

        # 执行规划
        logger.info("开始规划阶段...")
        plan_result = await self.planner.run(planning_prompt)

        # 尝试解析返回的JSON
        try:
            # 提取JSON部分
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", plan_result)
            if json_match:
                plan_json = json_match.group(1).strip()
            else:
                plan_json = plan_result.strip()

            plan = json.loads(plan_json)

            # 校验计划格式
            if not isinstance(plan, list):
                raise ValueError("计划必须是步骤列表")

            for step in plan:
                if (
                    not isinstance(step, dict)
                    or "step" not in step
                    or "service" not in step
                ):
                    raise ValueError(f"步骤格式错误: {step}")

            logger.info(f"规划完成，共{len(plan)}个步骤")

            # 结束规划阶段监控
            self.monitor.end_planning(plan)

            return plan

        except Exception as e:
            logger.error(f"解析规划结果时出错: {str(e)}")
            # 如果解析失败，构造简单计划
            plan = [
                {
                    "step": 1,
                    "service": "manus",
                    "action": "分析任务",
                    "input": prompt,
                    "description": "使用manus分析整体任务",
                }
            ]

            # 结束规划阶段监控（显示简单计划）
            self.monitor.end_planning(plan)

            return plan

    async def execution_phase(
        self, plan: List[Dict[str, Any]], original_prompt: str
    ) -> str:
        """执行阶段：按照计划逐步执行"""
        # 开始执行阶段监控
        self.monitor.start_execution()

        step_results = {}  # 存储每一步的执行结果
        all_results = []  # 按顺序保存所有步骤的输出

        logger.info("开始执行阶段...")

        # 按顺序执行每一步
        for step_data in sorted(plan, key=lambda x: x["step"]):
            step_num = step_data["step"]
            service_name = step_data["service"]
            action = step_data["action"]
            step_input = step_data.get("input", original_prompt)
            description = step_data.get("description", f"步骤{step_num}")

            # 更新步骤状态为开始
            self.monitor.update_step(step_num, "start")

            # 处理输入中的前置步骤引用 ${step.N}
            if isinstance(step_input, str):
                step_ref_pattern = r"\${step\.(\d+)}"
                step_refs = re.findall(step_ref_pattern, step_input)

                for ref in step_refs:
                    ref_num = int(ref)
                    if ref_num in step_results:
                        step_input = step_input.replace(
                            f"${{step.{ref}}}", step_results[ref_num]
                        )

            # 获取指定的服务
            agent = self.agents.get(service_name)
            if not agent:
                logger.warning(f"找不到服务: {service_name}，将使用manus代替")
                agent = self.agents.get("manus")
                if not agent:
                    error_msg = f"找不到代替服务: manus"
                    self.monitor.update_step(step_num, "error", error_msg)
                    raise ValueError(error_msg)

            # 执行当前步骤
            logger.info(f"执行步骤{step_num}: {description} (使用服务: {service_name})")

            # 构建步骤的提示词
            step_prompt = f"执行任务: {action}\n\n输入: {step_input}"

            try:
                start_time = asyncio.get_event_loop().time()
                step_result = await agent.run(step_prompt)
                elapsed = asyncio.get_event_loop().time() - start_time

                logger.info(f"步骤{step_num}完成，耗时: {elapsed:.2f}秒")

                # 存储结果
                step_results[step_num] = step_result
                all_results.append(f"步骤{step_num} [{service_name}]: {step_result}")

                # 更新步骤状态为完成
                self.monitor.update_step(step_num, "complete", step_result)

            except Exception as e:
                error_msg = f"执行步骤{step_num}时出错: {str(e)}"
                logger.error(error_msg)
                step_results[step_num] = error_msg
                all_results.append(f"步骤{step_num} [{service_name}] 失败: {str(e)}")

                # 更新步骤状态为错误
                self.monitor.update_step(step_num, "error", str(e))

        # 结束执行阶段监控
        self.monitor.end_execution()

        # 使用Manus整合所有结果
        if "manus" in self.agents and len(all_results) > 1:
            integration_prompt = (
                f"根据以下执行步骤的结果，针对原始请求提供综合回答。\n\n"
                f"原始请求: {original_prompt}\n\n"
                f"执行结果:\n" + "\n\n".join(all_results)
            )

            logger.info("整合所有执行结果...")
            final_result = await self.agents["manus"].run(integration_prompt)
            return final_result
        elif len(all_results) == 1:
            # 如果只有一个步骤，直接返回其结果
            return step_results[plan[0]["step"]]
        else:
            # 如果执行失败或没有步骤，返回错误信息
            return "执行计划失败，未产生有效结果"

    async def execute(self, prompt: str) -> Any:
        """执行多MCP工作流，实现planning-act模式"""
        try:
            # 初始化工作流和代理
            await self.initialize()

            # 检查是否是显式路由请求
            routed_prompt, target_agent = await self.route_request(prompt)
            if target_agent:
                logger.info(f"显式路由请求到服务: {target_agent.name}")
                return await target_agent.run(routed_prompt)

            # 执行Planning-Act模式
            logger.info("使用Planning-Act模式处理请求")

            # 1. 规划阶段
            plan = await self.planning_phase(prompt)

            # 2. 执行阶段
            result = await self.execution_phase(plan, prompt)

            return result

        finally:
            # 确保资源被清理
            await self.cleanup()
