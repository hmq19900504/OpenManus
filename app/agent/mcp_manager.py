import asyncio
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.agent.mcp import MCPAgent
from app.config import config
from app.logger import logger


class MCPAgentManager(BaseModel):
    """管理多个MCP代理的类，支持动态加载配置的MCP服务"""

    mcp_agents: Dict[str, MCPAgent] = Field(default_factory=dict)
    initialized: bool = False

    # 允许任意类型
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def load_from_config(self) -> None:
        """从配置文件加载MCP服务配置"""
        if not hasattr(config, "mcp") or not hasattr(config.mcp, "services"):
            logger.warning("配置中未找到MCP服务定义 (config.mcp.services)")
            return

        for service_name, service_config in config.mcp.services.items():
            logger.info(f"从配置加载MCP服务: {service_name}")
            agent = MCPAgent(name=service_name)

            # 存储代理以供后续使用
            self.mcp_agents[service_name] = agent

    async def initialize_agents(self) -> None:
        """初始化所有已加载的MCP代理"""
        if self.initialized:
            return

        if not self.mcp_agents:
            await self.load_from_config()

        init_tasks = []
        for service_name, agent in self.mcp_agents.items():
            if not hasattr(config.mcp.services, service_name):
                logger.warning(f"MCP服务配置缺失: {service_name}")
                continue

            service_config = getattr(config.mcp.services, service_name)

            # 获取连接信息
            connection_type = getattr(service_config, "connection_type", "stdio")
            command = getattr(service_config, "command", None)
            args = getattr(service_config, "args", [])
            server_url = getattr(service_config, "server_url", None)

            # 创建初始化任务
            task = asyncio.create_task(
                self._initialize_agent(
                    agent=agent,
                    connection_type=connection_type,
                    command=command,
                    args=args,
                    server_url=server_url,
                )
            )
            init_tasks.append(task)

        if init_tasks:
            await asyncio.gather(*init_tasks)
            self.initialized = True
            logger.info(f"已初始化 {len(init_tasks)} 个MCP服务")

    async def _initialize_agent(
        self,
        agent: MCPAgent,
        connection_type: str,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        server_url: Optional[str] = None,
    ) -> None:
        """初始化单个MCP代理"""
        try:
            logger.info(f"初始化MCP代理 {agent.name} (类型: {connection_type})")
            await agent.initialize(
                connection_type=connection_type,
                command=command,
                args=args or [],
                server_url=server_url,
            )
            logger.info(f"MCP代理 {agent.name} 初始化成功")
        except Exception as e:
            logger.error(f"初始化MCP代理 {agent.name} 失败: {str(e)}")

    async def cleanup(self) -> None:
        """清理所有MCP代理资源"""
        cleanup_tasks = []
        for name, agent in self.mcp_agents.items():
            task = asyncio.create_task(self._cleanup_agent(name, agent))
            cleanup_tasks.append(task)

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks)
            self.initialized = False

    async def _cleanup_agent(self, name: str, agent: MCPAgent) -> None:
        """清理单个MCP代理"""
        try:
            logger.info(f"清理MCP代理: {name}")
            await agent.cleanup()
        except Exception as e:
            logger.error(f"清理MCP代理 {name} 时出错: {str(e)}")

    def get_agent(self, name: str) -> Optional[MCPAgent]:
        """获取指定名称的MCP代理"""
        return self.mcp_agents.get(name)

    def get_all_agents(self) -> Dict[str, MCPAgent]:
        """获取所有MCP代理"""
        return self.mcp_agents
