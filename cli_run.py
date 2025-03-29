#!/usr/bin/env python
import argparse
import asyncio
import sys

from app.agent.manus import Manus
from app.agent.mcp import MCPAgent
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="OpenManus 命令行工具")

    # 运行模式选择
    parser.add_argument(
        "--mode",
        "-m",
        choices=["agent", "flow", "mcp"],
        default="agent",
        help="运行模式: agent (默认), flow, mcp",
    )

    # MCP 连接模式
    parser.add_argument(
        "--connection",
        "-c",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP 连接类型: stdio 或 sse (仅在 mcp 模式下使用)",
    )

    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/sse",
        help="SSE 连接的 URL (仅在 mcp 模式且 connection=sse 时使用)。多个服务器可用逗号分隔，例如 'http://server1.com/sse,http://server2.com/sse'",
    )

    # 提示词，必选参数
    parser.add_argument("prompt", help="要处理的提示词")

    return parser.parse_args()


async def run_agent(prompt: str) -> None:
    """运行 Manus Agent 处理提示词"""
    if not prompt.strip():
        logger.warning("提供了空的提示词。")
        return

    agent = Manus()
    try:
        logger.warning("正在处理您的请求...")
        result = await agent.run(prompt)
        logger.info("请求处理完成。")
        logger.info(f"结果: {result}")
    except KeyboardInterrupt:
        logger.warning("操作被中断。")


async def run_flow(prompt: str) -> None:
    """使用 PlanningFlow 处理提示词"""
    if prompt.strip().isspace() or not prompt:
        logger.warning("提供了空的提示词。")
        return

    agents = {
        "manus": Manus(),
    }

    try:
        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        logger.warning("正在处理您的请求...")

        try:
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 分钟超时
            )
            logger.info("请求处理完成。")
            logger.info(f"结果: {result}")
        except asyncio.TimeoutError:
            logger.error("请求处理在 1 小时后超时")
            logger.info("操作因超时而终止。请尝试更简单的请求。")

    except KeyboardInterrupt:
        logger.info("操作被用户取消。")
    except Exception as e:
        logger.error(f"错误: {str(e)}")


class MCPRunner:
    """MCP Agent 运行器类"""

    def __init__(self):
        self.root_path = config.root_path
        self.server_reference = "app.mcp.server"
        self.agents = []  # 存储多个 MCPAgent 实例
        self.results = []  # 存储来自多个代理的结果

    async def initialize(
        self,
        connection_type: str,
        server_urls: list[str] | None = None,
    ) -> None:
        """初始化 MCP Agent 连接"""
        if not server_urls:
            server_urls = ["http://127.0.0.1:8000/sse"]

        logger.info(
            f"使用 {connection_type} 连接初始化 MCPAgents，服务器: {server_urls}"
        )

        # 为每个服务器 URL 创建一个 MCPAgent
        for server_url in server_urls:
            agent = MCPAgent()

            if connection_type == "stdio":
                await agent.initialize(
                    connection_type="stdio",
                    command=sys.executable,
                    args=["-m", self.server_reference],
                )
            else:  # sse
                await agent.initialize(
                    connection_type="sse", server_url=server_url.strip()
                )

            self.agents.append(agent)
            logger.info(f"已通过 {connection_type} 连接到 MCP 服务器 {server_url}")

        if not self.agents:
            raise ValueError("无法初始化任何 MCP Agent")

    async def run_prompt(self, prompt: str) -> None:
        """运行多个 Agent 处理单个提示词，并聚合结果"""
        logger.warning("正在处理您的请求...")

        # 并行运行所有代理
        tasks = [agent.run(prompt) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"代理 {i+1} 出错: {str(result)}")
                self.results.append(f"错误: {str(result)}")
            else:
                logger.info(f"代理 {i+1} 完成处理")
                self.results.append(result)

        # 输出结果
        for i, result in enumerate(self.results):
            logger.info(f"服务器 {i+1} 结果: {result}\n{'-'*40}")

        logger.info("所有请求处理完成。")

    async def cleanup(self) -> None:
        """清理所有 Agent 资源"""
        for agent in self.agents:
            await agent.cleanup()
        logger.info("所有会话已结束")


async def run_mcp(prompt: str, connection: str, server_url: str) -> None:
    """运行 MCP Agent 处理提示词"""
    if not prompt.strip():
        logger.warning("提供了空的提示词。")
        return

    # 解析多个服务器 URL
    server_urls = [url.strip() for url in server_url.split(",") if url.strip()]
    if not server_urls:
        logger.error("没有提供有效的服务器 URL")
        return

    runner = MCPRunner()

    try:
        # 初始化 MCP Agent 连接
        await runner.initialize(connection, server_urls)
        await runner.run_prompt(prompt)
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"运行 MCPAgent 时出错: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        await runner.cleanup()


async def main() -> None:
    """主函数，根据命令行参数选择运行模式"""
    args = parse_args()

    if args.mode == "agent":
        await run_agent(args.prompt)
    elif args.mode == "flow":
        await run_flow(args.prompt)
    elif args.mode == "mcp":
        await run_mcp(args.prompt, args.connection, args.server_url)


if __name__ == "__main__":
    asyncio.run(main())
