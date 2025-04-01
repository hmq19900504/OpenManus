#!/usr/bin/env python
import argparse
import asyncio
import json
import time

from app.agent.manus import Manus
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger


async def run_multi_mcp(interactive=False):
    # 创建基础代理
    agents = {
        "manus": Manus(),  # 主规划代理
    }

    try:
        flow = FlowFactory.create_flow(
            flow_type=FlowType.MULTI_MCP,
            agents=agents,
        )

        if interactive:
            print("\n========================================")
            print("🔍 Planning-Act 多MCP代理交互模式")
            print("========================================")
            print("- 输入'exit'退出")
            print("- 使用@服务名 格式可以直接指定服务")
            print("- 例如: @sequential_thinking 分析问题")
            print("- 系统将自动使用planning-act模式执行任务")
            print("========================================\n")

            while True:
                prompt = input("\n🔹 请输入您的指令: ")
                if prompt.lower() in ["exit", "quit", "q"]:
                    break

                if not prompt.strip():
                    logger.warning("指令为空")
                    continue

                print("\n📋 开始处理请求...\n")
                try:
                    start_time = time.time()
                    result = await asyncio.wait_for(
                        flow.execute(prompt),
                        timeout=3600,  # 1小时超时
                    )
                    elapsed_time = time.time() - start_time

                    print("\n========================================")
                    print(f"✅ 执行完成! 耗时: {elapsed_time:.2f} 秒")
                    print("========================================")
                    print(f"\n📝 结果: \n{result}\n")

                except asyncio.TimeoutError:
                    logger.error("请求处理超时（1小时）")
                    print("\n❌ 执行超时！请尝试更简单的请求。")
                except Exception as e:
                    logger.error(f"处理请求时出错: {str(e)}")
                    print(f"\n❌ 执行出错: {str(e)}")
        else:
            # 非交互模式
            prompt = input("请输入您的指令: ")

            if not prompt.strip():
                logger.warning("指令为空")
                return

            print("\n📋 开始处理请求...\n")
            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    flow.execute(prompt),
                    timeout=3600,  # 1小时超时
                )
                elapsed_time = time.time() - start_time

                print("\n========================================")
                print(f"✅ 执行完成! 耗时: {elapsed_time:.2f} 秒")
                print("========================================")
                print(f"\n📝 结果: \n{result}\n")

            except asyncio.TimeoutError:
                logger.error("请求处理超时（1小时）")
                print("\n❌ 执行超时！请尝试更简单的请求。")
            except Exception as e:
                logger.error(f"处理请求时出错: {str(e)}")
                print(f"\n❌ 执行出错: {str(e)}")

    except KeyboardInterrupt:
        logger.info("用户取消操作。")
        print("\n🛑 操作已取消")
    except Exception as e:
        logger.error(f"错误: {str(e)}")
        print(f"\n❌ 系统错误: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description="运行Planning-Act多MCP服务协作工作流")
    parser.add_argument("--interactive", "-i", action="store_true", help="启用交互模式")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_multi_mcp(interactive=args.interactive))
