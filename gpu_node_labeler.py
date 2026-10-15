"""
模块名称: gpu_node_labeler
功能描述:
    作为 DaemonSet 运行在 Kubernetes 节点上的轻量级 GPU 探测与打标 Agent.
    本脚本通过 NVML (NVIDIA Management Library) 获取宿主机的 GPU 硬件信息(型号、卡数、显存、驱动版本等),
    对数据进行 Kubernetes Label 规范化清洗, 并通过 In-Cluster RBAC 凭据将对应信息作为 Label 自动打在当前 Node 资源上.

环境依赖:
    - python >= 3.8
    - pynvml (nvidia-ml-py)
    - kubernetes (Python SDK)
    - 宿主机需挂载或注入 NVIDIA Utility 运行时(提供 libnvidia-ml.so)
"""

import os
import re
import sys
import time
import logging
from typing import Dict, Optional
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import pynvml

# ==============================================================================
# 日志系统初始化
# ==============================================================================
# 支持通过环境变量 LOG_LEVEL 动态调节日志级别(默认为 INFO, 开发/排错可设为 DEBUG)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(process)d] [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,  # 确保容器日志由 stdout 输出以适配 containerd/dockerd 收集
)
logger = logging.getLogger("gpu-node-labeler")

# ==============================================================================
# 常量定义
# ==============================================================================
# 节点标签的前缀 Domain 规范, 建议遵循 Kubernetes 推荐的 DNS 子域格式
LABEL_PREFIX: str = os.getenv("LABEL_PREFIX", "gpu.custom.io")

# 单个 Kubernetes Label Value 的最大允许字符长度
MAX_LABEL_VALUE_LEN: int = 63


def sanitize_label_value(val: str) -> str:
    """
    清洗并规范化字符串, 使其完全契合 Kubernetes Label Value 的严格规范.

    Kubernetes Label 规范要求 (RFC 1123 / RFC 952):
    1. 长度不能超过 63 个字符.
    2. 可以为空, 或者仅由字母数字字符 ([a-z0-9A-Z])、中划线 (-)、下划线 (_) 或英文点 (.) 组成.
    3. 必须以字母数字字符 ([a-z0-9A-Z]) 开头和结尾.

    :param val: 原始字符串(如显卡型号: "NVIDIA A100-PCIE-40GB")
    :return: 清洗后的合规字符串(如: "NVIDIA-A100-PCIE-40GB")
    """
    if not val:
        logger.debug("输入值为空, 直接返回空字符串")
        return ""

    original_val = val
    # 步骤 1: 移除首尾空白, 并将所有非法字符(包含空格、特殊标点)替换为中划线 "-"
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "-", val.strip())

    # 步骤 2: 将连续多个中划线折叠合并为单个中划线, 提高标签可读性(如 "A100--PCIE" -> "A100-PCIE")
    cleaned = re.sub(r"-+", "-", cleaned)

    # 步骤 3: 去除首尾可能因替换残留的非字母数字字符(Label 要求必须以字母或数字起止)
    cleaned = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", cleaned)

    # 步骤 4: 强制截断至 K8s 允许的最大长度 63 位
    if len(cleaned) > MAX_LABEL_VALUE_LEN:
        logger.warning(
            f"标签值超过 {MAX_LABEL_VALUE_LEN} 字符限制, 已被截断! 原始值: '{original_val}', 截断后: '{cleaned[:MAX_LABEL_VALUE_LEN]}'"
        )
        cleaned = cleaned[:MAX_LABEL_VALUE_LEN]
        # 截断后可能导致末尾再次出现非法字符, 进行末尾收缩
        cleaned = re.sub(r"[^a-zA-Z0-9]+$", "", cleaned)

    logger.debug(f"Label 清洗完成: 原始值='{original_val}' -> 规范值='{cleaned}'")
    return cleaned


def get_gpu_info() -> Optional[Dict[str, str]]:
    """
    通过 NVML 驱动接口查询当前主机的 GPU 硬件和驱动信息.

    数据采集策略:
    - 若驱动未加载或硬件无 GPU, 捕获异常并返回 present=false 标签, 通知调度器此节点非 GPU 节点.
    - 采集的指标包括: 设备数量、驱动版本、首卡型号名称、首卡总显存(单位 GiB).
    - 假定同一计算节点内搭载的 GPU 卡型具有同质性(Homogeneous), 以 0 号卡代表整机卡型.

    :return: 包含待打标键值对的字典; 若发生未知硬件查询错误, 返回 None 以放弃本次打标.
    """
    logger.debug("开始尝试初始化 NVML 驱动库...")
    try:
        pynvml.nvmlInit()
        logger.debug("NVML 初始化成功")
    except pynvml.NVMLError as e:
        # 典型场景: 当前 Pod 运行在纯 CPU 节点, 或宿主机驱动崩溃/未安装 Container Toolkit
        logger.warning(
            f"NVML 初始化失败(该节点可能无 NVIDIA GPU 硬件或未安装驱动), 错误详情: {e}"
        )
        return {f"{LABEL_PREFIX}/present": "false"}

    try:
        # 获取物理/虚拟 GPU 实例的总数
        device_count = pynvml.nvmlDeviceGetCount()
        logger.info(f"NVML 探测到当前节点 GPU 设备总数: {device_count}")

        # 若物理卡数为 0, 标记 present 为 false
        if device_count == 0:
            logger.info("探测到的 GPU 卡数为 0, 标记为非 GPU 节点")
            return {f"{LABEL_PREFIX}/present": "false"}

        # 获取主机当前加载的 NVIDIA 显卡驱动版本
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        # 兼容性处理: 不同版本的 nvidia-ml-py 可能返回 bytes 或 str
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8")
        logger.debug(f"获取到底层驱动版本: {driver_version}")

        # 获取 Index 0 设备句柄(默认以 0 号卡代表整机硬件规格)
        logger.debug("正在查询 0 号 GPU 设备句柄及硬件元数据...")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # 获取产品型号名称(如 "NVIDIA A100-PCIE-40GB" 或 "Tesla T4")
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        logger.debug(f"获取到 0 号 GPU 型号: {gpu_name}")

        # 获取显存总量并换算为 GiB (1024^3 Bytes)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_total_gb = round(mem_info.total / (1024**3))
        logger.debug(f"0 号 GPU 总显存: {mem_info.total} 字节 (~{mem_total_gb} GiB)")

        # 构造需要同步至 Node 资源的 Label 字典
        labels = {
            f"{LABEL_PREFIX}/present": "true",
            f"{LABEL_PREFIX}/count": str(device_count),
            f"{LABEL_PREFIX}/model": sanitize_label_value(gpu_name),
            f"{LABEL_PREFIX}/memory-gb": str(mem_total_gb),
            f"{LABEL_PREFIX}/driver-version": sanitize_label_value(driver_version),
        }

        logger.debug(f"成功构造节点 GPU 标签数据集: {labels}")
        return labels

    except pynvml.NVMLError as e:
        logger.error(f"调用 NVML 查询 GPU 状态时发生错误: {e}", exc_info=True)
        return None
    finally:
        # 无论成功或失败, 严格释放 NVML 资源句柄, 避免显存管理句柄泄漏
        try:
            logger.debug("正在注销并释放 NVML 上下文...")
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.debug(f"关闭 NVML 时触发静默异常(通常可忽略): {e}")


def patch_node_labels(node_name: str, labels_to_apply: Dict[str, str]) -> None:
    """
    通过 Kubernetes CoreV1 API 为目标 Node 打上指定的 Labels.

    性能优化与防抖:
    - 优先拉取 Node 现有的 Labels.
    - 比对目标 Labels 与现有 Labels, 仅在发生属性漂移(缺失、值变更)时发起 PATCH 请求.
    - 降低对 kube-apiserver 的写压力, 防止造成 etcd 频繁写膨胀和集群事件雪崩.

    :param node_name: 待修改的目标 K8s 节点名称(由 Pod Downward API 注入)
    :param labels_to_apply: 待应用的标准化标签字典
    """
    logger.debug(f"开始准备向 APIServer 校验并提交节点 [{node_name}] 的标签变更...")

    # 1. 认证载入: 集群内部 Pod 默认加载 ServiceAccount Token, 外部调试则加载 ~/.kube/config
    try:
        config.load_incluster_config()
        logger.debug("成功加载 In-Cluster Kubernetes 认证凭据")
    except Exception as e:
        logger.debug(f"加载 In-Cluster 认证失败, 尝试回退加载本地 kubeconfig: {e}")
        try:
            config.load_kube_config()
            logger.debug("成功加载本地 kubeconfig 凭据")
        except Exception as kube_err:
            logger.critical(
                f"无法初始化 Kubernetes 客户端配置, 无法连接集群: {kube_err}"
            )
            return

    v1 = client.CoreV1Api()

    # 2. 读取当前节点现有的 Label 集
    try:
        logger.debug(f"正在从 API 查询节点 [{node_name}] 的最新规格与元数据...")
        node = v1.read_node(name=node_name)
        existing_labels = node.metadata.labels or {}
    except ApiException as e:
        logger.error(
            f"读取节点 [{node_name}] 失败, HTTP 状态码: {e.status}, 原因: {e.reason}"
        )
        return
    except Exception as e:
        logger.error(
            f"读取节点 [{node_name}] 发生未知网络/连接异常: {e}", exc_info=True
        )
        return

    # 3. 标签差异比对 (Diff Check)
    diff_detected = False
    diff_details = {}

    for target_k, target_v in labels_to_apply.items():
        current_v = existing_labels.get(target_k)
        if current_v != target_v:
            diff_detected = True
            diff_details[target_k] = {"current": current_v, "target": target_v}

    if not diff_detected:
        logger.info(
            f"节点 [{node_name}] 现有 GPU 标签与探测结果完全一致, 跳过本次 PATCH."
        )
        return

    logger.info(f"检测到节点 [{node_name}] 标签存在漂移, 差异清单: {diff_details}")

    # 4. 构造 Strategic Merge Patch 载荷并执行更新
    body = {"metadata": {"labels": labels_to_apply}}

    try:
        logger.debug(
            f"正在向 APIServer 发送 Strategic Merge Patch 请求, Payload: {body}"
        )
        # patch_node 采用 application/strategic-merge-patch+json, 只更新 metadata.labels 对应 key, 不覆盖其他已有标签
        v1.patch_node(name=node_name, body=body)
        logger.info(f"成功为节点 [{node_name}] 同步并写入 GPU 标签: {labels_to_apply}")
    except ApiException as e:
        logger.error(
            f"向节点 [{node_name}] 提交 Patch 失败, 可能缺少 RBAC nodes/patch 权限. HTTP: {e.status}, 详情: {e.body}"
        )
    except Exception as e:
        logger.error(
            f"向节点 [{node_name}] 提交 Patch 发生未捕获异常: {e}", exc_info=True
        )


def main() -> None:
    """
    Agent 运行主入口: 执行环境变量初始化校验, 并维系常驻周期的健康检测与打标循环.
    """
    logger.info("=" * 60)
    logger.info("Kubernetes GPU Node Labeler Agent 正在启动...")
    logger.info("=" * 60)

    # 1. 获取所在节点名称(强烈依赖 Downward API spec.nodeName)
    node_name = os.getenv("NODE_NAME")
    if not node_name:
        logger.critical(
            "关键环境变量 NODE_NAME 未配置! 无法确认需要打标的目标节点."
            "请检查 DaemonSet yaml 是否配置了 spec.containers[*].env 关联 spec.nodeName."
        )
        sys.exit(1)

    # 2. 获取同步周期配置(默认 300 秒探测并校验一次)
    interval_raw = os.getenv("SYNC_INTERVAL_SECONDS", "300")
    try:
        sync_interval = int(interval_raw)
        if sync_interval <= 0:
            raise ValueError("时间间隔必须大于 0")
    except ValueError as e:
        logger.warning(
            f"环境变量 SYNC_INTERVAL_SECONDS='{interval_raw}' 不合法 ({e}), 重置为默认值 300 秒."
        )
        sync_interval = 300

    logger.info(
        f"Agent 配置就绪: 目标 Node='{node_name}', 同步轮询周期={sync_interval}s"
    )

    # 3. 常驻巡检循环
    while True:
        try:
            logger.debug("--- 开始执行新一轮 GPU 探测巡检任务 ---")
            gpu_labels = get_gpu_info()

            if gpu_labels is not None:
                patch_node_labels(node_name, gpu_labels)
            else:
                logger.warning("本轮未采集到有效的 GPU 标签信息, 跳过本轮打标.")

        except Exception as loop_err:
            # 顶级保护: 防止运行时未处理异常击穿主线程导致 DaemonSet 崩溃陷入 CrashLoopBackOff
            logger.error(f"主巡检循环内发生未捕获异常: {loop_err}", exc_info=True)

        logger.debug(f"本轮处理完毕, Agent 线程挂起, 等待 {sync_interval} 秒...")
        time.sleep(sync_interval)


if __name__ == "__main__":
    main()
