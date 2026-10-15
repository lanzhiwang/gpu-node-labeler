# Kubernetes GPU Node Labeler

`k8s-gpu-node-labeler` 是一个基于 Python 与 `nvidia-ml-py` 开发的轻量级 Kubernetes 节点打标 Agent.

它以 DaemonSet 模式运行在集群各节点上, 周期性通过底层 NVML (NVIDIA Management Library) 获取宿主机的 GPU 硬件及驱动信息, 并自动为 Kubernetes Node 资源打上标准化的标签(Labels). 运维人员和调度器(如 K8s Default Scheduler / Volcano / YuniKorn)可借此实现精细化的异构算力调度与任务路由.

---

## 🌟 核心特性

- 云原生无侵入驱动适配: 利用 NVIDIA Container Toolkit 的 `utility` 驱动挂载能力, 容器内无需映射宿主机动态库路径, 即可直接使用 NVML.
- 不占用 GPU 调度配额: 通过环境变量向容器开放 GPU 设备访问权限, 无需在 Pod 资源申请中配置 `resources.limits: nvidia.com/gpu`, 避免浪费业务 GPU 配额.
- Label 规范自动清洗: 自动清洗 GPU 型号中的空格与特殊字符, 严格遵循 Kubernetes Label Value 规范(RFC 1123, 正则约束, 长度截断至 63 字符).
- 智能防抖与 Diff 比对: 采用 Strategic Merge Patch 机制, 仅在标签发生漂移或缺失时才向 APIServer 发起写请求, 有效保护 `kube-apiserver` 和 `etcd`.
- CPU 节点优雅兼容: 无 GPU 或未安装 NVIDIA 驱动的节点将被标记为 `present=false`, 不会造成 DaemonSet 崩溃陷入 `CrashLoopBackOff`.

---

## 🏷️ 节点标签规划

Agent 默认在节点上注入以下 Labels(前缀可通过环境变量自定义):

| 标签 Key                       | 示例值                    | 说明                              |
| :----------------------------- | :------------------------ | :-------------------------------- |
| `gpu.custom.io/present`        | `"true"` / `"false"`      | 当前节点是否存在可用 GPU          |
| `gpu.custom.io/count`          | `"8"`                     | 节点物理/虚拟 GPU 总卡数          |
| `gpu.custom.io/model`          | `"NVIDIA-A100-SXM4-80GB"` | GPU 产品型号(已规范化清洗)        |
| `gpu.custom.io/memory-gb`      | `"80"`                    | 单卡显存大小(单位: GiB, 四舍五入) |
| `gpu.custom.io/driver-version` | `"535.129.03"`            | 宿主机当前安装的 NVIDIA 驱动版本  |
