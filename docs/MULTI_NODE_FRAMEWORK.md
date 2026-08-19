# 多节点测试框架设计与使用指南

## 1. 目标与边界

多节点测试框架由可渲染的 Recipe 模板、Recipe Converter、可执行中间态、Runtime 和
Kubernetes/GitHub Actions 适配器组成。Converter 负责把模板中的测试场景转换成完整 plan；
Runtime 只执行已经展开的 plan，不反向解析 Recipe 或推导业务拓扑。

框架数据流如下：

```text
独立模板 YAML + config_params 默认值 + 可选 --set 覆盖
        │
        │ Converter：精确选择 test_id、渲染参数、静态分析、展开拓扑、校验
        ▼
被 Git 忽略的临时中间态（plan.yaml + plan-local scripts）
        │
        │ Runtime：按 plan 执行，不推导 Recipe 业务语义
        ▼
多节点服务、可选 Gateway、服务检查、AISBench、结果和日志
```

核心原则是：**Converter 负责理解模板，中间态负责显式表达，Runtime 负责可靠执行。**

## 2. 目录结构

```text
models/
├── en/DeepSeek/template_pd.yaml          # PD 模板，Converter 输入
├── en/Qwen/template2_non_pd.yaml         # 非 PD 模板，Converter 输入
├── zh/DeepSeek/template_pd.yaml          # 中文页面镜像，不作为 Converter 输入
└── zh/Qwen/template2_non_pd.yaml         # 中文页面镜像，不作为 Converter 输入

test/recipe/multi_node/
├── convert.py                            # 唯一的转换命令入口
├── converter/                            # Reader、Analyzer、Planner、Emitter
├── .generated/                           # 本地/CI 临时中间态，已加入 .gitignore
└── scripts/                              # Runtime 与 K8s/LWS 适配器
    ├── plan.py                           # 中间态协议读取与校验
    ├── runner.py                         # 节点生命周期与 leader-only stages
    ├── coordinator.py                    # 节点间 ready/stop/outcome 控制面
    ├── process.py                        # 进程组、信号、超时和清理
    ├── result.py                         # 结果协议与原子 JSON 写入
    ├── aisbench.py                       # AISBench 配置、执行和结果转换
    ├── install_aisbench.sh               # 固定版本 AISBench cache
    ├── run_online_dp.py                  # external DP launcher 适配
    ├── run.sh                            # 裸机、容器和 LWS 共用入口
    └── k8s/                              # LWS 模板、渲染和启动脚本

.github/workflows/verify_multi_node.yaml  # matrix 入口
.github/workflows/_verify_multi_node.yaml # reusable workflow
test/ut/multi_node_framework/             # 不依赖 NPU 的框架测试
```

`.generated/` 只用于本地查看或一次 CI 运行，不提交到 Git，也不作为需要维护的基准文件。

## 3. Runtime 架构

### 3.1 节点、Runner 与服务 rank

每个逻辑节点只运行一个 Runner，但一个 Runner 可以管理一个或多个服务进程：

```text
物理节点或 LWS Pod
└── run.sh
    └── runner.py
        └── node.launch 对应的进程组
            ├── 一个 vllm serve（internal DP 常见形式）
            └── 多个 vllm serve（external DP launcher 常见形式）
```

因此“一个节点一份 `service.log`”只表示一个受管 launcher，并不表示该节点只有一个 DP rank。
external DP 可以在 launcher 内为每个 rank 单独输出 `servers/rank-<rank>.log`；internal DP 的 ranks
由同一个 `vllm serve` 管理，通常共享 launcher 日志。

DP/TP、rank、设备映射、KV Connector、服务端口和 vLLM 参数全部由生成脚本明确给出，Runner
不重新推导。

### 3.2 控制面与数据面

`plan.nodes[0]` 是控制 leader。leader 上的 Coordinator 只交换：

- 节点是否 ready；
- 第一个全局 stop signal；
- 清理完成后的 `NodeOutcome`；
- leader 聚合的 `RunOutcome`。

模型请求、HCCL/Gloo、KV 数据、Gateway 流量和服务日志属于数据面，不经过 Coordinator。

所有节点都会加载同一 plan、启动本节点服务、检查本地 readiness、上报 ready，并在 stop 后清理
本节点进程组。只有 leader 会等待全组 ready、启动可选 Gateway、运行检查和评测、发布 stop，
最后聚合全局结果。

### 3.3 生命周期与失败收敛

```text
load plan/hosts
  -> start/wait coordinator
  -> start local launcher
  -> poll local rank readiness
  -> wait all nodes ready
  -> optional gateway readiness
  -> ordered stages
  -> publish first stop signal
  -> TERM process groups
  -> bounded wait
  -> KILL remaining groups
  -> report NodeOutcome
  -> leader writes RunOutcome
```

StopSignal 只是提前通知，不是最终结果。即使评测已经通过，只要服务进程无法正确清理，最终结果
仍然失败。观察到其他节点首先失败的节点记为 `aborted`，不会被误判成根因。

服务、Gateway 和 stage 都使用独立进程组；清理不会使用可能影响同机其他任务的全局 `pkill` 或
`killall`。

## 4. 临时可执行中间态

### 4.1 默认位置和结构

Converter 默认把 bundle 写入：

```text
test/recipe/multi_node/.generated/<recipe-stem-kebab>/<test-id>/
├── plan.yaml
├── nodes/
│   ├── node0/run.sh
│   └── nodeN/run.sh
├── gateway/run.sh                    # 可选
├── checks/
├── evaluations/
└── README.md                         # 仅记录本次生成来源
```

例如：

```text
.generated/template-pd/pd-2n2c/
.generated/template2-non-pd/dp-2n2c/
```

目录第一层来自模板文件名：去掉扩展名、转小写，并把非字母数字序列转换为 `-`；第二层使用
`test_id`。保留 `test_id` 子目录可以避免同一模板将来包含多个测试时互相覆盖。

这些文件是可再生的临时产物：

- 本地生成后可以查看和直接验证；
- CI checkout 后重新生成，再复制到 PVC 供 Pod 使用；
- 不提交到仓库，不要求维护生成文件 drift；
- 不应手工修改，修改应回到模板或 Converter。

### 4.2 `plan.yaml` 协议

当前协议是 `api_version: multi-node/v1`、`kind: MultiNodePlan`。Runtime 使用：

| 区域 | 内容 | 用途 |
| --- | --- | --- |
| `metadata` | plan 名称、模板、`test_id`、摘要 | 追踪本次转换来源 |
| `model` | 模型 ID、served name | 定位权重并构造请求 |
| `resources` | 每节点 NPU 数 | LWS resource request/limit |
| `nodes[]` | id、role、launch、可选 readiness | 节点顺序、启动和健康检查 |
| `gateway` | 可选 launch、port、health path | leader 侧统一入口 |
| `stages[]` | stage、step、超时和 inputs | leader-only 检查与评测 |

Runtime 加载时会校验协议版本和 kind、必需字段类型、非空且不重复的节点 ID、正整数资源与端口、
readiness 端口范围、leader endpoint，以及所有 launch/step 脚本确实存在于 plan 目录内。没有
Gateway 时 leader 必须声明 readiness。该校验用于阻止损坏或过期的 bundle 进入执行阶段，不重复
Converter 对 DP/TP、PD 角色、KV Connector 和 Gateway backend 等业务拓扑的推导。

`nodes[0]` 固定为 leader。节点数组顺序同时决定 hosts、LWS worker index 和
`MULTI_NODE_NODE_<index>_IP`，因此 Converter 必须稳定输出节点顺序。

Node readiness 可声明连续 HTTP 端口范围。没有独立 HTTP endpoint 的 internal-DP headless
节点可以省略 readiness；其 launcher 需要持续存活，最终连通性由 API 节点和服务检查确认。

Gateway 是 leader 上的普通受管进程，不是 Runtime 内置的 PD 特例。Gateway 存在时 stage 使用
Gateway endpoint，否则使用 API/leader endpoint。

每个 step 接收 `MULTI_NODE_STEP_INPUT_FILE`、`MULTI_NODE_STEP_ARTIFACT_DIR` 和
`MULTI_NODE_STEP_RESULT_FILE`。step 必须写出 JSON object，至少包含
`{"status": "passed"}`；AISBench 适配器负责把工具私有输出转换成这个公共协议。

## 5. Converter 输入契约

### 5.1 当前支持的模板

当前 Converter 明确只接受以下两个组合：

| 拓扑 | `--recipe` | `--test-id` |
| --- | --- | --- |
| PD external DP | `models/en/DeepSeek/template_pd.yaml` | `pd-2n2c` |
| 非 PD internal DP | `models/en/Qwen/template2_non_pd.yaml` | `dp-2n2c` |

传入其他 Recipe，或者给模板传入不匹配的 `test_id`，转换会立即失败。新增模型或拓扑时，需要
先按模板契约准备文档，再为其显式扩展 Converter 的识别和规划逻辑。

中文模板用于页面渲染和 schema 镜像。Converter 只读取上表中的英文模板，避免双语内容成为两份
运行事实来源。

### 5.2 场景字段

脚本型模板场景至少包含：

| 字段 | 约束 | 转换用途 |
| --- | --- | --- |
| `test_id` | 当前模板内唯一，小写 kebab-case | 精确选择测试 |
| `npu` | 非空字符串 | 目标硬件语义 |
| `deployment` | 精确为 `pd` 或 `non-pd` | 选择 planner |
| `case` | PD 为 `<P>p<D>d`；非 PD 为 `<N>-node` | 节点和角色数量 |
| `npu_per_node` | 正整数 | 每节点资源请求 |
| `aisbench` | 可选且不重复的 `accuracy` / `performance` | 选择公共评测 stage |
| `scripts` | 脚本映射，包含 `service-check` | 提取服务、Gateway 和检查 |

`precision`、`steps`、`meta`、`model` 和 `variants` 继续服务页面，当前 Converter 不依赖它们来
推导运行拓扑。实际模型从渲染后的 `vllm serve` 第一个参数提取；served name、DP/TP/rank、服务
端口、RPC 端口、KV 配置和 Gateway endpoints 同样从受支持命令静态提取，不再设置重复字段。

节点脚本使用零基角色编号，例如 `prefill-0-template`、`decode-0-launch`、`api-0` 和
`headless-0`。这样多个 Prefill、Decode 或 Headless 节点仍可唯一定位。端到端检查使用无节点
编号的 `service-check`。

### 5.3 页面嵌入与参数

正文通过 `{{script:name}}` 引用 `scripts` 中的完整脚本。前端把脚本按其 `language` 渲染成代码
块；Converter 则直接读取同一个脚本对象进行静态分析。只用于说明的普通命令继续写在
`steps[].content`，不需要为了 Converter 拆成独立字段。

`{{max_model_len}}` 等值占位符的来源与前端一致：

1. 读取模板顶层 `config_params`；
2. 用选中 scenario 的同名 `config_params` 覆盖；
3. 提取合并结果中的标量 `default`；
4. 最后应用可重复传入的 `--set name=value` 临时覆盖。

缺少脚本需要的参数、覆盖值无法解析，或者传入未被脚本使用的 `--set`，转换都会失败。
`$API_NODE_0_IP`、`${ASCEND_RT_VISIBLE_DEVICES}` 等 Shell 环境变量不是模板值占位符，Converter
会保留它们，Runtime 再注入真实节点信息。

### 5.4 转换顺序

```text
verify supported template + test_id pair
  -> parse YAML and select exact scenario
  -> merge config_params defaults
  -> apply optional --set overrides
  -> expand value placeholders
  -> static-analyze supported shell commands
  -> build PD or non-PD topology
  -> validate resource/rank/port/endpoint relationships
  -> emit complete candidate bundle
  -> bash -n generated shell scripts
  -> load plan with the real Runtime loader
  -> atomically replace this case's temporary output
```

Converter 不执行模板中的 Shell，也不依赖环境变量或临时文件在内部阶段间传递业务状态。

输出前会检查字段和 `test_id`、占位符、模型/served name 一致性、节点编号、DP/TP/rank、本地 rank
范围、NPU 预算、Mooncake 两侧角色和拓扑、Gateway endpoints、端口范围、readiness 数量、
AISBench 选择、路径逃逸和 Shell 语法。Runtime 仍保留文件、hosts、进程、HTTP 与结果 JSON 的
防御性检查，但不重复实现 Recipe 语义校验。

### 5.5 CLI 和默认输出

生成 PD 模板：

```bash
.venv/bin/python test/recipe/multi_node/convert.py \
  --recipe models/en/DeepSeek/template_pd.yaml \
  --test-id pd-2n2c
```

生成非 PD 模板：

```bash
.venv/bin/python test/recipe/multi_node/convert.py \
  --recipe models/en/Qwen/template2_non_pd.yaml \
  --test-id dp-2n2c
```

命令会打印实际生成目录。`--output` 是可选的本地调试参数，但目标仍必须位于
`test/recipe/multi_node/.generated/` 内；正常本地和 CI 流程都使用默认目录。临时覆盖示例：

```bash
.venv/bin/python test/recipe/multi_node/convert.py \
  --recipe models/en/Qwen/template2_non_pd.yaml \
  --test-id dp-2n2c \
  --set max_num_seqs=4
```

## 6. 本地或裸机验证

### 6.1 只验证生成和 plan 加载

先运行第 5.5 节的转换命令，再执行：

```bash
MULTI_NODE_PLAN=test/recipe/multi_node/.generated/template-pd/pd-2n2c/plan.yaml \
MULTI_NODE_VALIDATE_ONLY=true \
test/recipe/multi_node/scripts/run.sh
```

该模式不启动服务、不访问 NPU，也不需要 hosts，用于快速确认生成 bundle 能被真实 Runtime
加载。它不能替代真实多节点服务验证。

### 6.2 真实执行环境变量

每个节点至少设置：

| 环境变量 | 含义 |
| --- | --- |
| `MULTI_NODE_PLAN` | 生成的 `plan.yaml` 路径 |
| `MULTI_NODE_CLUSTER_IPS` | 按 `plan.nodes` 顺序排列的逗号分隔节点地址 |
| `MULTI_NODE_NODE_INDEX` | 当前节点在 `plan.nodes` 中的零基索引 |
| `ASCEND_RT_VISIBLE_DEVICES` | 当前节点分配的物理 NPU |

建议显式设置 `MULTI_NODE_INTERFACE` 为本节点 HCCL/Gloo 网卡。未设置时 Runner 根据当前节点 IP
反查网卡，无法唯一确定则失败。

常用可选项：

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MULTI_NODE_CONTROL_PORT` | `29599` | Coordinator 端口 |
| `MULTI_NODE_STARTUP_TIMEOUT_SECONDS` | `1800` | 全链路启动总期限 |
| `MULTI_NODE_RUN_TIMEOUT_SECONDS` | `7200` | leader stages 总期限 |
| `MULTI_NODE_PROGRESS_INTERVAL_SECONDS` | `30` | GitHub/终端心跳间隔，`0` 表示关闭 |
| `MULTI_NODE_ARTIFACT_ROOT` | `/tmp/multi-node` | 结果和日志目录 |
| `MULTI_NODE_PLOG_ROOT` | 未设置 | 设置后退出时复制 Ascend plog |
| `VLLM_ASCEND_ROOT` | `/vllm-workspace/vllm-ascend` | 上游 launcher/proxy 根目录 |

两台机器设置相同 `MULTI_NODE_PLAN` 和 `MULTI_NODE_CLUSTER_IPS`，分别把
`MULTI_NODE_NODE_INDEX` 设为 `0`、`1` 后运行同一个 `test/recipe/multi_node/scripts/run.sh`。
启动顺序没有要求：node0 建立 Coordinator，其他节点在共享 startup deadline 内等待。

### 6.3 AISBench

本地 `run.sh` 不隐式联网安装 AISBench。包含评测 stage 时先执行：

```bash
AIS_BENCH_ENVIRONMENT_IDENTITY='runtime=<image identity>' \
test/recipe/multi_node/scripts/install_aisbench.sh \
  --env-file /tmp/multi-node-aisbench.env

source /tmp/multi-node-aisbench.env
export MULTI_NODE_AISBENCH_BIN
export MULTI_NODE_AISBENCH_CACHE_KEY
export MULTI_NODE_AISBENCH_SOURCE
```

Installer 使用固定版本和带运行环境 identity 的 cache key。冷 cache 在临时目录构建并原子发布，
只有命令、数据集和 API 模板校验通过才复用。K8s 使用集群内部 PyPI；本地使用调用者已有 pip
配置。

## 7. GitHub Actions 流程

入口 workflow 的 matrix 每个 case 只声明三个字段：

```yaml
matrix:
  include:
    - name: deepseek-v2-lite-pd-2n2c
      recipe: models/en/DeepSeek/template_pd.yaml
      test_id: pd-2n2c
    - name: qwen3-30b-a3b-dp-2n2c
      recipe: models/en/Qwen/template2_non_pd.yaml
      test_id: dp-2n2c
```

- `name` 只用于 job、LWS、日志和 artifact 展示；
- `recipe` 指向独立模板；
- `test_id` 精确选择模板场景。

Reusable workflow 的顺序是：

1. checkout；
2. 安装 Converter 所需控制面依赖；
3. 运行 `convert.py --recipe ... --test-id ...`，使用默认输出目录；
4. 从 Converter 输出获得 `plan.yaml` 路径并读取节点/NPU 数；
5. 把包含 `.generated/` 的当前 workspace 复制到本次 PVC source；
6. 渲染和启动 LeaderWorkerSet；
7. Pod 执行 `run_lws.sh -> run.sh -> runner.py` 业务流程；
8. 收集结果和日志并清理 LWS/PVC 本次目录。

因此 workflow 不需要传 output、plan、模型路径、拓扑或参数文件。中间态也不会跨 workflow 运行
复用；每次都从当前 checkout 的模板和 Converter 重新生成。

LWS Pod 在 15 分钟总期限内并行等待创建和 Ready；Pod 内 Coordinator、service、全组 ready 和
Gateway 共享 startup deadline，stages 共享 run deadline。

每个 matrix case 是独立的 reusable-workflow job，并使用包含 case name 的 concurrency group。
同一 case 的重复运行会串行，不同 case 可以并行提交各自具有唯一名称的 LWS 和运行目录。每个
LWS 内使用匹配自身 `multi-node-run` 标签的 pod anti-affinity，确保同一测试的不同逻辑节点不会
落到同一物理节点；它不会把不同 case 绑定到同一个调度组。

每个节点轮询自己的服务 endpoint：状态变化时立即打印，未 Ready 时按
`MULTI_NODE_PROGRESS_INTERVAL_SECONDS` 打印心跳。leader 额外打印全组 ready 进度。Stage 原始
stdout/stderr 始终完整写入日志文件，GitHub 控制台打印开始、心跳、退出状态和验证后的结果摘要；
成功和失败使用同一日志策略。

完整 Kubernetes YAML 可能包含节点、IP、镜像、volume 和环境变量，默认不上传。
`MULTI_NODE_UPLOAD_K8S_DIAGNOSTICS="true"` 只应在确认 artifact/OBS 权限后显式开启。
日志和结果复制到本地 artifact bundle 后，无论 OBS/GitHub 上传是否成功，本次 run source、日志和
状态文件都会从共享 PVC 删除；上传故障不能成为残留运行目录的保留机制。

## 8. 结果与日志

```text
<artifact-root>/<plan-name>/
├── node0/
│   ├── service.log
│   ├── servers/rank-0.log           # external DP 可按 rank 拆分
│   ├── gateway.log                  # 可选
│   ├── <stage>/<step>.log
│   ├── <stage>/<step>/input.json
│   ├── <stage>/<step>/result.json
│   └── node-result.json
├── node1/
└── result.json                      # 仅 leader
```

Runtime 不根据成功或失败切换日志采集策略。服务、调度、Gateway 和 AISBench 日志始终按同一规则
保存；GitHub 控制台保持可观察心跳和简洁结果，避免重复灌入完整日志。

## 9. 测试与后续扩展

Converter 和 Runtime 的本地测试不依赖 NPU，主要覆盖：

- 只接受两个模板和固定 `test_id`；
- `config_params` 合并、`--set` 覆盖和缺失/多余参数；
- PD external DP 与非 PD internal DP 静态转换；
- 节点、rank、资源、端口、KV 和 Gateway 负例；
- 安全临时输出、原子替换、Shell 语法和 Runtime 协议/路径负例；
- 中英文模板的运行字段、完整脚本和有效参数默认值一致；
- Coordinator、进程组、deadline、结果协议、日志与 workflow 契约。

测试围绕 Converter、Runtime 和基础设施适配器的公开契约编写。需要长期防止回归的行为使用
正向输出、结构化 YAML/JSON 或真实子进程结果验证。

新增模型或拓扑时建议按以下顺序推进：

1. 从相应独立模板复制并完成页面内容；
2. 明确新的模板路径、`test_id` 和固定字段约束；
3. 为新拓扑增加独立 analyzer/planner adapter，不执行任意 Shell；
4. 增加正反例单测和 validate-only；
5. 再把 `name`、`recipe`、`test_id` 加入 workflow matrix；
6. 最后在真实 NPU CI 验证生成 bundle。

Runtime core 不导入 Kubernetes SDK，不读取 GitHub Actions context；Infrastructure adapter 不理解
PD、DP/TP 或 AISBench 指标；plan-local scripts 不解析 Recipe；Coordinator 不传输日志或模型
数据。新增实现应继续保持这些边界。
