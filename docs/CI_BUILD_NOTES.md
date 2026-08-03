# vllm-ascend-recipes CI/CD 建设踩坑记录

> 目标：为 `vllm-ascend-recipes` 仓库搭建 Recipe 端到端验证流水线，覆盖 YAML 契约校验 → 页面可渲染 → 模型可部署 → aisbench 性能测试 → OBS 留证。

## 流水线架构

```
PR 合入前:
  models/** 变更 → detect(路径识别) → validate(YAML+i18n+Build) → verify-deploy(A2 runner)
    → CANN容器 → 解析recipe → vllm serve → curl验证 → aisbench → 停服务 → 上传OBS

合入后:
  定时/推送/手动 → nightly-recipe-verify → 扫描全部recipe → 按硬件分类 → 全量验证
```

## 踩坑记录

### 一、镜像与依赖

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 1 | `quay.io/ascend/vllm-ascend` 拉不到 | CI 集群网络不通 quay.io | 改用 `swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/vllm-ascend:v0.23.0rc1` |
| 2 | SWR `:latest` 标签实际指向 vllm 0.11.0 | latest 标签未更新，不支持 `--tokenizer-mode deepseek_v4` | 显式指定 `v0.23.0rc1` 标签 |
| 3 | CANN 镜像 `pip install vllm vllm-ascend` 后 `vllm` CLI 找不到 | vllm 模块可 `import` 但二进制不在 `PATH` | `hash -r` + `export PATH="/usr/local/bin:/root/.local/bin:$PATH"`；或创建 `/usr/local/bin/vllm` wrapper 脚本 |
| 4 | `pip install ais_bench_benchmark` SSL 证书错误 | 未配 pip 内部镜像，直连 PyPI 失败 | 先执行 `pip config set global.index-url http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple` |

### 二、容器与环境

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 5 | `apt-get update` 卡死 | vllm-ascend 镜像已预装所有包，apt 走缓存服务器不通 | 删除 `apt-get install` 步骤，依赖镜像自带 |
| 6 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 不生效 | 每个 `step` 是独立 shell，环境变量不跨 step 传递 | 在 verify step 内部 source（同一 shell 上下文） |

### 三、Runner 与网络

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 7 | Runner offline | 上轮 job 跑完 runner 进程断连 | 到 runner 机器重启 GitHub Actions runner service |
| 8 | `gh-proxy.test.osinfra.cn` 502 Bad Gateway | Git 代理服务挂了 | 运维修复代理 |
| 9 | Fork PR 无法使用主仓 self-hosted runner | GitHub Actions 安全策略禁止 fork 访问主仓 runner | 合并 PR 后在主仓分支测试 |

### 四、Recipe 解析

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 10 | `Found 0 scenario(s)` | recipe 用 ` ```shell ` 而非 ` ```bash ` | 正则匹配两者：`r'```(?:bash\|shell)...'` |
| 11 | `%%CONFIG:prefix-caching%%` 未被去除 | `\w+` 不匹配连字符 `-`，导致 `prefix-caching` 无法匹配 | 改为 `[^%]+` 匹配除 `%` 外的所有字符 |
| 12 | 多行命令被截断（JSON 参数丢失） | 逐行解析只拿首行，`\` 续行后的 JSON 内容丢失 | **提取整个 bash code block 内容**而非逐行拼接 |
| 13 | `your_model_path` / `your_eagle3_model_path` 占位符未替换 | recipe 使用占位符作为示例 | Python 端 `replace` 为实际权重路径；eagle3 行直接删除 |
| 14 | A3 场景在 A2 runner 上执行报错 | 未按硬件过滤场景 | 加入 `if hw_key == 'atlas_800_a2' and 'A3' in s.get('npu', '')` 跳过 |
| 15 | 多节点 PD 分离场景报 `--port $2` | 位置参数未赋值（`$2` `$3` 等） | 检测 `$[0-9]` 跳过所有多节点场景 |

### 五、命令执行

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 16 | `eval "$cmd"` 特殊字符报错 | 多行命令含引号、`$` 符号、转义字符等 | `cat` 拼接临时脚本文件，`bash script.sh &` 执行 |
| 17 | `set -u` 导致 `LD_PRELOAD: unbound variable` | recipe 中 `export LD_PRELOAD=...:$LD_PRELOAD` 引用未定义变量 | temp 脚本用 `set -eo pipefail` 代替 `set -euo pipefail` |
| 18 | curl 多行命令拆成单行执行失败 | `while read` 逐行处理，每一行单独 eval | 写入临时脚本整体执行（同 #16） |
| 19 | 全局 `verification` 字段的 curl 未被执行 | parser 只扫描 `scenarios[].steps[]` | 额外提取顶层 `verification` 字段，追加到每个 scenario 的 verify_cmds |

### 六、退出与流程控制

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 20 | `exit $STATUS` 不生效，流水线一直 running | `echo \| while read` 管道在子 shell 中执行，变量修改不传给父 shell | Python 写元数据到文件，shell 用 `done < /tmp/scenario_list.txt` 文件重定向读取 |
| 21 | vllm serve 进程杀不掉 | `kill $PID` 只杀 bash 父进程，Python 子进程残留 | `kill -TERM -- -$pid` 杀整个进程组，`kill -KILL` 兜底 |
| 22 | curl 验证后服务被杀了但 benchmark 没跑 | 代码顺序：kill 在 benchmark 之前 | 调整为：**serve → curl → benchmark → kill** |

### 七、Workflow 文件

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 23 | GitHub 报 "This run likely failed because of a workflow file issue" | YAML 缩进不一致（OBS upload step 7空格 vs with 8空格） | 统一为 8 空格缩进 |
| 24 | `devops-actions/actionlint@v0` 不存在 | action 版本号 `v0` 不是有效 tag | 改为 `reviewdog/action-actionlint@v1` |
| 25 | Fork PR 中 `git diff origin/main...HEAD` 报 `fatal: bad revision` | merge commit checkout 没有 `origin/main` 分支引用 | 用 `dorny/paths-filter` 的 `list-files: csv` + GitHub API 获取文件变更列表 |
| 26 | OBS 上传只在 Nightly 触发 | condition 写死 `trigger_type == 'nightly'` | 改为 `if: ${{ always() }}`（PR + Nightly 都上传） |
| 27 | PR 未同步最新 fork commit | fork PR sync 延迟 | push 空 commit 或修改任意 workflow YAML 强制触发 |

### 八、模型与权重

| # | 问题 | 原因 | 解决 |
|---|------|------|------|
| 28 | DeepSeek-V4-Flash W8A8 OOM | 284B 参数 > 8×32G=256GB HBM | 切换到 Qwen3-30B-A3B（30B 总参 / 3B 激活，W8A8≈30GB） |
| 29 | `ModelSlim Quantization Config Not Found` | w8a8 权重非 ModelSlim 格式，缺 `quant_model_description.json` | 保留 `--quantization ascend` 参数；确保权重路径正确指向 w8a8 目录 |
| 30 | 权重路径与 recipe 不匹配 | recipe 用 `vllm-ascend/` 目录，实际在 `Eco-Tech/` | Python 脚本中 `str.replace` 替换权重路径 |
| 31 | `--speculative-config` eagle3 报 model path 无效 | `your_eagle3_model_path` 未被替换 | 整行 `--speculative-config` 通过正则删除 |

## 核心文件

```
.github/workflows/
├── _recipe_verify.yml       # 可复用 recipe 验证 workflow
├── pr-recipe-verify.yml     # PR 触发（路径检测 → validate → deploy verify）
├── nightly-recipe-verify.yml # Nightly 看护（定时/推送/手动）
├── lint.yml                 # 路径过滤 lint（已有，增强）
├── deploy.yml               # GitHub Pages 部署（已有）
├── preview-build.yml        # PR Preview 构建（已有）
└── preview-deploy.yml       # PR Preview Netlify 部署（已有）

scripts/
└── verify-recipe.sh         # 核心脚本：解析 recipe → 安装依赖 → vllm serve → curl 验证 → aisbench → 结果输出
```

## 关键设计决策

1. **镜像选择**：优先用预装 vllm-ascend 的镜像（跳过 pip install），版本必须 ≥ v0.23.0（支持 `--tokenizer-mode deepseek_v4`）
2. **命令传递**：Python 写临时文件 → shell `cat` 拼接 → `bash script.sh &` 执行，避免 eval 特殊字符问题
3. **子 shell**：永远不用 `echo | while read`，改用文件重定向 `done < file`
4. **进程清理**：`kill -TERM -- -$pid` 杀进程组，`kill -KILL` 兜底
5. **路径替换**：recipe 占位符在 Python 端用 `str.replace` + `re.sub` 处理，不修改源文件
6. **场景过滤**：按硬件（A2/A3）和部署模式（单节点/多节点）自动跳过不适用的场景

## 阶段目标

- [x] 2026-07-30 前：自动资料转化 Recipe + 基础 lint + Astro Build + PR Preview + 代表性部署验证（Qwen3-30B-A3B 跑通）
- [ ] 2026-08-30 前：模型 F0/Maintainer/CI 工程责任链 + 中英文检视 + 主仓冒烟用例 + 失败归因 + 修复闭环
- [ ] 2026-09-30 前：A5 Runner + A5 基础模型验证 + A2/A3 稳定回归基线
