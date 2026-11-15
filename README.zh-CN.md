# 8 GB 笔记本上的 SmolDataEnvs：0.6B 数据分析智能体的 SFT 蒸馏 vs. GRPO

本项目在**一台消费级笔记本**（Windows 11 + RTX 4060 Laptop 8 GB 显存）上，用 **Qwen3-0.6B**
复现了 [SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) 的智能体训练流程，
包含：本地沙箱、评测框架、SFT 与 GRPO 训练脚本，以及下面所有实验的原始结果。

**核心结论：在这套硬件上，SFT（对录制的智能体轨迹做蒸馏）是唯一显著的收益来源，而 GRPO
反而把模型训差了。** 仓库主要作为一个"小模型 RL 的可复现负结果"公开，同时提供产生这个结论的
工具链。

**这是一个人的业余项目，非常希望得到社区帮助。** 如果你有想法或闲置算力，请看下面的
[恳请社区帮助](#恳请社区帮助)；最有价值的三件事是：工具调用并行化、筛选的批量调度、以及在
2B/4B 上用 `num_generations=8` 复现。

## 实验结果（同一批 90 道留出题、同一解码设置）

评测设置：`eval` split 的 easy + medium（16 + 74 题），agent 模式，贪婪解码，每题最多 12 次
模型调用、每次 1024 token，Docker 沙箱。

| 阶段 | v1（宽松判分） | v2（严格判分） | 交卷率 | 打满 12 轮 |
|---|---|---|---|---|
| Qwen3-0.6B，未训练 | 0.0% | 0.0% | 0% | — |
| **SFT（1 epoch，411 步）** | **25.6%**（23/90） | **21.1%**（19/90） | **63%** | 29 |
| 从 SFT 做 GRPO，17 步 | 20.0% | 16.7% | 53% | 35 |
| 从 SFT 做 GRPO，138 步 | 17.8% | 15.6% | 53% | 37 |
| GRPO checkpoint-50 | 同样低于 SFT | — | — | — |

分档（v1 / v2）：

| 模型 | easy（16） | medium（74） | 没见过 Kaggle 数据集（21） |
|---|---|---|---|
| SFT | 56.3% / 43.8% | 18.9% / 16.2% | 4.8% / 4.8% |
| GRPO-138 | 50.0% / 43.8% | **10.8% / 9.5%** | 9.5% / 9.5%（2/21，噪声） |

逐题配对（同一批题、完全相同的评测设置）：

- 总体 v1：双方都对 15 题、**仅 SFT 对 8 题**、**仅 GRPO 对 1 题** → McNemar **p = 0.039**
- medium v1：8 双对、**6 仅 SFT 对、0 仅 GRPO 对** → **p = 0.031**
- 总体 v2：13 双对、6 仅 SFT 对、1 仅 GRPO 对 → p = 0.125

训练奖励在 138 步里**上升**（严格奖励均值 0.40 → 0.56），留出集准确率却**下降**——典型的
"优化了采样器、没优化模型"。

**训练曲线（SwanLab）**：<https://swanlab.cn/@cl666/SmolData-GRPO/v1/jwt6a9/overview> —
包含 GRPO 各轮的奖励、交卷率、零方差组比例与每步耗时。原始 summary 和 138 步训练日志也放在
[`reports/evals/`](reports/evals/)。

## 仓库内容

| 文件 | 作用 |
|---|---|
| `smolenv_local.py` | 本地沙箱（Docker / bwrap / macOS sandbox-exec）、任务文件摆放、工具执行、双判分（v1 + v2） |
| `checker_v2.py` | 严格答案检查器（RL 的奖励） |
| `eval_harness.py` | Agent 模式评测；支持本地 HF 模型或任意 OpenAI 兼容端点；本地批量生成 |
| `train_sft_local.py` | 对录制的智能体轨迹做 LoRA SFT |
| `train_grpo_agent.py` | 基于 TRL `environment_factory` 的 GRPO（多轮 bash 工具、答案写入 `/workdir/answer.txt`） |
| `train_common.py` | profile 加载、任务池过滤、SwanLab 接入 |
| `train_plan.yaml` | 全部超参与实验计划 |
| `scripts/prescreen_until.py` | 持续筛选训练题，直到凑够 N 道"半对半错"（非零方差） |
| `scripts/stage_task_files.py` / `merge_lora.py` | 任务数据摆放 / adapter 合并 |
| `reports/rl_degradation_analysis_138.md` | 138 步 GRPO 的详细复盘（含逐步证据） |
| `grpo_pool_150/` | 138 道池子（`grpo_pool_150.json`）、筛选报告与原始 rollout |

## 基于哪个项目

本项目**不是从零实现**，而是 SmolDataEnvs 训练协议的 **Windows / 8 GB 移植版**，来源如下：

| 来源 | 提供什么 |
|---|---|
| [`adithya-s-k/FineEnvs`](https://github.com/adithya-s-k/FineEnvs) @ `08a5622` 的 `04-smoldataenvs/scripts/` | 参考实现（`train_grpo.py`、`train_sft.py`、`eval_pass1.py`、`rollout.py`）；`train_plan.yaml` 里所有默认值都来自这里，每一处偏离都列在该文件的 `deviations:` 里 |
| [`FineEnvs/SmolDataEnvs`](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) @ `b2bf3564` | 5,394 道可验证的数据分析任务，以及官方（宽松）判分器 |
| [`FineEnvs/SmolDataEnvs-sft`](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-sft) @ `a9fa95ab` | 4,677 条录制的多轮 bash 轨迹（SFT 用） |
| 本仓库的 `README.dgx-spark-4b.md` | 同一协议的 DGX Spark / Qwen3.5-4B 版本，本移植版由它改写而来 |
| `vendor/fineenvs_rollout.py`（Apache-2.0）、`vendor/smoldataenvs_grader.py`（MIT） | 原样引入的参考代码，见 [`NOTICE`](NOTICE) |

本移植版改了什么：用本地沙箱（`smolenv_local.py`）替代 Hugging Face 的托管沙箱；加入严格检查器
（`checker_v2.py`）作为 RL 奖励；评测改为本地 HF 批量生成；补齐 Windows 侧适配；模型换成
Qwen3-0.6B（原项目是 2B–4B）。

## 硬件与软件

| | |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop，8 GB GDDR6，24 SM（Ada，CC 8.9） |
| CPU / 内存 | i7-12650H / 16 GB |
| 系统 | Windows 11 |
| 沙箱 | Docker Desktop（WSL2），每次工具调用一个一次性容器 |
| 栈 | PyTorch 2.13 + CUDA 12.6、transformers 5.17、TRL 1.14、PEFT、datasets |

本机实测：Qwen3-0.6B 单序列约 **19 tok/s**，batch=4 聚合约 **85 tok/s**（2.65 GiB），
batch=8 约 **55 tok/s**（4.17 GiB）。

## SFT

**数据**：`FineEnvs/SmolDataEnvs-sft`，4,677 条多轮 bash 轨迹
（easy 1,402 / medium 2,640 / hard 635）。训练前做两道过滤：

1. 丢弃含**连续两个 assistant 回合**的样本（511 条，录制工具造成的脏数据）；
2. 超过 **2,048 token** 的轨迹**整条丢弃**而不是截断（截断会把写 `answer.txt` 的那一步切掉）。
   最终约 **3,300 条**：easy ≈1,300、medium ≈1,850、hard ≈260。

**配置**（`train_plan.yaml → sft.agent_format`）：

| | |
|---|---|
| LoRA | r=16，alpha=32，dropout 0.05，`all-linear` |
| 学习率 | 1e-4（cosine），1 epoch = **411 步** |
| Batch | per-device 2 × 梯度累积 4 = 8 |
| 最大长度 | 2,048（超长丢弃，绝不右截断） |
| 精度 | bf16 + gradient checkpointing |
| Loss | 只对 assistant token（TRL `assistant_only_loss`），使用 Qwen3 训练模板 |

未训练的 0.6B **完全不会写答案文件**（easy 0/16）；SFT 一个 epoch 后就学会了整套智能体循环，
在 90 题留出集上达到 25.6% / 21.1%——这是整个项目里唯一一次大幅提升。

## GRPO

**任务池**：GRPO 需要模型"有时能做对"的题；全对 / 全错的组优势为 0，不产生梯度。
`scripts/prescreen_until.py` 用 SFT 策略、T=0.8、每题采样 4 次，保留答对 1–3 次的题：

```
4,151 道合格训练题（easy + medium，输入完整、≤100 MB）
  → 实际筛 370 道 → 138 道 mixed（easy 63 / medium 75），全错 183、全对 49
  → 138 道池子（批量生成下约 2.6 小时，~3 条 rollout/分钟）
```

**训练配置**（`train_plan.yaml → profiles.agent_qwen3_06b_lora`）：

| | |
|---|---|
| 算法 | GRPO，`loss_type="dapo"`、`epsilon=0.2`、`scale_rewards="group"`、`beta=0`（无参考模型 / 无 KL） |
| 每次更新的 rollout | **1 个 prompt × 4 条采样**（`per_device=1`、`steps_per_generation=4`、`num_generations=4`） |
| 奖励 | 严格检查器（v2）权重 1 + `0.05 × submitted` |
| LoRA | r=16，alpha=32 |
| 学习率 | 3e-6 |
| 预算 | `max_completion_length=3072`（含工具输出），每轮 2048，最多 6 次工具调用 |
| 步数 | 138 步 = 池子一遍；3.8 小时；显存峰值 7.9/8.0 GiB |

**为什么退化（有据可查，不是猜测）：**

1. **每次更新只有 4 条 rollout**，低于 GRPO 稳定估计组优势所需（实践建议 ≥8）。8 GB 显存
   在保留 3k token 多轮轨迹的前提下放不下更多。
2. **没有 KL 锚定**（`beta=0`）：没有任何东西把策略拴在 SFT 附近，小批量梯度噪声占主导。
   退化在第 50 步时就已经完成。
3. **训练采样 vs. 评测贪婪**：训练奖励涨、留出集跌——策略学会了"继续试"（在 T=0.8 下偶尔
   能蒙对），丢掉了 SFT 的"停下来写答案"。交卷率 63% → 53%，打满 12 轮 29 → 37。
4. **死循环不是 RL 造成的**：SFT 的 90 条轨迹里就有 25 条会把**同一条报错命令**原样重复到
   11 次；GRPO 把它变成 31 条，但没有创造它。真正的修法在 SFT 数据（补"报错后改思路"的轨迹）。
5. **池子窄**：138 道题、每题一个 epoch 只见一次；训练奖励曲线每一步来自不同题子集，不能
   当作泛化曲线来读。

完整复盘：[`reports/rl_degradation_analysis_138.md`](reports/rl_degradation_analysis_138.md)。

## 评测协议

- `eval` split 的 `easy,medium` 共 90 题；同时用数据集自带宽松判分（v1）和严格检查器（v2）。
  v2 是 v1 的子集，两者差距来自格式/容差。
- Agent 模式：模型在 bash 容器里工作，≤12 次模型调用、每次 1024 token、贪婪解码、关闭 thinking。
- 本地 HF 生成是**批量化**的（`--hf-batch-size 4`、并发 8），比单条路径快约 3×。批处理会改变
  浮点归约顺序，因此**数字只在相同批处理设置下可比**——上表全部是 batch=4、并发 8 的结果。

## 复现

```bash
# 0) 摆放任务数据（需要 HF bucket 列表，见 scripts/stage_task_files.py）
python scripts/stage_task_files.py

# 1) SFT（对录制轨迹做 LoRA）
python train_sft_local.py --model models/Qwen3-0.6B --out runs/sft-0.6b-2048-e1 --max-length 2048

# 2) 评测 SFT（easy+medium）
python eval_harness.py run --hf-model runs/sft-0.6b-2048-e1/merged \
  --split eval --tiers easy,medium --mode agent --sandbox win-docker \
  --concurrency 8 --hf-batch-size 4 --out runs/eval-sft-easy-medium

# 3) 筛选 mixed 任务池（到 --target 或 --max-hours 为止）
python scripts/prescreen_until.py --target 150 --tiers easy,medium --n-samples 4 \
  --concurrency 16 --hf-batch-size 6 --max-hours 8 --out grpo_pool_150 \
  --pool-out grpo_pool_150/grpo_pool_150.json

# 4) 从 SFT 模型做 GRPO
python train_grpo_agent.py --profile agent_qwen3_06b_lora \
  --init-from runs/sft-0.6b-2048-e1/merged --swanlab --run-dir runs/grpo-agent-plus150 \
  --set task_pool_file=grpo_pool_150/grpo_pool_150.json --set max_steps=138 \
  --set save_steps=25 --set learning_rate=3e-6 --set reward_submitted_weight=0.05
```

无 GPU 冒烟：`python eval_harness.py selftest`（mock 模型 + 真沙箱）、
`python train_grpo_agent.py --profile agent_qwen3_06b_lora --dry-run`。

## 如果重来，会怎么改

- **这个规模优先蒸馏，而不是 RL。** 对 0.6B 来说，从更强模型的轨迹做 SFT 蒸馏是可靠杠杆；
  GRPO 需要模型本身有余量去探索、每个 prompt 更多采样、以及 KL 锚定。
- SFT 用 **4,096 上下文**（2,048 会丢掉 25% 的 medium、59% 的 hard 轨迹——恰好是模型最缺的
  "错误恢复 + 收尾"行为）。
- 数据里补**报错后换思路、N 轮内交卷**的轨迹；当前最主要的失败模式就是同一条命令重复到上限。
- 如果还要在这台机器上试 RL：`beta≈0.02`（KL）、`lr=1e-6`、≤50 步，并且同时评采样（T>0）
  和贪婪两种解码。

## 恳请社区帮助

这是一个在一台 8 GB 笔记本上完成的个人项目，**我们非常希望得到社区的帮助**，无论是工程还是
实验设计。最需要的几件事：

1. **工具调用的并行/异步执行。** TRL 1.14 的 `_tool_call_loop` 在跨 rollout 时是串行执行的
   （只有同一条 rollout 内部的多个调用会 gather）。异步或线程池改造能把每步时间压缩 10–25%。
   我们试过 `async def` 路线，但 TRL 是"按 rollout"聚合的，所以没有收益。
2. **筛选用的批量生成调度器。** `eval_harness.py` 现在能把 4–6 个本地 HF 请求合批，筛选快了
   约 3×；如果再做一个跨任务的合批调度（连续批处理），建池的 2.6 小时还能再降。
3. **8 GB 上带 KL 的 GRPO。** `beta > 0` 需要一份参考模型（约 1.2 GiB），在 3k token 多轮轨迹
   下放不下；更便宜的锚定方案（adapter 快照、周期性刷新、带验证集的早停）会很有价值。
4. **Linux / bwrap 下的对齐复现。** 在 Linux 上重跑一遍，告诉我们退化是否同样出现；我们猜会，
   但 Windows 的分页和 Docker 延迟可能有影响。
5. **更好的 SFT 数据。** 展示"工具报错后换思路、N 轮内交卷"的轨迹是模型最大的缺口。如果你有
   这类数据，我们愿意测试。
6. **2B / 4B 的复现。** 显存更大的话，用同样的协议配 `num_generations=8`、`beta>0` 跑一遍，
   就能把"小模型上限"和"小批量 RL"两个原因分开。

想帮忙的话，开 issue 或 PR 都可以；或者直接告诉我们你想跑哪个实验——上面的"复现"一节有完整
命令，我们很乐意一起调整。

## 踩过的坑（替你踩完了）

下面每一条都真实消耗过时间；要么已在代码里修掉，要么在上面的命令里写死了。

**环境**

- **Docker Desktop 没启动。** 没有引擎时 `detect_backend()` 返回 `none`，工具调用直接在宿主机
  Windows 上执行，第一行就报 `ModuleNotFoundError: No module named 'resource'`（Unix 模块），
  模型反复重试到轮数上限，**所有奖励都是 0**，训练日志却看不出任何错误。长跑前先确认
  `smolenv_local.detect_backend()`。
- **HF 直连超时**：`HF_ENDPOINT=https://hf-mirror.com` 加 `HF_HUB_DISABLE_XET=1` 才能下。
- **0 字节输入文件**：下载中断会留下空 CSV，判分时会被静默当成"答错"；现在沙箱把 0 字节文件
  视为缺失文件（任务变成 ungraded，而不是被打分）。
- **SwanLab 0.10.5**：`swanlab.package.has_api_key` 被移除、初始化前调 `get_run()` 会抛异常、
  项目名改走 `SWANLAB_PROJ_NAME`、中文 GBK 控制台遇 emoji 直接崩
  （`sys.stdout.reconfigure(encoding="utf-8")` 可解）。
- **`python -I` 会把当前目录从 `sys.path` 里去掉**，仓库内的 import 会失败，需要手动插入。

**数据与判分**

- **连续两条 assistant 消息**：4,677 条 SFT 轨迹里有 511 条是 `assistant, assistant` 相邻
  （录制工具的产物），会让模型学成"自问自答"，必须丢弃。
- **智能体轨迹绝不能右截断**：截断会切掉最后写 `answer.txt` 的那一步；只能按长度整条丢弃。
- **长度过滤要用 TRL 的训练模板**：Qwen3 原生模板会把历史 assistant 回合里的空
  `<think></think>` 去掉，导致少算 8–20 个 token；评测 prompt 也必须用同一模板，否则第 2 轮
  起就和训练不一致。
- **`exclude_modules="visual.*"` 在 PEFT 里匹配不到任何东西**（PEFT 是全匹配），要写
  `.*visual.*`。

**TRL 语义（最坑的几个）**

- **梯度累积不会增加 prompt 数。** `per_device=1`、`grad_accum=4`、`num_generations=4` 时：
  `steps_per_generation=4`、`generation_batch_size=4`（序列数）= **每个 optimizer step 只有
  1 个 prompt × 4 条采样**。证据：每步 `epoch` 涨 `1/池子大小`、`num_tokens` 涨约 4k。梯度累积
  只是把这一组拆成 4 个微步。
- **工具调用跨 rollout 是串行的。** `async def` 工具只在**同一条 rollout 内部**被 gather，
  把 `bash` 改成异步不会带来任何并行。
- **`reward_weights` 只覆盖你传的函数。** TRL 会把环境的 `get_reward` 以权重 1 追加，所以
  `[0, 0.05, 0]` = 严格奖励 + 0.05 × 交卷。
- **YAML 1.1 会把 `3e-6` 解析成字符串**，最终在 AdamW 里报
  `TypeError: '<=' not supported between instances of 'float' and 'str'`。写 `3.0e-6`，或者让
  `--set` 自动把数字字符串转成 float（本仓库已处理）。
- **`task_pool_file` 指向的文件不存在时，过去会静默退化成随机训练题**；现在会直接报错。

**评测**

- **GRPO adapter 的基座是 SFT 合并模型**，不是 `models/Qwen3-0.6B`。拿底座去加载这个 adapter，
  会得到一个"从不写答案、90 题 80 秒跑完"的模型——看起来像"base 模型"，不像错误。
- **批处理会改变贪婪解码的结果。** bf16 批量矩阵乘的舍入不同，而 0.6B 很多 next-token 分布
  接近 50/50 平票，轨迹就会翻转。数字**只在相同批处理设置下可比**；换设置必须重新跑基线
  （batch=1 与 batch=4 之间，easy 16 题里翻了 8 题）。
- **显存是硬墙。** batch=4、4k prompt 单独跑约 2.7 GiB，但在评测框架里（长对话、8 并发）
  实测峰值 **7.9/8.0 GiB**；筛选配置（batch 6、≤7 轮）峰值 6.5 GiB；batch=8 直接 OOM。
  训练只生成 4 条序列就占 7.95/8.0 GiB。
- **`--concurrency` 只并行沙箱、不并行生成**（除非开批处理）：旧的本地 HF 客户端所有请求都排在
  同一把锁上，单条约 19 tok/s。

## 局限

- 单一模型（Qwen3-0.6B）、单一随机种子、单一 90 题留出集（每次运行 95% CI ≈ ±9pp）；
  表中每个数字都是一次运行的结果。
- GRPO 的 138 道池子是用 SFT 策略在 T=0.8 下筛的，因此天然偏向第一轮；后续轮次会有偏。
- 数字来自 Windows + Docker(WSL2) 后端；Linux/bwrap 下可能略有差异（工具延迟、无 WDDM 分页）。

## 许可与致谢

本移植版代码遵循仓库 LICENSE。任务数据来自
[FineEnvs/SmolDataEnvs](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs) 与
[FineEnvs/SmolDataEnvs-sft](https://huggingface.co/datasets/FineEnvs/SmolDataEnvs-sft)；
沙箱设计遵循 Hugging Face FineEnvs 的环境协议。见 [`NOTICE`](NOTICE)。
