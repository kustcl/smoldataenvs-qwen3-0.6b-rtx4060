# 138 道任务这次 GRPO 的退化分析

范围：2026-11-15 的 runs/grpo-agent-plus150，138 道训练题、138 步、每题 4 次采样，初始模型为合并后的 SFT Qwen3-0.6B。只分析现有记录，未重新训练或推理。

## 实际退化

同一组 90 道 eval easy/medium，严格评分从 SFT 的 19/90（21.11%）降至 RL 的 14/90（15.56%）。逐题比较丢失 6 题、新增 1 题，McNemar 双侧精确 p=0.125；可以确认这次评测分数下降，但不能据此断言模型全部通用能力显著下降。

未提交答案：33 → 42；达到 12 轮上限：29 → 37；重复命令总数：275 → 329；已提交任务内正确率：33.33% → 29.17%。退化同时出现在错误恢复/提交和问题理解/计算逻辑。

## 最有直接证据的训练机制问题

1. 提交奖励经过组内标准差归一化后，不能理解成始终只有 0.05 的影响。
   实际优化奖励为 strict + 0.05 * submitted，scale_rewards=group。138 步中有 17 步所有样本都答错，仅提交与否产生优势；日志中共 39 条错误轨迹得到正优势。第 9 步四条严格奖励全为 0，唯一提交的错误轨迹优势为 +1.494，其余为 -0.498。这会强化错误解法的整段 assistant 行为；终局奖励不能识别其中哪些计算步骤错了。这是明确的优化信号问题，但其对最终净降分的因果贡献仍需消融验证。

2. 单步仅一题四条轨迹，且没有参考模型约束。
   generation_batch_size=4、num_generations=4，实际每步只有一道独立问题。beta=0、无 SFT 保持损失，LoRA 更新 all-linear。局部题目上的偶然解法容易影响已有行为。num_iterations=1 且这次更新前没有参数变化时，old log-prob 使用当前值 detach，实际 138 步 clip_ratio 全为 0；不能把 PPO clipping 当作相对 SFT 的保护。

3. 138 题是筛选出的窄分布，并非泛化训练曲线。
   池中 easy=63、medium=75；评测 easy=16、medium=74。每题只见一次，训练前后区间奖励来自不同题目，因此末段训练奖励上涨不证明同一评测分布上的能力提升。138 步中 22 步梯度为 0；剩余有信号的步骤中 17 步仅由提交奖励驱动。

4. eval_every=25 没有接入训练评测。
   实际 eval_strategy=no，无 eval_dataset，trainer_state 的 best_model_checkpoint=null。当前测的是 final，不能判断 checkpoint-25/50/75 等是否曾优于 final 或 SFT。

## 轨迹证据

- 0001_133_1133625_qa_3：SFT 在 SyntaxError 后修正 BJP+ 筛选并提交 256219；RL 重复错误代码至轮数上限，没有答案。
- 0011_355_11355842_qa_1：RL 反复把二维 DataFrame 传入 np.polyfit，重复 expected 1D vector for x，未提交。
- 0001_116_1116342_qa_3：问题问 cap-surface，SFT 正确统计得到 y；RL 改查 cap-shape 得到 x。
- 0001_104_1104901_qa_3：问题问 neutral 情绪最多的航空公司，SFT 筛选后得到 Delta；RL 统计全部情绪，还从输出中选错结果。
- 0001_134_1134780_qa_3：问题问 party=others 的数量，RL 去 candidate 列匹配 others，提交 0，正确答案 1883。

丢失的六题中，三题未提交、两题标签错误、一题数字错误。不能把全部下降归结为格式问题。

## 次要风险与排查结果

训练 max_completion_length=3072 包括工具输出，工具循环上限 6；评测每轮最多生成 1024，最多 12 次模型调用，prompt 左截断至 8192。预算和执行循环存在分布差异。但 SFT 的 19 条正确评测轨迹都不超过 3072 completion tokens，且工具调用均不超过 6，因此不能把预算不足列为已证实首因。

训练 clipped_ratio 平均仅 1.81%，但该指标按最后 token 是否 EOS 判断，无法识别所有工具循环/输出预算提前结束。评测中 RL 有 21 次调用的完整 prompt 超过 8192，SFT 为 5 次；这与错误循环累积上下文相符，也会进一步损伤后续恢复。

LoRA dropout=0.05 且 disable_dropout=false，而评测 net.eval()；这是应消除的随机性/策略分布差异，贡献尚未消融验证。训练 repetition_penalty=1.0、评测为 1.05，三次评测设置本身一致，因此这一差异不是 RL/SFT 比较单独引入的偏差。

已核对 RL adapter 的 base_model_name_or_path 与评测加载的 SFT merged 一致。现有证据不支持“加载错底座”“训练用了 lenient grader”或“普遍缺少输入文件”作为首因。

## 建议顺序

先评估现有 checkpoint-25/50/75/100/125/138，保持现有 90 题设置一致，用逐题差异定位退化出现时间；这一步不用重训。

下一次从同一 SFT 出发，先做 reward_submitted_weight=0 的单变量消融；将纯提交信号与正确性训练分开。再比较显式 KL 约束（例如 beta=0.01/0.02，小规模校准；参考模型必须是 SFT）、LoRA dropout=0、每步多道独立题目，并真正接入 heldout 评测。train_grpo_agent.py 当前没有传递 beta/disable_dropout 等配置，仅加 --set 不会使其生效，需先接入 GRPOConfig。

不建议仅把学习率继续降低或继续增加步数：本次实际学习率已为 3e-6，且只训练一遍；奖励信号与约束缺失应优先验证。

复核脚本：reports/audit_rl_degradation.py；统计与优势值证据：reports/rl_degradation_evidence.json。
