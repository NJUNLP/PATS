# This code requires the reward model to be Qwen2.5-Math-PRM-7B and the policy model to be Qwen2.5-7B-Instruct.
import os
import argparse
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams
import torch.nn.functional as F
import regex
import random
import numpy as np
import time
def set_seed(seed):
    """设置所有随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 经过实验发现，模型有时候会出现乱码，为此当步骤出现乱码我们会进行过滤，但是Qwen2.5-7B-Instruct几乎不会生成乱码
allowed_pattern = regex.compile(
    r"[^\p{Latin}\p{Greek}\d\s"                     # 拉丁字母、希腊字母、数字、空白
    r"\+\-\*/=\^\_\'\"`~"
    r"‘’“”\u00B9\u00B2\u00B3"                      # 智能引号及上标 ¹ (U+00B9), ² (U+00B2), ³ (U+00B3)
    r"\u2070-\u209F"                               # 整个上标和下标区块（包括 ⁰、⁴、⁵、⁶、⁷、⁸、⁹ 以及其他上标/下标字符）
    r"⌊⌋⌈⌉⎡⎤"                                   # 地板/天花板符号
    r"–—"                                         # en dash 和 em dash
    r"…⋯"                                         # 省略号 (U+2026) 和 数学省略号 (U+22EF)
    r"⋅·"                                         # 乘法点（既允许 Unicode 的 ⋅ 也允许中点 ·）
    r"→←↑↓↔"                                     # 箭头符号
    r"∥"                                         # 平行符号
    r"ℝℤℕℚℂ"                                    # 常用数集符号
    r"\.,:;?!\(\)\{\}\[\]\\\$%<>|&@#"              # 标点及其他符号
    r"√∛∜∝∞±×÷°≈≠≡≅"                             # 根号、比例、无穷、正负、比较符号
    r"≤≥−"                                       # 小于等于、大于等于、Unicode 减号
    r"∂∇∫∮∑∏"                                    # 偏导、梯度、积分、求和、求积符号
    r"∈∉∋∅∪∩⊂⊆⊄⊇⊃"                              # 集合关系符号（已新增交集符号 ∩）
    r"∧∨¬⇒⇔∀∃∴∵"                                # 逻辑与量词符号
    r"⊕⊗⊥∠"                                     # 直和、直积、正交、角度符号
    r"′″"                                        # 单、双撇
    r"≪≫"                                        # 较小/较大符号
    r"\U0001D400-\U0001D7FF"                       # 数学花体及其他数学风格字母
    r"\uFF21-\uFF3A\uFF41-\uFF5A"                  # 全角拉丁字母（大写和小写）
    r"\u2100-\u214F"                              # Letterlike Symbols 区块
    r"△□♦♣◆"                                    # 几何图形符号
)

# 阶段配置（调整为三种设定）
PHASE_CONFIGS = [
    {"sample_num": 8, "temperature": 0.6},  # 阶段0：深度慢思考
    {"sample_num": 4, "temperature": 0.6},  # 阶段1：聚焦思考
    {"sample_num": 2, "temperature": 0.6},  # 阶段2：收敛和快速执行
]

def find_illegal_chars(text: str) -> list:
    """查找非法字符（保持原有实现）"""
    return allowed_pattern.findall(text)

def is_math_answer_valid(answer: str) -> bool:
    """答案有效性检查（保持原有实现）"""
    return not bool(find_illegal_chars(answer))

def get_next_step(args, policy_model, question, previous_steps, phase):
    with open(args.policy_prompt_path, "r") as f:
        prompt = f.read()
    previous_step = "" if len(previous_steps) == 0 else "\n\n".join(previous_steps)
    now_prompt = prompt.format(question, previous_step)
    now_prompt = now_prompt.replace("left-brackets", "{").replace("right-brackets", "}")
    now_prompt += "\n\n"

    if args.all_fast:
        config = PHASE_CONFIGS[2]  # 快速执行，使用阶段2的配置
    elif args.all_slow:
        config = PHASE_CONFIGS[0]  # 慢思考，使用阶段0的配置
    else:
        config = PHASE_CONFIGS[phase]

    sampling_params = SamplingParams(
        n=config["sample_num"],
        temperature=config["temperature"],
        max_tokens=1024,
        stop=["\n\n"],
        include_stop_str_in_output=True
    )
    llm_outputs = policy_model.generate(now_prompt, sampling_params)

    new_steps_candidates = []
    str_set = set()
    for output_obj in llm_outputs[0].outputs:
        gen_text = output_obj.text
        if not is_math_answer_valid(gen_text):
            print(f"-------------------\nInvalid step: {gen_text}\n-------------------")
            continue
        if gen_text.endswith("\n\n"):
            gen_text = gen_text[:-2].strip()
        else:
            gen_text += "<|eot_id|>"
        gen_text = gen_text

        if gen_text in str_set:
            continue
        new_steps_candidates.append(gen_text)
        str_set.add(gen_text)
    return new_steps_candidates

def make_prompt(tokenizer, question: str, steps: list) -> str:
    """构造奖励模型输入（保持原有实现）"""
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

def get_reward(model, tokenizer, question: str, steps: list) -> float:
    """计算步骤奖励（优化错误处理）"""
    try:
        conversation_str = make_prompt(tokenizer, question, steps)
        input_ids = tokenizer.encode(conversation_str, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        logits = outputs.logits[0]
        step_sep_id = tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)[0]
        
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        
        rewards = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            if len(positive_probs) > 0:
                rewards.extend(positive_probs.cpu().tolist())
        
        return rewards[-1] if rewards else 0.0
    except Exception as e:
        print(f"计算奖励时出错：{str(e)}")
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="渐进式推理流程")
    parser.add_argument("--policy_model_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--policy_prompt_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_begin", type=int, default=0)
    parser.add_argument("--data_end", type=int, default=None)
    parser.add_argument("--fast_threshold", type=float, default=0.85)
    parser.add_argument("--slow_threshold", type=float, default=0.75)
    parser.add_argument("--all_fast", action="store_true")
    parser.add_argument("--all_slow", action="store_true")
    parser.add_argument("--initial_mode", type=int, default=0)
    parser.add_argument("--penalty_times", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于确保结果可重现")
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    end = len(dataset) if args.data_end is None else args.data_end
    dataset = [dataset[i] for i in range(args.data_begin, end)]
    

    # 初始化模型（优化设备分配）
    policy_model = LLM(
        model=args.policy_model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False
    )
    
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)

    results = []
    for data in tqdm(dataset, desc="处理问题"):
        question = data["problem"]
        previous_steps = []
        used_steps = set()
        iteration_history = []
        current_phase = args.initial_mode  # 初始阶段
        max_iteration = 35
        consecutive_low = 0
        
        # 强制模式覆盖
        if args.all_fast:
            current_phase = 2  # 快速执行，使用阶段2的配置
        elif args.all_slow:
            current_phase = 0  # 慢思考，使用阶段0的配置

        for iter_count in range(1, max_iteration + 1):
            # 生成候选步骤
            candidates = get_next_step(args, policy_model, question, previous_steps, current_phase)
            if not candidates:
                iteration_history.append({
                    "iteration": iter_count,
                    "status": "failed",
                    "phase": current_phase
                })
                continue

            # 评估候选步骤
            valid_candidates = []
            for candidate in candidates:
                if candidate in used_steps:
                    continue
                
                try:
                    reward = get_reward(reward_model, tokenizer, question, previous_steps + [candidate])
                except Exception as e:
                    print(f"评估步骤时出错：{str(e)}")
                    continue
                
                valid_candidates.append({
                    "step": candidate,
                    "reward": reward,
                    "valid": reward >= args.slow_threshold
                })

            # 选择最佳候选
            best_candidate = None
            if valid_candidates:
                valid_rewards = [c for c in valid_candidates if c["valid"]]
                if valid_rewards:
                    best_candidate = max(valid_rewards, key=lambda x: x["reward"])
                else:
                    best_candidate = max(valid_candidates, key=lambda x: x["reward"])

            # 记录迭代信息
            iter_data = {
                "iteration": iter_count,
                "phase": current_phase,
                "candidates": valid_candidates,
                "selected": None,
                "new_phase": current_phase,
                "terminated": False
            }

            # 处理无有效候选情况
            if not best_candidate:
                iteration_history.append(iter_data)
                continue

            # 应用惩罚机制
            if best_candidate["reward"] < 0.4:
                if consecutive_low < args.penalty_times:
                    # 第一次低奖励：惩罚并重新生成
                    consecutive_low += 1
                    current_phase = 0
                    iter_data["selected"] = {
                        "step": best_candidate["step"],
                        "reward": best_candidate["reward"],
                        "accepted": False
                    }
                    iteration_history.append(iter_data)
                    continue  # 跳过后续处理
                else:
                    # 第二次低奖励：强制接受并重置计数器
                    consecutive_low = 0
            else:
                # 正常情况：重置计数器
                consecutive_low = 0

            # 接受候选并更新阶段
            previous_steps.append(best_candidate["step"])
            used_steps.add(best_candidate["step"])
            
            # 渐进阶段调整逻辑
            new_phase = current_phase
            if best_candidate["reward"] >= args.fast_threshold:
                new_phase = min(current_phase + 1, 2)  # 最大阶段为2
            elif best_candidate["reward"] < args.slow_threshold:
                new_phase = 0

            # 强制模式锁定
            if args.all_fast:
                new_phase = 2
            elif args.all_slow:
                new_phase = 0

            # 更新迭代数据
            iter_data.update({
                "selected": {
                    "step": best_candidate["step"],
                    "reward": best_candidate["reward"],
                    "accepted": True
                },
                "new_phase": new_phase,
                "terminated": "<|eot_id|>" in best_candidate["step"]
            })
            
            current_phase = new_phase
            iteration_history.append(iter_data)

            # 终止条件检查
            if "<|eot_id|>" in best_candidate["step"]:
                break

        # 保存结果
        results.append({
            "question": question,
            "history": iteration_history,
            "final_steps": previous_steps,
            "success": any("<|eot_id|>" in step for step in previous_steps)
        })
        print(f"问题id {data['id']}处理完成")

    # 保存输出文件
    output_file = os.path.join(args.output_dir, f"result-{args.data_begin}-{end}-progressive.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"处理完成！结果保存至：{output_file}")