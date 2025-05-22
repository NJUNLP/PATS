# 获取脚本所在的目录
SCRIPT_DIR=$(dirname "$0")

# 进入脚本所在的目录
cd "$SCRIPT_DIR"

# 现在你可以在这个目录下执行其他操作
echo "当前目录: $(pwd)"


# our pipeline
python ../pipeline.py \
--policy_model_path [policy_model_path] \
--reward_model_path [reward_model_path] \
--policy_prompt_path [../prompts/math-zero-shot-qwen-continue.txt] \
--data_path [data_path] \
--output_dir [output_dir] \
--data_begin [data_begin] \
--data_end [data_end] \
--fast_threshold [fast_threshold] \
--slow_threshold [slow_threshold] \
--seed [seed] \
--initial_mode 0 \ # 0: Complex 1: Medium 2: Simple
--penalty_times 1 # 0: No Penalty 1: One time penalty 100: Infinite times penalty
