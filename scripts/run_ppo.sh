#!/bin/bash
# run_ppo.sh — PPO 训练入口（GSM8K / Qwen2.5-0.5B-Instruct / 单卡 4090）
#
# 用法:
#   bash scripts/run_ppo.sh 1        # 首次：训练到 1 个 epoch（= 29 步）
#   bash scripts/run_ppo.sh 5        # 续训到"总计 5 个 epoch"（= 145 步）
#
#   DETACH=1 bash scripts/run_ppo.sh 5   # 后台跑，日志写 $LOG
#
# 三个必须知道的约定（都是踩过的坑）:
#   1. trainer.total_epochs 是"从零算起的绝对总数"，不是"再加几个 epoch"。
#      续训时填 5 = 跑到第 145 步。
#   2. 续训必须沿用同一个 EXPERIMENT：verl 靠
#      checkpoints/<PROJECT>/<EXPERIMENT>/latest_checkpointed_iteration.txt 定位续训点，
#      换名字就接不上（会从头开始）。
#   3. SAVE_FREQ 默认远大于总步数 => 只在收尾存一次存档。
#      单个 checkpoint 约 12GB，而系统盘只有 30GB；开启定期存档时"新旧两份并存"
#      （24GB）会把磁盘写满并让训练崩在 torch.save 上。
set -eu

EPOCHS=${1:-1}
VERL_REPO=${VERL_REPO:-/root/verl}
DATA_DIR=${DATA_DIR:-/root/data/gsm8k}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
PROJECT=${PROJECT:-verl_gsm8k_ppo}
EXPERIMENT=${EXPERIMENT:-ppo_gsm8k}
PY=${PY:-/opt/conda/envs/verl/bin/python3}
SAVE_FREQ=${SAVE_FREQ:-1000000}
TEST_FREQ=${TEST_FREQ:-25}
LOG=${LOG:-/root/verl_ppo.log}

cd "$VERL_REPO"
export PYTHONPATH="$VERL_REPO"
export PYTHONUNBUFFERED=1
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}   # 想只写本地就设 SWANLAB_MODE=local

ARGS=(
  data.train_files="$DATA_DIR/train.parquet"
  data.val_files="$DATA_DIR/test.parquet"
  data.train_batch_size=256
  data.max_prompt_length=512
  data.max_response_length=512
  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  critic.optim.lr=1e-5
  critic.model.path="$MODEL"
  critic.ppo_micro_batch_size_per_gpu=4
  algorithm.kl_ctrl.kl_coef=0.001
  'trainer.logger=[console,swanlab]'
  trainer.project_name="$PROJECT"
  trainer.experiment_name="$EXPERIMENT"
  trainer.val_before_train=False
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.save_freq="$SAVE_FREQ"
  trainer.max_actor_ckpt_to_keep=1
  trainer.max_critic_ckpt_to_keep=1
  trainer.test_freq="$TEST_FREQ"
  trainer.total_epochs="$EPOCHS"
)

echo "[run_ppo] epochs=$EPOCHS (约 $((EPOCHS * 29)) 步) | project=$PROJECT | experiment=$EXPERIMENT"
echo "[run_ppo] save_freq=$SAVE_FREQ test_freq=$TEST_FREQ log=$LOG"

if [ "${DETACH:-0}" = "1" ]; then
  setsid nohup "$PY" -m verl.trainer.main_ppo "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
  echo "[run_ppo] detached pid=$!  (停止: kill -9 -\$(cat $LOG.pid) 或按 ps 找主进程)"
else
  "$PY" -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee "$LOG"
fi
