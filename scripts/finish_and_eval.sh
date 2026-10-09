#!/bin/bash
# finish_and_eval.sh — 收尾流水线：合并 FSDP checkpoint -> HF 模型 -> 评测 -> 多 run 对比
#
# 用法:
#   bash scripts/finish_and_eval.sh 145          # 合并/评测 global_step_145，并与已有 run 对比
#   STEP=145 MERGED_DIR=/root/merged bash scripts/finish_and_eval.sh 145
#
# 依赖: eval_gsm8k.py / compare_evals.py（同仓库根目录）
set -eu

STEP=${1:?用法: bash scripts/finish_and_eval.sh <GLOBAL_STEP>}
VERL_REPO=${VERL_REPO:-/root/verl}
CKPT_DIR=${CKPT_DIR:-$VERL_REPO/checkpoints/${PROJECT:-verl_gsm8k_ppo}/${EXPERIMENT:-ppo_gsm8k}}
MERGED_DIR=${MERGED_DIR:-/root/merged}
EVAL_DIR=${EVAL_DIR:-/root/eval_out}
TOKENIZER=${TOKENIZER:-Qwen/Qwen2.5-0.5B-Instruct}
DATA=${DATA:-/root/data/gsm8k/test.parquet}
PY=${PY:-/opt/conda/envs/verl/bin/python3}
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

CKPT="$CKPT_DIR/global_step_${STEP}/actor"
OUT="$MERGED_DIR/global_step_${STEP}"
mkdir -p "$MERGED_DIR" "$EVAL_DIR"

if [ ! -d "$CKPT" ]; then
  echo "找不到 checkpoint: $CKPT"
  ls -l "$CKPT_DIR" 2>&1 || true
  exit 1
fi

echo "=== 1/3 合并 actor checkpoint -> HF 模型 ==="
rm -rf "$OUT"
PYTHONPATH="$VERL_REPO" "$PY" -m verl.model_merger merge \
  --backend fsdp --local_dir "$CKPT" --target_dir "$OUT" 2>&1 | tail -6
du -sh "$OUT"

echo "=== 2/3 评测（严格 + 宽松双口径）==="
# 注意：离线 vLLM 必须关掉 FlashInfer 采样内核，否则 EngineCore 起不来
VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH="$VERL_REPO" "$PY" "$REPO_ROOT/eval_gsm8k.py" \
  --model "$OUT" --tokenizer "$TOKENIZER" --data "$DATA" \
  --tag "step${STEP}" --out "$EVAL_DIR/step${STEP}.jsonl" 2>&1 | tail -8

echo "=== 3/3 对比所有已评测的 run ==="
shopt -s nullglob
RUNS=()
for f in "$EVAL_DIR"/*.jsonl; do
  tag=$(basename "$f" .jsonl)
  RUNS+=(--run "${tag}=${f}")
done
shopt -u nullglob
if [ ${#RUNS[@]} -eq 0 ]; then echo "没有找到任何评测结果"; exit 1; fi
printf '  参与对比: %s\n' "$(printf '%s ' "${RUNS[@]}" | sed 's/--run //g')"
"$PY" "$REPO_ROOT/compare_evals.py" "${RUNS[@]}" --out-md "$EVAL_DIR/compare_all.md"

df -h / | tail -1
echo "=== finish_and_eval.sh 完成 ==="
