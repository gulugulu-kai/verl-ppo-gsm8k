#!/bin/bash
# setup_env.sh — 环境修复 / 依赖安装 / 兼容补丁（幂等，可重复执行）
#
# 适用场景：bitahub 实例，镜像 pytorch:2.11.0-cuda12.8-cudnn9-py312-ubuntu24.04-systemd，
#          conda env "verl"，以及一个带 transfer_queue 集成的 verl fork。
#
# 用法：ENV_NAME=verl VERL_REPO=/root/verl bash scripts/setup_env.sh
set -u

ENV_NAME=${ENV_NAME:-verl}
CONDA_ROOT=${CONDA_ROOT:-/opt/conda}
VERL_REPO=${VERL_REPO:-/root/verl}
PY=${CONDA_ROOT}/envs/${ENV_NAME}/bin/python3
SP=${CONDA_ROOT}/envs/${ENV_NAME}/lib/python3.12/site-packages

echo "== 0) 基础检查 =="
if [ ! -x "$PY" ]; then echo "找不到解释器: $PY"; exit 1; fi
"$PY" -V
[ -d "$VERL_REPO/verl" ] || { echo "找不到 verl 仓库: $VERL_REPO"; exit 1; }

echo "== 1) transfer_queue 缺 version 文件（手工拷贝进 site-packages 时漏掉） =="
if [ ! -f "$SP/transfer_queue/version/version" ]; then
  mkdir -p "$SP/transfer_queue/version"
  echo 0.1.0 > "$SP/transfer_queue/version/version"
  echo "   已创建 transfer_queue/version/version"
else
  echo "   已存在，跳过"
fi

echo "== 2) verl/version/version（git 跟踪文件，容易漏拷） =="
if [ ! -f "$VERL_REPO/verl/version/version" ]; then
  ( cd "$VERL_REPO" && git checkout -- verl/version ) && echo "   已从 git 还原"
else
  echo "   已存在，跳过"
fi

echo "== 3) pandas（源码树拷贝会缺 pandas/util/version.py） =="
if ! "$PY" -c 'import pandas' >/dev/null 2>&1; then
  echo "   重装 pandas"
  "$PY" -m pip install --force-reinstall --no-deps pandas
else
  echo "   pandas 正常，跳过"
fi

echo "== 4) 自动补齐缺失依赖（该 fork 未声明的，如 cachetools / packaging） =="
cd "$VERL_REPO"
PYTHONPATH="$VERL_REPO" "$PY" - <<'PYEOF'
import importlib, subprocess, sys

mods = [
    'verl', 'verl.trainer.main_ppo', 'verl.trainer.ppo.v1',
    'verl.experimental.agent_loop', 'verl.workers.rollout.llm_server',
    'verl.workers.rollout.router', 'verl.workers.rollout.hf_rollout',
    'verl.trainer.ppo.ray_trainer', 'verl.utils.dataset.rl_dataset',
]
alias = {'yaml': 'pyyaml', 'sklearn': 'scikit-learn', 'cv2': 'opencv-python',
         'PIL': 'pillow', 'google': 'protobuf'}

for attempt in range(6):
    fails = []
    for m in mods:
        try:
            importlib.import_module(m)
        except ModuleNotFoundError as e:
            name = e.name or ''
            if name.startswith('verl'):
                print('   结构性缺失（非依赖问题），忽略:', m, name)
            else:
                fails.append(name)
        except Exception as e:
            print('   其他失败:', m, type(e).__name__, str(e)[:160])
    if not fails:
        print('   preflight clean (attempt %d)' % attempt)
        break
    pkgs = sorted({alias.get(n.split('.')[0], n.split('.')[0]) for n in fails})
    print('   安装:', pkgs)
    subprocess.run([sys.executable, '-m', 'pip', 'install'] + pkgs)
PYEOF

echo "== 5) vLLM（该 verl 版本只注册 vllm/sglang/trtllm 的 async rollout） =="
if ! "$PY" -c 'import vllm' >/dev/null 2>&1; then
  echo "   安装 vllm（注意：会把 torch 2.14.x 降到 2.13.x，这是 vLLM 的硬性依赖）"
  "$PY" -m pip install vllm
else
  echo "   vllm 已安装，跳过"
fi
"$PY" -c 'import torch, vllm; print("   torch", torch.__version__, "| vllm", vllm.__version__)'

echo "== 6) 3 处兼容补丁（幂等：已打过则跳过） =="
VERL_REPO="$VERL_REPO" "$PY" - <<'PYEOF'
import os

R = os.environ['VERL_REPO']
Q = chr(34)

# 6a) load_valuehead_model 里写死了 flash_attention_2，但环境无 flash-attn -> 改 sdpa
p = os.path.join(R, 'verl/utils/model.py')
if os.path.exists(p):
    src = open(p).read()
    old = 'attn_implementation=' + Q + 'flash_attention_2' + Q
    new = 'attn_implementation=' + Q + 'sdpa' + Q
    n = src.count(old)
    if n:
        open(p, 'w').write(src.replace(old, new))
        print('   [6a] patched %s (%d 处)' % (p, n))
    else:
        print('   [6a] 无需修改')

# 6b) 模型 config 的默认 attn 实现
p = os.path.join(R, 'verl/workers/config/model.py')
if os.path.exists(p):
    nl = chr(10)
    lines = open(p).read().split(nl)
    hit = 0
    for i, ln in enumerate(lines):
        if 'attn_implementation = self.override_config.get' in ln and 'flash_attention_2' in ln:
            lines[i] = ln.replace('flash_attention_2', 'sdpa')
            hit += 1
    if hit:
        open(p, 'w').write(nl.join(lines))
        print('   [6b] patched %s (%d 行)' % (p, hit))
    else:
        print('   [6b] 无需修改')

# 6c) trl>=1 已删除 AutoModelForCausalLMWithValueHead，而 is_trl_available() 仍返回 True
p = os.path.join(R, 'verl/models/transformers/monkey_patch.py')
if os.path.exists(p):
    src = open(p).read()
    old = (
        '    if is_trl_available():\n'
        '        try:\n'
        '            from trl.experimental.ppo import AutoModelForCausalLMWithValueHead  # type: ignore\n'
        '        except ImportError:\n'
        '            from trl import AutoModelForCausalLMWithValueHead  # type: ignore'
    )
    new = (
        '    try:\n'
        '        from trl.experimental.ppo import AutoModelForCausalLMWithValueHead  # type: ignore\n'
        '    except ImportError:\n'
        '        try:\n'
        '            from trl import AutoModelForCausalLMWithValueHead  # type: ignore\n'
        '        except ImportError:\n'
        '            AutoModelForCausalLMWithValueHead = None\n'
        '\n'
        '    if AutoModelForCausalLMWithValueHead is not None:'
    )
    if old in src:
        open(p, 'w').write(src.replace(old, new))
        print('   [6c] patched %s' % p)
    else:
        print('   [6c] 无需修改（或源码结构已变，请手工核对）')
PYEOF

echo "== 7) 校验 =="
cd "$VERL_REPO" && PYTHONPATH="$VERL_REPO" "$PY" -c 'import verl; print("   verl", verl.__version__, "OK")'
df -h / | tail -1
echo "== setup_env.sh 完成 =="
