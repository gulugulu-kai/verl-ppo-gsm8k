#!/usr/bin/env python3
"""GSM8K evaluation using the SAME prompt template and scoring as the verl PPO training run.

Usage:
  python3 eval_gsm8k.py --model Qwen/Qwen2.5-0.5B-Instruct --out /root/eval_out/base.jsonl
  python3 eval_gsm8k.py --model /root/merged/global_step_N --tokenizer Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import json
import os
import time

import pandas as pd
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--data', default='/root/data/gsm8k/test.parquet')
    ap.add_argument('--out', default=None)
    ap.add_argument('--limit', type=int, default=0, help='0 means all rows')
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    ap.add_argument('--max_model_len', type=int, default=1024)
    ap.add_argument('--tensor_parallel_size', type=int, default=1)
    ap.add_argument('--tag', default='eval')
    args = ap.parse_args()

    tok_path = args.tokenizer or args.model
    out_path = args.out or ('/root/eval_out/%s.jsonl' % args.tag)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.read_parquet(args.data)
    if args.limit > 0:
        df = df.iloc[:args.limit].reset_index(drop=True)
    print('[eval] tag=%s rows=%d data=%s' % (args.tag, len(df), args.data), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    prompts = []
    for msgs in df['prompt']:
        prompts.append(
            tokenizer.apply_chat_template(list(msgs), add_generation_prompt=True, tokenize=False)
        )
    ground_truths = [rm['ground_truth'] for rm in df['reward_model']]

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tokenizer=tok_path,
        dtype='bfloat16',
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, n=1)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    gen_s = time.time() - t0

    from verl.utils.reward_score.gsm8k import compute_score

    scores = []
    n_with_hash = 0
    n_truncated = 0
    examples = []
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, out in enumerate(outputs):
            text = out.outputs[0].text
            finish = out.outputs[0].finish_reason
            gt = ground_truths[i]
            sc = float(compute_score(text, gt))
            scores.append(sc)
            if '####' in text:
                n_with_hash += 1
            if finish == 'length':
                n_truncated += 1
            rec = {
                'index': i,
                'output': text,
                'ground_truth': gt,
                'score': sc,
                'finish_reason': finish,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if len(examples) < 3:
                examples.append(rec)

    n = max(1, len(scores))
    acc = sum(scores) / n
    result = {
        'tag': args.tag,
        'model': args.model,
        'n': len(scores),
        'accuracy': round(acc, 4),
        'format_rate': round(n_with_hash / n, 4),
        'truncated_rate': round(n_truncated / n, 4),
        'gen_seconds': round(gen_s, 1),
        'details': out_path,
    }
    print('[eval] n=%d accuracy=%.4f format_rate=%.4f truncated=%.4f gen_seconds=%.1f'
          % (result['n'], result['accuracy'], result['format_rate'], result['truncated_rate'], gen_s),
          flush=True)
    print('[eval] sample answers:')
    for ex in examples:
        print('  gt=%s score=%s finish=%s | %s' % (
            ex['ground_truth'], ex['score'], ex['finish_reason'],
            ex['output'].replace('\n', ' ')[:180]), flush=True)
    print('RESULT_JSON=' + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
