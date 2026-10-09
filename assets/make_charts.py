#!/usr/bin/env python3
"""生成 README 里用的结果图。

数据来源（全部为本次实验实测值，非示意）：
  1) eval_comparison —— 用 eval_gsm8k.py 在 GSM8K test 全量 1319 题上评测（greedy）
  2) val_curve       —— 训练日志里的 val-core/openai/gsm8k/acc/mean@1（每 25 步一次）
  3) reward_epoch1   —— 第 1 个 epoch（step 1~29）的训练奖励 critic/score/mean

用法：python assets/make_charts.py     （需要 matplotlib，见 requirements.txt）
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, GREY = "#4C78A8", "#E8894A", "#5FA05A", "#999999"


def finish(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def eval_comparison():
    """图 1：三个模型的评测对比（分组柱状图）。"""
    groups = ["原始模型\nQwen2.5-0.5B-Instruct", "训练 1 epoch\n(step 29)", "训练 5 epoch\n(step 145)"]
    strict = [0.15, 44.12, 53.37]     # 严格口径：必须按 #### 格式答对（= PPO 奖励口径）
    lenient = [46.40, 49.28, 53.45]   # 宽松口径：抽出最终数字比对（真实解题能力）
    fmt = [2.27, 92.95, 98.26]        # 输出里带 #### 的比例

    x = range(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    b1 = ax.bar([i - w for i in x], strict, w, label="严格口径（即 PPO 奖励）", color=BLUE)
    b2 = ax.bar(list(x), lenient, w, label="宽松口径（真实解题能力）", color=ORANGE)
    b3 = ax.bar([i + w for i in x], fmt, w, label="格式合规率（含 ####）", color=GREEN)

    for bars in (b1, b2, b3):
        for r in bars:
            ax.annotate("%.2f%%" % r.get_height(),
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_ylabel("比例 (%)", fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_title("GSM8K 测试集评测结果（全量 1319 题，贪心解码）", fontsize=11.5, pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    finish(fig, "eval_comparison.png")


def val_curve():
    """图 2：训练内验证准确率（每 25 步一次）。"""
    steps = [50, 75, 100, 125, 145]
    acc = [50.49, 50.64, 51.18, 52.31, 51.78]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhspan(min(acc), max(acc), color=BLUE, alpha=0.08, zorder=0)
    ax.plot(steps, acc, "-o", color=BLUE, linewidth=2, markersize=7, zorder=3)
    for s, a in zip(steps, acc):
        ax.annotate("%.2f%%" % a, (s, a), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color="#333333")
    peak = acc.index(max(acc))
    ax.annotate("峰值 %.2f%%\n(step %d)" % (max(acc), steps[peak]),
                (steps[peak], acc[peak]), textcoords="offset points",
                xytext=(8, -36), fontsize=9, color="#B4552D")
    ax.set_xlabel("训练步数 (global step)", fontsize=10)
    ax.set_ylabel("验证准确率 (%)", fontsize=10)
    ax.set_ylim(min(acc) - 1.3, max(acc) + 1.3)
    ax.set_title("训练内验证准确率：在 52% 附近形成平台", fontsize=11.5, pad=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.5, -0.30,
            "阴影带 = 5 个采样点的取值范围（跨度仅 1.8 pp）；1319 题的标准误约 1.4 pp",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#666666")
    finish(fig, "val_curve.png")


def reward_epoch1():
    """图 3：第 1 个 epoch 的奖励曲线（完整 29 步）。"""
    r = [0.0156, 0.0078, 0.0039, 0.0039, 0.0234, 0.0, 0.0039, 0.0078, 0.0078, 0.0039,
         0.0117, 0.0117, 0.0078, 0.0078, 0.0156, 0.0156, 0.0352, 0.0469, 0.0664,
         0.0703, 0.0859, 0.1211, 0.1523, 0.2344, 0.2578, 0.2813, 0.3086, 0.3125, 0.3398]
    steps = list(range(1, len(r) + 1))

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.axvspan(0.5, 16.5, color=GREY, alpha=0.12, zorder=0)
    ax.plot(steps, r, "-o", color=ORANGE, linewidth=2, markersize=5, zorder=3)
    ax.annotate("前 16 步：奖励几乎为 0\n（在摸索输出格式）", (8, 0.02), ha="center",
                fontsize=9.5, color="#555555", xytext=(8.5, 0.10), textcoords="data")
    ax.annotate("第 17 步起飞 → 0.34", (17.5, 0.30), fontsize=9.5, color="#B4552D")
    ax.axvline(16.5, color=GREY, linestyle="--", linewidth=1)
    ax.set_xlabel("训练步数 (global step)", fontsize=10)
    ax.set_ylabel("训练奖励 critic/score/mean", fontsize=10)
    ax.set_xlim(0.2, 29.8)
    ax.set_ylim(-0.02, 0.40)
    ax.set_title("第 1 个 epoch 的奖励曲线：先学会交卷格式，奖励才开始增长",
                 fontsize=11.5, pad=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    finish(fig, "reward_epoch1.png")


if __name__ == "__main__":
    eval_comparison()
    val_curve()
    reward_epoch1()
    print("done.")
