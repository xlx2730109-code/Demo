"""分析匈牙利均衡分配的置信度：哪些trial被强制修改了、修改前后的分数差有多大"""
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ssvepdetect import ssvepDetect

SRATE = 250
FREQS = [8, 9, 10, 11, 12, 13, 14, 15]
NUM_TRIALS = 48
PER_CLASS = NUM_TRIALS // len(FREQS)
DATA_ROOT = Path(r"E:/HuanCun/Desktop/数据c3")
CLASS_NAMES = ["8Hz", "9Hz", "10Hz", "11Hz", "12Hz", "13Hz", "14Hz", "15Hz"]


def analyze_confidence(scores, balanced_labels):
    """
    对每个trial计算置信度指标，并标出匈牙利强制修改的trial。

    Parameters
    ----------
    scores : np.ndarray, shape (48, 8)
        FBCCA scores for each trial and each frequency.
    balanced_labels : list[int], length 48
        经匈牙利均衡后的最终预测。

    Returns
    -------
    list[dict], each containing trial-level info.
    """
    raw_labels = np.argmax(scores, axis=1)
    sorted_scores = np.sort(scores, axis=1)
    max_score = sorted_scores[:, -1]
    second_max = sorted_scores[:, -2]
    margin = max_score - second_max           # 冠军与亚军的分数差
    margin_ratio = margin / (second_max + 1e-10)  # 相对差距

    info = []
    for i in range(NUM_TRIALS):
        forced = (raw_labels[i] != balanced_labels[i])
        info.append({
            "trial": i,
            "raw_pred": int(raw_labels[i]),
            "balanced_pred": int(balanced_labels[i]),
            "forced": forced,
            "max_score": max_score[i],
            "second_max": second_max[i],
            "margin": margin[i],
            "margin_ratio": margin_ratio[i],
            "scores": scores[i].round(4),
        })
    return info


def run_subject(datapath):
    data = np.loadtxt(datapath, delimiter=",", skiprows=1, dtype=np.float64)
    points = data.shape[0] // NUM_TRIALS
    data_len = points / SRATE
    detector = ssvepDetect(SRATE, FREQS, data_len)

    score_rows = []
    for i in range(NUM_TRIALS):
        epoch = data[i * points : (i + 1) * points, :6].transpose()
        score_rows.append(detector.detect_scores(epoch))

    scores = np.asarray(score_rows, dtype=np.float64)

    # 匈牙利均衡
    expanded_labels = np.repeat(np.arange(scores.shape[1]), PER_CLASS)
    row_ind, col_ind = linear_sum_assignment(-scores[:, expanded_labels])
    balanced = np.empty(scores.shape[0], dtype=int)
    balanced[row_ind] = expanded_labels[col_ind]

    return scores, balanced.tolist()


def print_report(name, scores, balanced_labels, info):
    forced_trials = [d for d in info if d["forced"]]
    num_forced = len(forced_trials)
    raw_labels = [d["raw_pred"] for d in info]
    raw_counts = [raw_labels.count(c) for c in range(8)]

    print(f"\n{'='*65}")
    print(f"{name}  原始分布: {raw_counts}  |  均衡修改: {num_forced}/{NUM_TRIALS}")
    print(f"{'='*65}")

    if num_forced == 0:
        print("  无需修改，置信度天然满足每类6个。")
        print(f"  平均冠军-亚军分差: {np.mean([d['margin'] for d in info]):.4f}")
        print(f"  最小分差: {min(d['margin'] for d in info):.4f}")
        return

    # 被强制修改的trial的置信度
    forced_margins = [d["margin"] for d in forced_trials]
    unforced_margins = [d["margin"] for d in info if not d["forced"]]

    print(f"\n  -> 置信度对比:")
    print(f"    未修改trial平均分差: {np.mean(unforced_margins):.4f}")
    print(f"    强制修改trial平均分差: {np.mean(forced_margins):.4f}")

    # 低置信度trial统计 (margin < 0.05)
    low_conf = [d for d in info if d["margin"] < 0.05]
    print(f"    分差<0.05的低置信trial: {len(low_conf)}个")
    for d in low_conf:
        tag = " [forced]" if d["forced"] else ""
        print(f"      trial{d['trial']:2d}: "
              f"raw={d['raw_pred']}({CLASS_NAMES[d['raw_pred']]}) "
              f"=> bal={d['balanced_pred']}({CLASS_NAMES[d['balanced_pred']]}) "
              f"margin={d['margin']:.4f}{tag}")

    # 列出所有被强制修改的trial
    print(f"\n  -> 被匈牙利强制修改的trial ({num_forced}个):")
    forced_trials_sorted = sorted(forced_trials, key=lambda d: d["margin"])
    for d in forced_trials_sorted:
        print(f"    trial{d['trial']:2d}: "
              f"{d['raw_pred']}({CLASS_NAMES[d['raw_pred']]}) "
              f"=> {d['balanced_pred']}({CLASS_NAMES[d['balanced_pred']]})  "
              f"margin={d['margin']:.4f}  "
              f"max={d['max_score']:.4f} 2nd={d['second_max']:.4f}")

    # 汇总：强制修改是否合理？
    questionable = [d for d in forced_trials if d["margin"] > 0.10]
    if questionable:
        print(f"\n  [!] 注意: {len(questionable)}个强制修改的trial分差>0.10，"
              f"原始argmax置信度较高却被强行更改:")
        for d in questionable:
            print(f"    trial{d['trial']:2d}: "
                  f"{d['raw_pred']}({CLASS_NAMES[d['raw_pred']]}) "
                  f"=> {d['balanced_pred']}({CLASS_NAMES[d['balanced_pred']]})  "
                  f"margin={d['margin']:.4f}")
    else:
        print(f"\n  [OK] 所有强制修改的trial分差都<0.10，原始argmax本身就很模糊，"
              f"匈牙利均衡是合理的。")


if __name__ == "__main__":
    datapaths = sorted(DATA_ROOT.glob("Task*/S*.csv"), key=lambda p: int(p.stem[1:]))

    for datapath in datapaths:
        scores, balanced = run_subject(datapath)
        info = analyze_confidence(scores, balanced)
        print_report(datapath.stem, scores, balanced, info)
