"""约束门控原型：先锁定高置信trial，剩余名额内均衡分配，确保每类6个。"""
import csv
from collections import Counter
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


def confidence_margins(score_rows):
    scores = np.asarray(score_rows, dtype=np.float64)
    top2 = np.sort(scores, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def balance_predictions(score_rows, per_class=PER_CLASS):
    """原始匈牙利均衡（全部trial参与）。"""
    scores = np.asarray(score_rows, dtype=np.float64)
    expanded_labels = np.repeat(np.arange(scores.shape[1]), per_class)
    row_ind, col_ind = linear_sum_assignment(-scores[:, expanded_labels])
    results = np.empty(scores.shape[0], dtype=int)
    results[row_ind] = expanded_labels[col_ind]
    return results.tolist()


def apply_confidence_gate(raw_results, balanced_results, score_rows, threshold):
    """当前门控方案：低于阈值的接受匈牙利修改。"""
    margins = confidence_margins(score_rows)
    gated_results = list(raw_results)
    accepted = []
    blocked = []
    for i in range(len(raw_results)):
        if raw_results[i] != balanced_results[i]:
            if margins[i] <= threshold:
                gated_results[i] = balanced_results[i]
                accepted.append(i)
            else:
                blocked.append(i)
    return gated_results, accepted, blocked


def constrained_gating(raw_results, score_rows, threshold):
    """
    约束门控方案 v2：
    1. 每类最多锁定PER_CLASS个最高置信的trial，保持raw结果
    2. 超出的高置信trial释放回浮动池
    3. 浮动池在剩余名额下跑匈牙利均衡
    """
    margins = confidence_margins(score_rows)
    scores = np.asarray(score_rows, dtype=np.float64)
    n = len(raw_results)
    n_classes = len(FREQS)

    # 1. 按类分组，每类按margin降序排列
    fixed = set()
    for cls in range(n_classes):
        # 找到raw_pred==cls的trial，按margin降序排列
        cls_trials = [(i, margins[i]) for i in range(n) if raw_results[i] == cls]
        cls_trials.sort(key=lambda x: -x[1])  # margin降序
        # 锁定margin>阈值 且 不超过PER_CLASS个
        locked = 0
        for i, m in cls_trials:
            if m > threshold and locked < PER_CLASS:
                fixed.add(i)
                locked += 1

    # 2. 浮动池 = 未锁定的trial
    floating = [i for i in range(n) if i not in fixed]

    # 3. 统计每类已锁定的数量和剩余名额
    fixed_counts = Counter(raw_results[i] for i in fixed)
    remaining = {c: PER_CLASS - fixed_counts.get(c, 0) for c in range(n_classes)}
    total_remaining = sum(remaining.values())

    if total_remaining != len(floating):
        # 兜底：退化为全匈牙利
        expanded_labels = np.repeat(np.arange(n_classes), PER_CLASS)
        row_ind, col_ind = linear_sum_assignment(-scores[:, expanded_labels])
        final = np.empty(n, dtype=int)
        final[row_ind] = expanded_labels[col_ind]
        return final.tolist(), list(fixed), floating, []

    # 4. 构建名额扩展标签
    class_indices = []
    for c in range(n_classes):
        if remaining[c] > 0:
            class_indices.extend([c] * remaining[c])
    target_labels = np.array(class_indices)

    # 5. 浮动trial在剩余名额下跑匈牙利
    if len(floating) > 0:
        float_scores = scores[floating]
        cost_matrix = -float_scores[:, target_labels]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_labels = target_labels[col_ind]
    else:
        assigned_labels = []

    # 6. 组装最终结果
    final_results = [0] * n
    for i in fixed:
        final_results[i] = raw_results[i]
    for idx, i in enumerate(floating):
        final_results[i] = int(assigned_labels[idx])

    forced = [i for idx, i in enumerate(floating)
              if raw_results[i] != assigned_labels[idx]]

    return final_results, list(fixed), floating, forced


def count_predictions(results):
    counts = Counter(results)
    return [counts.get(c, 0) for c in range(len(FREQS))]


def run_subject(datapath, threshold=0.15):
    data = np.loadtxt(datapath, delimiter=",", skiprows=1, dtype=np.float64)
    points = data.shape[0] // NUM_TRIALS
    data_len = points / SRATE
    detector = ssvepDetect(SRATE, FREQS, data_len)

    score_rows = []
    for i in range(NUM_TRIALS):
        epoch = data[i * points : (i + 1) * points, :6].transpose()
        scores = detector.detect_scores(epoch)
        score_rows.append(scores)
    raw_results = [int(np.argmax(s)) for s in score_rows]

    # baseline: 原始匈牙利
    balanced_results = balance_predictions(score_rows)

    # baseline: 当前门控
    gated_results, acc, blk = apply_confidence_gate(raw_results, balanced_results, score_rows, threshold)

    # new: 约束门控
    constrained_results, high_conf, low_conf, forced = constrained_gating(raw_results, score_rows, threshold)

    return {
        "name": datapath.stem,
        "raw_counts": count_predictions(raw_results),
        "balanced_counts": count_predictions(balanced_results),
        "gated_counts": count_predictions(gated_results),
        "constrained_counts": count_predictions(constrained_results),
        "changed_balanced": sum(a != b for a, b in zip(raw_results, balanced_results)),
        "gated_acc": len(acc), "gated_blk": len(blk),
        "conf_high": len(high_conf), "conf_low": len(low_conf),
        "conf_forced": len(forced),
    }


if __name__ == "__main__":
    datapaths = sorted(DATA_ROOT.glob("Task*/S*.csv"),
                       key=lambda p: int(p.stem[1:]))

    print(f"{'被试':>4s}  {'raw分布':>24s}  {'均衡(Hungarian)':>24s}  "
          f"{'门控(gated)':>24s}  {'约束门控(新)':>24s}  "
          f"{'均衡改':>5s}  {'门控拦':>5s}  {'约束改':>5s}")
    print("-" * 140)

    for datapath in datapaths:
        info = run_subject(datapath, threshold=0.15)
        print(f"{info['name']:>4s}  "
              f"{str(info['raw_counts']):>24s}  "
              f"{str(info['balanced_counts']):>24s}  "
              f"{str(info['gated_counts']):>24s}  "
              f"{str(info['constrained_counts']):>24s}  "
              f"{info['changed_balanced']:>5d}  "
              f"{info['gated_blk']:>5d}  "
              f"{info['conf_forced']:>5d}")
