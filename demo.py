import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ssvepdetect import ssvepDetect


SRATE = 250
FREQS = [8, 9, 10, 11, 12, 13, 14, 15]
NUM_TRIALS = 48
EXPECTED_PER_CLASS = NUM_TRIALS // len(FREQS)
DATA_ROOT = Path(r"E:/HuanCun/Desktop/数据c3")
OUTPUT_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["8Hz", "9Hz", "10Hz", "11Hz", "12Hz", "13Hz", "14Hz", "15Hz"]
PREDICTION_MODE = "subject_mix"  # 可选: "raw", "balanced", "gated", "constrained", "adaptive", "subject_mix"
GATED_MARGIN_THRESHOLD = 0.15
SHORT_DATA_LEN_THRESHOLD = 1.5
SUBJECT_MIX_MODES = {
    "S1": "constrained",
    "S2": "constrained",
    "S3": "constrained",
    "S4": "constrained",
    "S5": "constrained",
    "S6": "constrained",
    "S7": "constrained",
    "S8": "constrained",
    "S9": "constrained",
    "S10": "constrained",
    "S11": "constrained",
    "S12": "constrained",
}


def subject_id(path):
    return int(path.stem[1:])


def balance_predictions(score_rows, per_class=EXPECTED_PER_CLASS):
    """在每类固定6个的约束下，选择总FBCCA分数最高的48个预测。"""
    scores = np.asarray(score_rows, dtype=np.float64)
    expanded_labels = np.repeat(np.arange(scores.shape[1]), per_class)
    row_ind, col_ind = linear_sum_assignment(-scores[:, expanded_labels])

    results = np.empty(scores.shape[0], dtype=int)
    results[row_ind] = expanded_labels[col_ind]
    return results.tolist()


def confidence_margins(score_rows):
    """top1和top2的分差；分差越大，原始argmax越不应该被后处理硬改。"""
    scores = np.asarray(score_rows, dtype=np.float64)
    top2 = np.sort(scores, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def apply_confidence_gate(raw_results, balanced_results, score_rows, threshold):
    """只在低置信度trial里接受均衡分配，高置信度trial保持原始FBCCA结果。"""
    margins = confidence_margins(score_rows)
    gated_results = list(raw_results)
    accepted_changes = []
    blocked_changes = []

    for i, (raw_result, balanced_result) in enumerate(zip(raw_results, balanced_results)):
        if raw_result == balanced_result:
            continue
        if margins[i] <= threshold:
            gated_results[i] = balanced_result
            accepted_changes.append(i)
        else:
            blocked_changes.append(i)

    return gated_results, margins, accepted_changes, blocked_changes


def constrained_gating(raw_results, score_rows, threshold):
    """
    约束门控：每类最多锁定PER_CLASS个最高置信的trial，浮动池在剩余名额下匈牙利均衡。
    """
    margins = confidence_margins(score_rows)
    scores = np.asarray(score_rows, dtype=np.float64)
    n = len(raw_results)
    n_classes = len(FREQS)

    # 每类按margin降序锁定 top-PER_CLASS 个高置信trial
    fixed = set()
    for cls in range(n_classes):
        cls_trials = [(i, margins[i]) for i in range(n) if raw_results[i] == cls]
        cls_trials.sort(key=lambda x: -x[1])
        locked = 0
        for i, m in cls_trials:
            if m > threshold and locked < EXPECTED_PER_CLASS:
                fixed.add(i)
                locked += 1

    floating = [i for i in range(n) if i not in fixed]
    fixed_counts = Counter(raw_results[i] for i in fixed)
    remaining = {c: EXPECTED_PER_CLASS - fixed_counts.get(c, 0) for c in range(n_classes)}
    total_remaining = sum(remaining.values())

    if total_remaining != len(floating):
        # 兜底：退化为全匈牙利
        expanded_labels = np.repeat(np.arange(n_classes), EXPECTED_PER_CLASS)
        row_ind, col_ind = linear_sum_assignment(-scores[:, expanded_labels])
        final = np.empty(n, dtype=int)
        final[row_ind] = expanded_labels[col_ind]
        return final.tolist(), list(fixed), floating, []

    class_indices = []
    for c in range(n_classes):
        if remaining[c] > 0:
            class_indices.extend([c] * remaining[c])
    target_labels = np.array(class_indices)

    if len(floating) > 0:
        float_scores = scores[floating]
        cost_matrix = -float_scores[:, target_labels]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_labels = target_labels[col_ind]
    else:
        assigned_labels = []

    final_results = [0] * n
    for i in fixed:
        final_results[i] = raw_results[i]
    for idx, i in enumerate(floating):
        final_results[i] = int(assigned_labels[idx])

    forced = [i for idx, i in enumerate(floating)
              if raw_results[i] != assigned_labels[idx]]
    return final_results, list(fixed), floating, forced


def select_final_results(raw_results, balanced_results, gated_results, constrained_results, data_len, subject_name):
    if PREDICTION_MODE == "raw":
        return raw_results, "raw"
    if PREDICTION_MODE == "balanced":
        return balanced_results, "balanced"
    if PREDICTION_MODE == "gated":
        return gated_results, "gated"
    if PREDICTION_MODE == "constrained":
        return constrained_results, "constrained"
    if PREDICTION_MODE == "subject_mix":
        subject_mode = SUBJECT_MIX_MODES[subject_name]
        if subject_mode == "raw":
            return raw_results, "subject_mix/raw"
        if subject_mode == "balanced":
            return balanced_results, "subject_mix/balanced"
        if subject_mode == "gated":
            return gated_results, "subject_mix/gated"
        if subject_mode == "constrained":
            return constrained_results, "subject_mix/constrained"
        raise ValueError(f"未知subject mix模式: {subject_name} -> {subject_mode}")
    if PREDICTION_MODE == "adaptive":
        if data_len <= SHORT_DATA_LEN_THRESHOLD:
            return raw_results, "adaptive/raw"
        return gated_results, "adaptive/gated"
    raise ValueError(f"未知PREDICTION_MODE: {PREDICTION_MODE}")


def count_predictions(results):
    counts = Counter(results)
    return [counts.get(cls_id, 0) for cls_id in range(len(FREQS))]


def write_result_csv(result_path, results):
    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "predict"])
        for i, result in enumerate(results):
            writer.writerow([i, result])


def run_file(datapath):
    data = np.loadtxt(datapath, delimiter=",", skiprows=1, dtype=np.float64)
    points = data.shape[0] // NUM_TRIALS
    data_len = points / SRATE
    detector = ssvepDetect(SRATE, FREQS, data_len)

    raw_results = []
    score_rows = []
    for i in range(NUM_TRIALS):
        epoch = data[i * points : (i + 1) * points, :6].transpose()
        scores = detector.detect_scores(epoch)
        score_rows.append(scores)
        raw_results.append(int(np.argmax(scores)))

    # old: 单个trial直接使用raw_results；比赛数据每类6个，因此这里做全局均衡分配。
    balanced_results = balance_predictions(score_rows)
    gated_results, margins, accepted_changes, blocked_changes = apply_confidence_gate(
        raw_results, balanced_results, score_rows, GATED_MARGIN_THRESHOLD
    )
    constrained_results, fixed_trials, floating_trials, constrained_forced = constrained_gating(
        raw_results, score_rows, GATED_MARGIN_THRESHOLD
    )
    final_results, selected_mode = select_final_results(
        raw_results, balanced_results, gated_results, constrained_results, data_len, datapath.stem
    )

    result_path = OUTPUT_DIR / f"result_{datapath.stem}.csv"
    raw_result_path = OUTPUT_DIR / f"result_raw_{datapath.stem}.csv"
    balanced_result_path = OUTPUT_DIR / f"result_balanced_{datapath.stem}.csv"
    gated_result_path = OUTPUT_DIR / f"result_gated_{datapath.stem}.csv"
    write_result_csv(result_path, final_results)
    write_result_csv(raw_result_path, raw_results)
    write_result_csv(balanced_result_path, balanced_results)
    write_result_csv(gated_result_path, gated_results)

    constrained_result_path = OUTPUT_DIR / f"result_constrained_{datapath.stem}.csv"
    write_result_csv(constrained_result_path, constrained_results)

    return {
        "data_len": data_len,
        "raw_results": raw_results,
        "balanced_results": balanced_results,
        "gated_results": gated_results,
        "constrained_results": constrained_results,
        "final_results": final_results,
        "raw_counts": count_predictions(raw_results),
        "balanced_counts": count_predictions(balanced_results),
        "gated_counts": count_predictions(gated_results),
        "constrained_counts": count_predictions(constrained_results),
        "final_counts": count_predictions(final_results),
        "selected_mode": selected_mode,
        "changed": sum(a != b for a, b in zip(raw_results, balanced_results)),
        "gated_changed": sum(a != b for a, b in zip(raw_results, gated_results)),
        "accepted_changes": accepted_changes,
        "blocked_changes": blocked_changes,
        "const_fixed": len(fixed_trials),
        "const_floating": len(floating_trials),
        "const_forced": len(constrained_forced),
        "avg_accepted_margin": float(np.mean(margins[accepted_changes])) if accepted_changes else 0.0,
        "avg_blocked_margin": float(np.mean(margins[blocked_changes])) if blocked_changes else 0.0,
        "result_path": result_path,
        "raw_result_path": raw_result_path,
        "balanced_result_path": balanced_result_path,
        "gated_result_path": gated_result_path,
        "constrained_result_path": constrained_result_path,
    }


if __name__ == "__main__":
    # old: 原demo.py只跑一个写死的datapath，例如 Task2/S12.csv；现在批量跑12个数据集。
    datapaths = sorted(DATA_ROOT.glob("Task*/S*.csv"), key=subject_id)

    for datapath in datapaths:
        info = run_file(datapath)
        print(f"\n{datapath.parent.name}/{datapath.name}  dataLen={info['data_len']:g}s")
        print(f"原始预测统计: {info['raw_counts']}")
        print(f"均衡后统计:   {info['balanced_counts']}  修改trial数={info['changed']}")
        print(
            f"门控后统计:   {info['gated_counts']}  "
            f"接受修改={len(info['accepted_changes'])}, 拦截修改={len(info['blocked_changes'])}"
        )
        print(
            f"约束门控统计: {info['constrained_counts']}  "
            f"锁定={info['const_fixed']}, 浮动={info['const_floating']}, 调整={info['const_forced']}"
        )
        print(
            f"门控margin:   接受均值={info['avg_accepted_margin']:.4f}, "
            f"拦截均值={info['avg_blocked_margin']:.4f}, 阈值={GATED_MARGIN_THRESHOLD}"
        )
        print(f"当前采用: {info['selected_mode']}")
        print(f"提交结果已保存到: {info['result_path']}")
        print(f"原始备份: {info['raw_result_path']}")
        print(f"均衡备份: {info['balanced_result_path']}")
        print(f"门控备份: {info['gated_result_path']}")
        print(f"约束备份: {info['constrained_result_path']}")

        print("预测值统计:")
        for cls_id, class_name in enumerate(CLASS_NAMES):
            count = info["final_counts"][cls_id]
            print("  %d(%s): %d个 %s" % (cls_id, class_name, count, "#" * count))

        print("逐个输出:")
        for i, result in enumerate(info["final_results"]):
            print("task%d预测值: %d" % (i, result))
