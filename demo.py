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

    result_path = OUTPUT_DIR / f"result_{datapath.stem}.csv"
    write_result_csv(result_path, balanced_results)

    return {
        "data_len": data_len,
        "raw_results": raw_results,
        "balanced_results": balanced_results,
        "raw_counts": count_predictions(raw_results),
        "balanced_counts": count_predictions(balanced_results),
        "changed": sum(a != b for a, b in zip(raw_results, balanced_results)),
        "result_path": result_path,
    }


if __name__ == "__main__":
    # old: 原demo.py只跑一个写死的datapath，例如 Task2/S12.csv；现在批量跑12个数据集。
    datapaths = sorted(DATA_ROOT.glob("Task*/S*.csv"), key=subject_id)

    for datapath in datapaths:
        info = run_file(datapath)
        print(f"\n{datapath.parent.name}/{datapath.name}  dataLen={info['data_len']:g}s")
        print(f"原始预测统计: {info['raw_counts']}")
        print(f"均衡后统计:   {info['balanced_counts']}  修改trial数={info['changed']}")
        print(f"结果已保存到: {info['result_path']}")

        print("预测值统计:")
        for cls_id, class_name in enumerate(CLASS_NAMES):
            count = info["balanced_counts"][cls_id]
            print("  %d(%s): %d个 %s" % (cls_id, class_name, count, "#" * count))

        print("逐个输出:")
        for i, result in enumerate(info["balanced_results"]):
            print("task%d预测值: %d" % (i, result))
