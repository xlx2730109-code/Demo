import numpy as np
import csv
from ssvepdetect import ssvepDetect

if __name__ == '__main__':
    # 实验数据路径（比赛时改成实际数据路径）
    # datapath = r'E:/HuanCun/Desktop/Data/D2.csv'
    datapath = r'E:/HuanCun/Desktop/Data/D1.csv'

    # 实验参数
    srate = 250
    dataLen = 4

    sd = ssvepDetect(srate, [8, 9, 10, 11, 12, 13, 14, 15], dataLen)

    data = []
    with open(datapath, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data.append([float(_) for _ in row])
    data = np.array(data, dtype=np.float64)

    points = dataLen * srate
    results = []
    stimIDs = []
    corr = []

    for i in range(48):
        epoch = data[i*points:(i+1)*points, :6]
        epoch = epoch.transpose()
        res = sd.detect(epoch)
        results.append(res)
        stim = int(data[i*points, -1])
        stimIDs.append(stim)
        if res == stim:
            correct = 1
        else:
            correct = 0
        corr.append(correct)

    print("准确率: %.2f" % (sum(corr)/48))

    # 自动保存 result.csv
    result_path = r'E:/HuanCun/Desktop/技术文件/脑机接口基础算法/三人行/Demo/result.csv'
    try:
        with open(result_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['task', 'predict'])
            for i in range(48):
                writer.writerow([i, results[i]])
        print("结果已保存到: %s" % result_path)
    except PermissionError:
        print("result.csv 被占用，无法自动保存。请手动记录。")

    # 预测值统计
    print("\n预测值统计:")
    class_names = ['8Hz', '9Hz', '10Hz', '11Hz', '12Hz', '13Hz', '14Hz', '15Hz']
    for cls_id in range(8):
        count = results.count(cls_id)
        bar = '#' * count
        print("  %d(%s): %d个 %s" % (cls_id, class_names[cls_id], count, bar))

    # 逐个输出
    print("\n逐个输出:")
    for i in range(48):
        print("task%d预测值: %d" % (i, results[i]))
