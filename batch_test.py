import numpy as np
import csv
from ssvepdetect import ssvepDetect

srate = 250
freqs = [8, 9, 10, 11, 12, 13, 14, 15]

subjects_task1 = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
subjects_task2 = ['S7', 'S8', 'S9', 'S10', 'S11', 'S12']

base = r'E:/HuanCun/Desktop/数据c3'

for name in subjects_task1:
    dataLen = 3  # Task1: 3 seconds per trial
    sd = ssvepDetect(srate, freqs, dataLen)
    datapath = f'{base}/Task1/{name}.csv'
    data = []
    with open(datapath, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data.append([float(_) for _ in row])
    data = np.array(data, dtype=np.float64)
    points = dataLen * srate  # 750
    results = []
    for i in range(48):
        epoch = data[i*points:(i+1)*points, :6]
        epoch = epoch.transpose()
        res = sd.detect(epoch)
        results.append(res)
    counts = [results.count(c) for c in range(8)]
    print(f"\n=== {name} (Task1, 3s/trial) ===")
    print(f"  预测分布: {counts}")
    print(f"  是否均匀(全6个): {all(c == 6 for c in counts)}")
    acc = sum(1 for i in range(48) if results[i] == int(data[i*points, -1])) / 48
    print(f"  准确率: {acc:.2%}")

for name in subjects_task2:
    dataLen = 1  # Task2: 1 second per trial
    sd = ssvepDetect(srate, freqs, dataLen)
    datapath = f'{base}/Task2/{name}.csv'
    data = []
    with open(datapath, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data.append([float(_) for _ in row])
    data = np.array(data, dtype=np.float64)
    points = dataLen * srate  # 250
    results = []
    for i in range(48):
        epoch = data[i*points:(i+1)*points, :6]
        epoch = epoch.transpose()
        res = sd.detect(epoch)
        results.append(res)
    counts = [results.count(c) for c in range(8)]
    print(f"\n=== {name} (Task2, 1s/trial) ===")
    print(f"  预测分布: {counts}")
    print(f"  是否均匀(全6个): {all(c == 6 for c in counts)}")
    acc = sum(1 for i in range(48) if results[i] == int(data[i*points, -1])) / 48
    print(f"  准确率: {acc:.2%}")
