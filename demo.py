import numpy as np      # numpy：用来处理数组和数学计算
import csv               # csv：用来读写CSV格式的数据文件
from ssvepdetect import ssvepDetect   # 从ssvepdetect.py导入SSVEP检测器


# 程序入口：只有直接运行这个文件时才会执行下面的代码
if __name__ == '__main__':
    # 数据文件路径（正式比赛时改成实际数据的路径）
    # datapath = r'E:/HuanCun/Desktop/Data/D2.csv'
    datapath = r'E:/HuanCun/Desktop/Data/D1.csv'

    # 实验参数
    srate = 250    # 采样率=250Hz，每秒采集250个数据点
    dataLen = 4    # 每个trial的时长=4秒（比赛Task1改3，Task2改1）
                   # 注意：用D1/D2练习数据模拟3秒/1秒时，D1/D2每个trial固定1000点
                   # 不能只改dataLen，还需把下方 i*points 改成 i*1000

    # 创建SSVEP检测器：输入采样率、8个刺激频率(8-15Hz)、数据长度
    sd = ssvepDetect(srate, [8, 9, 10, 11, 12, 13, 14, 15], dataLen)

    # 从CSV文件读取数据
    data = []
    with open(datapath, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)              # 跳过第一行表头
        for row in csv_reader:
            data.append([float(_) for _ in row])   # 把字符串转成数字
    data = np.array(data, dtype=np.float64)        # 转成numpy数组

    points = dataLen * srate     # 每个trial的采样点数，例如4*250=1000
    results = []                 # 存48个预测结果
    stimIDs = []                 # 存48个真实标签（仅示例数据有）
    corr = []                    # 存每个trial对错（1对0错）

    # 循环处理48个trial
    for i in range(48):
        # 取出第i个trial的6个通道，从第i*points行取到第(i+1)*points行
        epoch = data[i*points:(i+1)*points, :6]
        # 转置：变成"每行一个通道，每列一个时间点"的格式
        epoch = epoch.transpose()
        # 调用检测器，返回预测结果0-7（分别对应8-15Hz）
        res = sd.detect(epoch)
        results.append(res)      # 存预测结果
        # 读取真实标签（CSV最后一列，正式比赛没有这一列）
        stim = int(data[i*points, -1])
        stimIDs.append(stim)
        # 判断预测对不对
        if res == stim:
            correct = 1   # 对了
        else:
            correct = 0   # 错了
        corr.append(correct)

    # 打印准确率 = 正确个数 ÷ 总个数(48)
    print("准确率: %.2f" % (sum(corr)/48))

    # 自动保存预测结果到result.csv
    result_path = r'E:/HuanCun/Desktop/技术文件/脑机接口基础算法/三人行/Demo/result.csv'
    try:
        with open(result_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['task', 'predict'])   # 写表头
            for i in range(48):
                writer.writerow([i, results[i]])   # 写每个trial的预测值
        print("结果已保存到: %s" % result_path)
    except PermissionError:
        # 如果文件被其他程序(如Excel)打开，会报这个错
        print("result.csv 被占用，无法自动保存。请手动记录。")

    # 统计每个频率被预测了多少次
    print("\n预测值统计:")
    class_names = ['8Hz', '9Hz', '10Hz', '11Hz', '12Hz', '13Hz', '14Hz', '15Hz']
    for cls_id in range(8):
        count = results.count(cls_id)          # 数一数这个频率出现了几次
        bar = '#' * count                       # 画柱状条
        print("  %d(%s): %d个 %s" % (cls_id, class_names[cls_id], count, bar))

    # 逐个输出48个预测值（核对用）
    print("\n逐个输出:")
    for i in range(48):
        print("task%d预测值: %d" % (i, results[i]))
