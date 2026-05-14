import numpy as np
import csv
# from ss import ssvepDetect
from ssvepdetect import ssvepDetect

if __name__ == '__main__':
    # 实验数据路径
    # datapath = r'../ExampleData/D2.csv'
    datapath = r'E:/HuanCun/Desktop/Data/D1.csv'
    # datapath = r'E:/HuanCun/Desktop/Data/D2.csv'

    # 实验参数
    srate = 250 # 采样率250Hz     默认
    dataLen = 4 # example数据长度为4秒     省赛不止4秒，按省赛给的数据长度来设置

    # 实例化ssvep检测器
    # 输入参数：
    # srate: 采样率250Hz
    # 实验刺激频率：[8,9,10,11,12,13,14,15] 不要修改
    # dataLen：待分析数据信号片段的长度
    sd = ssvepDetect(srate,[8,9,10,11,12,13,14,15],dataLen)

    # 读取数据
    data = [] # 用来存储所有原始数据

    # 打开CSV文件，读取数据
    with open(datapath, mode='r') as file:
        csv_reader = csv.reader(file)
        # 跳过第一行表头
        next(csv_reader)

        for row in csv_reader:
            # csv以字符串形式存储，需要转换成浮点型
            rowvalue = [float(_) for _ in row]
            # 所有数据整理后存入data列表中
            data.append(rowvalue)

    # 将列表型转换成np.array型，便于后续处理
    data = np.array(data,dtype=np.float64)

    points = dataLen * srate
    results = []
    stimIDs = []
    corr = []

    # 每个数据中都有48个片段
    for i in range(48):
        epoch = data[i*points:(i+1)*points,:6] # 把这一段的6个通道信号片段取出
        epoch = epoch.transpose() # 以行来组织，每一行是一个通道的数据
        res = sd.detect(epoch) # 识别，得到的结果res取值范围是0-7
        results.append(res)
        # 如果这是示例数据，则能够得到真值
        stim = int(data[i*points,-1])
        stimIDs.append(stim)

        if res == stim:
            correct = 1
        else:
            correct = 0

        corr.append(correct)

    print("正确率： %.2f"%(sum(corr)/48))

    # results里面包含了所有的预测值，应当按顺序填写到result.csv中，并将结果反馈至组委会
    for i in range(48):
        print("task%d预测值：%d"%(i,results[i]))







import numpy as np
import csv
# from ss import ssvepDetect
from ssvepdetect import ssvepDetect

if __name__ == '__main__':
    # 实验数据路径
    # datapath = r'../ExampleData/D2.csv'
    # datapath = r'E:/HuanCun/Desktop/Data/D1.csv'
    datapath = r'E:/HuanCun/Desktop/Data/D2.csv'


    # 实验参数
    srate = 250 # 采样率250Hz
    dataLen = 4 # example数据长度为4秒

    # 实例化ssvep检测器
    # 输入参数：
    # srate: 采样率250Hz
    # 实验刺激频率：[8,9,10,11,12,13,14,15] 不要修改
    # dataLen：待分析数据信号片段的长度
    sd = ssvepDetect(srate,[8,9,10,11,12,13,14,15],dataLen)

    # 读取数据
    data = [] # 用来存储所有原始数据

    # 打开CSV文件，读取数据
    with open(datapath, mode='r') as file:
        csv_reader = csv.reader(file)
        # 跳过第一行表头
        next(csv_reader)

        for row in csv_reader:
            # csv以字符串形式存储，需要转换成浮点型
            rowvalue = [float(_) for _ in row]
            # 所有数据整理后存入data列表中
            data.append(rowvalue)

    # 将列表型转换成np.array型，便于后续处理
    data = np.array(data,dtype=np.float64)

    points = dataLen * srate
    results = []
    stimIDs = []
    corr = []

    # 每个数据中都有48个片段
    for i in range(48):
        epoch = data[i*points:(i+1)*points,:6] # 把这一段的6个通道信号片段取出
        epoch = epoch.transpose() # 以行来组织，每一行是一个通道的数据
        res = sd.detect(epoch) # 识别，得到的结果res取值范围是0-7
        results.append(res)
        # 如果这是示例数据，则能够得到真值
        stim = int(data[i*points,-1])
        stimIDs.append(stim)

        if res == stim:
            correct = 1
        else:
            correct = 0

        corr.append(correct)

    print("正确率： %.2f"%(sum(corr)/48))

    # results里面包含了所有的预测值，应当按顺序填写到result.csv中，并将结果反馈至组委会
    for i in range(48):
        print("task%d预测值：%d"%(i,results[i]))









# # ============================================================
# # 冗余指令映射演示（不修改上方任何代码，仅用于验证效果）
# # 将 8 个频率映射到 4 个指令，每个指令对应 2 个频率
# # 即使认错频率，只要落在同一指令内就不影响操作
# # ============================================================
# print("\n===== 冗余指令映射验证 =====")

# # 指令映射表：每个频率(0-7)对应的指令编号
# # 0(8Hz)+4(12Hz) = 前进(命令0)
# # 1(9Hz)+5(13Hz) = 左转(命令1)
# # 2(10Hz)+6(14Hz) = 右转(命令2)
# # 3(11Hz)+7(15Hz) = 停止/鸣笛(命令3)
# freq_to_cmd = [0, 1, 2, 3, 0, 1, 2, 3]
# cmd_names = ["前进", "左转", "右转", "停止/鸣笛"]

# # 计算指令级准确率
# cmd_correct = 0
# for i in range(48):
#     pred_cmd = freq_to_cmd[results[i]]
#     true_cmd = freq_to_cmd[stimIDs[i]]
#     if pred_cmd == true_cmd:
#         cmd_correct += 1

# print("频率级准确率: %d/48 = %.2f" % (sum(corr), sum(corr)/48))
# print("指令级准确率: %d/48 = %.2f" % (cmd_correct, cmd_correct/48))
# print("提升: +%d/48 = +%.2f" % (cmd_correct - sum(corr), (cmd_correct - sum(corr))/48))

# # 显示详细对比
# print("\n详细对比：")
# for i in range(48):
#     pred_cmd = freq_to_cmd[results[i]]
#     true_cmd = freq_to_cmd[stimIDs[i]]
#     freq_ok = "O" if results[i] == stimIDs[i] else "X"
#     cmd_ok = "O" if pred_cmd == true_cmd else "X"
#     hz = 8 + stimIDs[i]
#     pred_hz = 8 + results[i]
#     print("task%2d 真实%2dHz(%-6s) 预测%2dHz(%-6s) 频%1s 指%1s" % (
#         i, hz, cmd_names[true_cmd], pred_hz, cmd_names[pred_cmd], freq_ok, cmd_ok))

# ============================================================
# 窗口长度对比测试（不修改上方已有代码）
# 测试不同数据长度对准确率的影响，找到"速度-精度"最优折中
# 场地任务中：窗口越短 → 响应越快 → 比赛用时越短
# ============================================================
# print("\n" + "=" * 50)
# print("窗口长度对比测试")
# print("=" * 50)

# # 要测试的窗口长度（秒）
# window_lengths = [2.0, 2.5, 3.0, 3.5, 4.0]

# # 读取数据（复用上方的 data 变量，但这里再读一遍确保独立）
# for wl in window_lengths:
#     n_samples = int(wl * srate)
#     if n_samples > 1000:
#         continue

#     # 为当前窗口长度新建检测器（自动重建对应长度的参考模板和滤波器）
#     sd_w = ssvepDetect(srate, [8, 9, 10, 11, 12, 13, 14, 15], wl)

#     corr_w = 0
#     for i in range(48):
#         start = i * 1000
#         epoch = data[start:start + n_samples, :6].T
#         res = sd_w.detect(epoch)
#         stim = int(data[start, -1])
#         if res == stim:
#             corr_w += 1

#     acc = corr_w / 48
#     bar = "#" * int(acc * 50)
#     print("%.1fs (%3d点)  %3d/48 = %.2f  %s" % (wl, n_samples, corr_w, acc, bar))

# # ============================================================
# # 场地任务实操建议
# # ============================================================
# print("\n" + "=" * 50)
# print("场地任务实操分析：离线 vs 在线差异")
# print("=" * 50)

# offline_lines = [
#     ("环境因素", [
#         "屏幕刷新率不匹配（60Hz屏幕呈现8-15Hz刺激可能产生频闪混叠）",
#         "现场灯光干扰（日光灯50/100Hz频闪与SSVEP信号重叠）",
#         "电磁干扰（电机、电源、其他设备产生额外噪声）",
#     ]),
#     ("操作者因素", [
#         "眨眼/眼动产生大幅伪迹（比SSVEP信号大数十倍）",
#         "视觉疲劳导致SSVEP响应幅度下降（盯久了大脑不再强烈响应）",
#         "分心/偏离刺激目标（看错了频率但算法仍在运行）",
#         "忘记指令映射关系（操作犹豫增加误触概率）",
#     ]),
#     ("系统差异", [
#         "离线数据是分段干净的trial -> 在线是连续流信号，无明确起点",
#         "离线固定4秒 -> 在线要决策[何时输出指令]，涉及滑动窗口策略",
#         "在线需实时处理 -> 必须考虑计算延迟",
#     ]),
# ]

# for category, items in offline_lines:
#     print("\n[%s]" % category)
#     for item in items:
#         print("  - " + item)

# print("\n总结：离线98%不代表在线也能到98%。场地任务中")
# print("建议先用较长窗口(3-4s)保证稳定，熟悉后再逐步缩短。")
# print("同时用冗余指令映射兜底，减少误判损失。")

# # ============================================================
# # 无标签自检：没有stimID时通过置信度选择最优谐波数
# # 适用场景：比赛测试数据没有标签，但仍想选最佳参数
# # ============================================================
# print("\n" + "=" * 50)
# print("无标签自检：通过置信度选择最优谐波数")
# print("=" * 50)

# from scipy import signal as scipysignal
# from sklearn.cross_decomposition import CCA

# test_window = 3.0
# n_samples = int(test_window * srate)

# for n_harm in [1, 2, 3, 4]:
#     # 为当前谐波数构建参考信号
#     templLen = int(test_window * srate)
#     t = np.linspace(0, (templLen - 1) / srate, templLen)
#     templates = []
#     for freq in [8, 9, 10, 11, 12, 13, 14, 15]:
#         refs = []
#         for h in range(1, n_harm + 1):
#             refs.append(np.sin(2 * np.pi * freq * h * t))
#             refs.append(np.cos(2 * np.pi * freq * h * t))
#         templates.append(np.vstack(refs))

#     # FBCCA 4子带
#     nyquist = srate / 2
#     filter_banks = []
#     for low in range(6, 38, 8):
#         b, a = scipysignal.butter(4, [low / nyquist, 90 / nyquist], btype='bandpass')
#         filter_banks.append((b, a))
#     sw = np.array([(k + 1) ** (-1.25) + 0.25 for k in range(len(filter_banks))])

#     margins = []
#     preds = []

#     for i in range(48):
#         start = i * 1000
#         epoch = data[start:start + n_samples, :6].T

#         # 预处理
#         d = scipysignal.detrend(epoch, axis=1)
#         b, a = scipysignal.iircomb(50, 35, ftype='notch', fs=srate)
#         d = scipysignal.filtfilt(b, a, d)
#         fs2 = srate / 2
#         N, Wn = scipysignal.ellipord([7 / fs2, 90 / fs2], [3 / fs2, 100 / fs2], 3, 40)
#         b1, a1 = scipysignal.ellip(N, 1, 90, Wn, 'bandpass')
#         d = scipysignal.filtfilt(b1, a1, d)

#         # 选通道
#         sel = np.argsort(d.std(axis=1))[-3:]
#         d = d[sel]
#         d -= d.mean(axis=1, keepdims=True)

#         # FBCCA
#         rho = np.zeros((len(filter_banks), 8))
#         for si, (b_fb, a_fb) in enumerate(filter_banks):
#             flt = scipysignal.filtfilt(b_fb, a_fb, d)
#             for fi, t in enumerate(templates):
#                 c = CCA(n_components=1)
#                 c.fit(flt.T, t.T)
#                 dt, tt = c.transform(flt.T, t.T)
#                 rho[si, fi] = np.corrcoef(dt[:, 0], tt[:, 0])[0, 1]
#         scores = sw @ rho
#         preds.append(int(np.argmax(scores)))
#         sorted_s = sorted(scores, reverse=True)
#         margins.append((sorted_s[0] - sorted_s[1]) / (sorted_s[0] + 1e-10))

#     avg_conf = np.mean(margins)
#     dist = np.bincount(preds, minlength=8)
#     balance = 1.0 - np.std(dist) / (np.mean(dist) + 1e-10)

#     print("谐波=%d  置信度=%.3f  分布均匀度=%.2f  预测分布=%s" % (
#         n_harm, avg_conf, balance, str(dist.tolist())))

# print("\n说明：")
# print("  置信度越高 -> 算法对结果越确定 -> 准确率可能越高")
# print("  分布均匀度接近1.0 -> 48 trial应~6个/类，合理")
# print("  综合选置信度最高且分布均匀的参数")
# print("  (无标签只能间接推断，有标签时仍以准确率为准)")
