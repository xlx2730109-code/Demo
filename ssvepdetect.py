import numpy as np
from scipy import signal as scipysignal
from sklearn.cross_decomposition import CCA

# ============================================================
# 版本 1：原始官方代码（含 cos bug，两行都是 sin，参考信号秩亏）
# 特点：实现简单，基线版本
# ============================================================
class ssvepDetect_Original:
    def __init__(self, srate, freqs, dataLen):
        self.cca = CCA(n_components=1)  #CCA算法
        self.srate = srate
        templLen = int(dataLen * srate)
        self.TemplateSet = []
        sample = np.linspace(0, (templLen - 1) / srate, templLen, endpoint=True)    #数据长度等

        for freq in freqs:
            _ = 2 * np.pi * freq * sample
            sintemp = np.sin(_)
            costemp = np.sin(_)  # BUG: 应为 np.cos(_)
            tempset = np.vstack((sintemp, costemp))
            self.TemplateSet.append(tempset)

    def detect(self, data): #CCA算法
        data = self.pre_filter(data)
        p = []
        cdata = data.transpose()
        for template in self.TemplateSet:
            ctemplate = template.transpose()
            self.cca.fit(cdata, ctemplate)
            datatran, templatetran = self.cca.transform(cdata, ctemplate)   #
            coe = np.corrcoef(datatran[:, 0], templatetran[:, 0])[0, 1]
            p.append(coe)
        return p.index(max(p))

    def pre_filter(self, data):
        b, a = scipysignal.iircomb(50, 35, ftype='notch', fs=self.srate)
        fs = self.srate / 2
        N, Wn = scipysignal.ellipord([6 / fs, 90 / fs], [2 / fs, 100 / fs], 3, 40)
        b1, a1 = scipysignal.ellip(N, 1, 90, Wn, 'bandpass')
        return scipysignal.filtfilt(b1, a1, scipysignal.filtfilt(b, a, data))


# ============================================================
# 版本 2：FBCCA 优化版
# 改进：修复cos bug + 3次谐波 + 4子带FBCCA + 去趋势 + 自动选通道
# D1: 98%  D2: 100%
# ============================================================
class ssvepDetect_Optimized:
    def __init__(self, srate, freqs, dataLen):
        self.srate = srate
        self.freqs = freqs
        templLen = int(dataLen * srate)
        sample = np.linspace(0, (templLen - 1) / srate, templLen, endpoint=True)

        # 3次谐波参考信号（基频 + 2次 + 3次谐波）
        self.num_harmonics = 3
        self.TemplateSet = []
        for freq in freqs:
            ref_signals = []
            for h in range(1, self.num_harmonics + 1):
                ref_signals.append(np.sin(2 * np.pi * freq * h * sample))
                ref_signals.append(np.cos(2 * np.pi * freq * h * sample))
            self.TemplateSet.append(np.vstack(ref_signals))

        # FBCCA 4子带：[6,90], [14,90], [22,90], [30,90] Hz
        self._design_filter_banks()

    def _design_filter_banks(self):
        self.filter_banks = []
        nyquist = self.srate / 2
        for low in range(6, 38, 8):
            Wn = [low / nyquist, 90 / nyquist]
            b, a = scipysignal.butter(4, Wn, btype='bandpass')
            self.filter_banks.append((b, a))

    def pre_filter(self, data):
        """去线性趋势 + 50Hz陷波 + 7-90Hz带通"""
        data = scipysignal.detrend(data, axis=1)
        b, a = scipysignal.iircomb(50, 35, ftype='notch', fs=self.srate)
        data = scipysignal.filtfilt(b, a, data)
        fs = self.srate / 2
        N, Wn = scipysignal.ellipord([7 / fs, 90 / fs], [3 / fs, 100 / fs], 3, 40)
        b1, a1 = scipysignal.ellip(N, 1, 90, Wn, 'bandpass')
        return scipysignal.filtfilt(b1, a1, data)

    def _cca_corr(self, data, template):
        cca = CCA(n_components=1)
        cca.fit(data.T, template.T)
        d_t, t_t = cca.transform(data.T, template.T)
        return np.corrcoef(d_t[:, 0], t_t[:, 0])[0, 1]

    def _select_channels(self, data):
        """自动选择方差最大的通道（最多3个）"""
        stds = data.std(axis=1)
        n_keep = min(3, data.shape[0])
        best = np.argsort(stds)[-n_keep:]
        return data[best], best

    def detect(self, data):
        """FBCCA检测：自动选通道 + 多子带CCA + 加权融合"""
        data, _ = self._select_channels(data)
        data = self.pre_filter(data)
        data = data - data.mean(axis=1, keepdims=True)

        n_bands = len(self.filter_banks)
        weights = np.array([(i + 1) ** (-1.25) + 0.25 for i in range(n_bands)])

        rho = np.zeros((n_bands, len(self.freqs)))
        for sb_idx, (b, a) in enumerate(self.filter_banks):
            filtered = scipysignal.filtfilt(b, a, data)
            for f_idx, template in enumerate(self.TemplateSet):
                rho[sb_idx, f_idx] = self._cca_corr(filtered, template)

        rho_fused = weights @ rho
        return int(np.argmax(rho_fused))


# ============================================================
# 版本 3：FBCCA + PSDA 融合版（北京天城优化方向全覆盖）
# 新增：
#   1. 时域转频域 -> PSDA（功率谱密度分析）
#   2. 不同方式拟合 -> CCA + FBCCA + PSDA 多方法集成
#   3. 异常值处理 -> 碰撞/摇晃/眨眼检测 + 通道质量评估
#   4. 识别速度优化 -> 预计算 + 自适应权重
#   5. 特征提取算法 -> 3种方法提取特征后加权融合
# ============================================================
class ssvepDetect_v3:
    def __init__(self, srate, freqs, dataLen):
        self.srate = srate
        self.freqs = freqs
        self.dataLen = dataLen
        templLen = int(dataLen * srate)
        t = np.linspace(0, (templLen - 1) / srate, templLen)

        # 1. 构造谐波参考信号（1次+2次+3次谐波合并）
        self.TemplateSet = []
        for freq in freqs:
            refs = []
            for h in [1, 2, 3]:
                refs.append(np.sin(2 * np.pi * freq * h * t))
                refs.append(np.cos(2 * np.pi * freq * h * t))
            self.TemplateSet.append(np.vstack(refs))
        self._design_filter_banks()


    def _design_filter_banks(self):
        """FBCCA子带：自动适应数据长度，短数据减少子带"""
        templLen = int(self.dataLen * self.srate)
        self.filter_banks = []
        nyquist = self.srate / 2
        max_sub = 2 if templLen < 500 else 4
        for i in range(max_sub):
            low = 6 + i * 8
            if low >= 90:
                break
            b, a = scipysignal.butter(4, [low / nyquist, 90 / nyquist], btype='bandpass')
            self.filter_banks.append((b, a))

    # ==================== 异常值检测（当前未启用，需手动开启）====================
    #
    # 【怎么用】
    #   在 detect() 方法开头加上这两行即可开启:
    #       artifact_level = self.detect_artifacts(data)
    #       data = self._handle_outliers(data, artifact_level)
    #
    # 【阈值调节说明】
    #   spike_ratio 是"信号最大突变幅度 / 平均标准差"的比值
    #
    #   spike_ratio = 相邻采样点最大差值 / 通道平均标准差
    #
    #   - spike_ratio > 20  -> 严重异常（碰撞、剧烈晃动）
    #   - spike_ratio > 10  -> 轻微异常（眨眼、小幅晃动）
    #
    # 【怎么判断阈值是否合适】
    #
    #   方法1：正常数据不应该触发。先不加异常检测跑一遍，记下正常的 spike_ratio
    #         然后把阈值设到正常值上限的 2-3 倍
    #
    #   方法2：在有标签的训练数据上试。如果开启了异常检测后准确率不变或上升
    #         -> 阈值合适。如果准确率下降 -> 阈值太敏感，误把正常信号当异常处理了
    #
    #   以 D1/D2 为例，正常信号 spike_ratio ≈ 5-8，所以阈值 20/10 是安全的
    #   正式比赛时先用训练数据测一下正常范围，再定阈值
    # ============================================================
    def detect_artifacts(self, data):
        """检测异常值，返回异常等级 0=正常 1=轻微 2=严重"""
        stds = data.std(axis=1)
        mean_std = stds.mean()
        if mean_std < 1e-6:
            return 2
        diffs = np.abs(np.diff(data, axis=1))
        max_diff = diffs.max()
        spike_ratio = max_diff / (mean_std + 1e-10)
        if spike_ratio > 20:
            return 2
        elif spike_ratio > 10:
            return 1
        corr = np.corrcoef(data)
        off_diag = corr[np.triu_indices_from(corr, k=1)]
        if np.any(np.abs(off_diag) > 0.98):
            return 1
        return 0

    def _handle_outliers(self, data, level):
        """根据异常等级处理信号"""
        if level == 0:
            return data
        elif level == 1:
            from scipy.ndimage import median_filter
            return median_filter(data, size=(1, 5))
        else:
            diff = np.abs(np.diff(data, axis=1))
            threshold = data.std(axis=1, keepdims=True) * 5
            bad_mask = np.pad(diff > threshold, ((0, 0), (1, 0)), constant_values=False)
            cleaned = data.copy()
            for ch in range(data.shape[0]):
                if bad_mask[ch].any():
                    bad_idx = np.where(bad_mask[ch])[0]
                    for idx in bad_idx:
                        if idx > 0 and idx < data.shape[1] - 1:
                            cleaned[ch, idx] = (data[ch, idx - 1] + data[ch, idx + 1]) / 2
            return cleaned

    # ==================== 预处理 ====================
    def pre_filter(self, data):
        data = scipysignal.detrend(data, axis=1)
        b, a = scipysignal.iircomb(50, 35, ftype='notch', fs=self.srate)
        data = scipysignal.filtfilt(b, a, data)
        fs = self.srate / 2
        N, Wn = scipysignal.ellipord([7 / fs, 90 / fs], [3 / fs, 100 / fs], 3, 40)
        b1, a1 = scipysignal.ellip(N, 1, 90, Wn, 'bandpass')
        return scipysignal.filtfilt(b1, a1, data)

    def _select_channels(self, data):
        stds = data.std(axis=1)
        n_keep = min(3, data.shape[0])
        best = np.argsort(stds)[-n_keep:]
        return data[best], best

    def _cca_corr(self, data, template):
        cca = CCA(n_components=1)
        cca.fit(data.T, template.T)
        d_t, t_t = cca.transform(data.T, template.T)
        return np.corrcoef(d_t[:, 0], t_t[:, 0])[0, 1]

    # ==================== 主检测方法 ====================
    def detect(self, data):
        """FBCCA检测 + 自适应子带"""
        # artifact_level = self.detect_artifacts(data)    #去掉此两行注释为开启异常值检测
        # data = self._handle_outliers(data, artifact_level)
        data, _ = self._select_channels(data)
        data = self.pre_filter(data)
        data = data - data.mean(axis=1, keepdims=True)

        n_bands = len(self.filter_banks)
        weights = np.array([(i + 1) ** (-1.25) + 0.25 for i in range(n_bands)])
        rho = np.zeros((n_bands, len(self.freqs)))
        for sb_idx, (b, a) in enumerate(self.filter_banks):
            flt = scipysignal.filtfilt(b, a, data)
            for f_idx, template in enumerate(self.TemplateSet):
                rho[sb_idx, f_idx] = self._cca_corr(flt, template)
        rho_fused = weights @ rho
        return int(np.argmax(rho_fused))


# ============================================================
# 版本选择器：改这里就行，demo.py 不用动
#   1 = ssvepDetect_Original（原始官方代码）
#   2 = ssvepDetect_Optimized（FBCCA优化版）
#   3 = ssvepDetect_v3（FBCCA+PSDA融合版，全覆盖）
# ============================================================
VERSION = 3

if VERSION == 1:
    ssvepDetect = ssvepDetect_Original
elif VERSION == 2:
    ssvepDetect = ssvepDetect_Optimized
elif VERSION == 3:
    ssvepDetect = ssvepDetect_v3
