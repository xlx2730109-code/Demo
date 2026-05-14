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
# 版本选择器：改这里就行，demo.py 不用动
#   1 = ssvepDetect_Original（原始官方代码）
#   2 = ssvepDetect_Optimized（FBCCA优化版，推荐）
# ============================================================
VERSION = 2

if VERSION == 1:
    ssvepDetect = ssvepDetect_Original
elif VERSION == 2:
    ssvepDetect = ssvepDetect_Optimized
