import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class DifferentiablePhysioExtractor(nn.Module):
    # 基于全序列 1D RFFT + 自适应汉宁窗 + 频谱去底噪
    # 彻底解决短窗 STFT 导致的低频(呼吸)泄漏，以及平坦底噪造成的心率期望值偏移
    def __init__(self, fs=30.0):
        super().__init__()
        self.fs = fs

    def forward(self, bvp_signals, freq_range=(0.7, 3.0), dynamic_fs=None):
        fs = float(dynamic_fs) if dynamic_fs is not None else self.fs
        seq_len = bvp_signals.shape[-1]

        # 1. 信号去均值与归一化 (彻底消除直流基线漂移)
        bvp_signals = bvp_signals - bvp_signals.mean(dim=-1, keepdim=True)
        bvp_signals = bvp_signals / (bvp_signals.std(dim=-1, keepdim=True) + 1e-8)

        # 2. 全序列自适应汉宁窗 (强制首尾归零，消除信号强行截断带来的宽带频率泄漏)
        window = torch.hann_window(seq_len).to(device=bvp_signals.device, dtype=bvp_signals.dtype)
        bvp_signals = bvp_signals * window

        # 3. 高分辨率 1D 傅里叶变换 (动态补零至少1024点)
        # 将原先极差的频率分辨率提升到 0.029Hz (~1.7 BPM) 的工业级精度
        n_fft_dynamic = max(1024, 1 << math.ceil(math.log2(seq_len)))
        
        fft_out = torch.fft.rfft(bvp_signals, n=n_fft_dynamic, dim=-1)
        power_spectrum = torch.abs(fft_out) ** 2  # [Batch, Freq_Bins]

        # 计算高精度的对应频率轴
        freqs = torch.linspace(
            0, fs / 2, n_fft_dynamic // 2 + 1, device=bvp_signals.device, dtype=power_spectrum.dtype
        )

        # 4. 提取目标生理频段
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if not torch.any(mask):
            return torch.full(
                (bvp_signals.shape[0],), 0.0, device=bvp_signals.device, dtype=power_spectrum.dtype
            )

        valid_freqs = freqs[mask]
        power_masked = power_spectrum[:, mask]

        # 5. 频谱去底噪 (核心破局点：一刀切掉平均底噪，防止算法将期望值拖拽向 110bpm 的频段中点)
        noise_floor = power_masked.mean(dim=-1, keepdim=True)
        power_masked = F.relu(power_masked - noise_floor)

        # 6. 计算频率重心 (恢复极度锐利的 temperature=0.05，像手术刀一样精准剥离出最高的真实波峰)
        temperature = 0.05
        power_norm = power_masked / (power_masked.max(dim=-1, keepdim=True)[0] + 1e-8)
        weights = F.softmax(power_norm / temperature, dim=-1)
        pred_hr_hz = torch.sum(weights * valid_freqs.unsqueeze(0), dim=-1)

        return pred_hr_hz * 60.0


class KalmanFilter1D:
    # 卡尔曼滤波器，用于平滑推理时的输出
    def __init__(self, process_variance=1e-3, measurement_variance=1e-1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_measurement = 0.0
        self.posteri_error_estimate = 1.0
        self.is_initialized = False

    def update(self, measurement):
        if not self.is_initialized:
            self.estimated_measurement = measurement
            self.is_initialized = True
            return self.estimated_measurement

        priori_estimate = self.estimated_measurement
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (
            priori_error_estimate + self.measurement_variance
        )
        self.estimated_measurement = priori_estimate + blending_factor * (
            measurement - priori_estimate
        )
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.estimated_measurement


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except (RuntimeError, pickle.UnpicklingError, EOFError, ValueError):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)