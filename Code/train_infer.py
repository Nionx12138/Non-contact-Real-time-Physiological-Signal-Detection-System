import os
import cv2
import time
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

from face_tracker import RobustFaceTracker
from titan_real_ssm import RealSSM                      # ← 实数模型
from utils import DifferentiablePhysioExtractor, KalmanFilter1D, safe_torch_load

try:
    from config import COMMON_CONFIG, INFER_CONFIG, PHYSIO_CONFIG
except ImportError:
    COMMON_CONFIG = {
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "SEQ_LEN": 300,
        "FS": 30.0,
        "D_MODEL": 64,
    }
    INFER_CONFIG = {
        "KALMAN_Q_HR": 0.01,
        "KALMAN_R_HR": 0.1,
        "KALMAN_Q_RR": 0.01,
        "KALMAN_R_RR": 0.1,
        "MODEL_PATH": "titan_real_ssm_best.pth",        # ← 使用实数权重
        "MAX_FACE_LOST_FRAMES": 30,
        "HR_RANGE": (40, 180),
        "RR_RANGE": (6, 30),
        "FPS_WINDOW": 30,
        "RR_MIN_SECONDS": 20.0,
    }
    PHYSIO_CONFIG = {"HR_BAND": (0.7, 3.0), "RR_BAND": (0.1, 0.5)}


def _infer_d_model_from_checkpoint(checkpoint, default_d_model):
    if not isinstance(checkpoint, dict):
        return int(default_d_model)
    ckpt_config = checkpoint.get("config", {})
    if isinstance(ckpt_config, dict) and "d_model" in ckpt_config:
        return int(ckpt_config["d_model"])
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if isinstance(state_dict, dict) and "fc_out.weight" in state_dict:
        fc_w = state_dict["fc_out.weight"]
        if hasattr(fc_w, "shape") and len(fc_w.shape) == 2 and fc_w.shape[1] % 2 == 0:
            return int(fc_w.shape[1] // 2)
    return int(default_d_model)


def run_realtime():
    requested_device = COMMON_CONFIG["DEVICE"] if torch.cuda.is_available() else "cpu"
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    DEVICE = torch.device(requested_device)

    model_path = INFER_CONFIG.get("MODEL_PATH", "titan_real_ssm_best.pth")
    if not os.path.isabs(model_path):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(project_dir, model_path)

    if not os.path.exists(model_path):
        print(f"❌ 警告: 找不到权重文件 {model_path}，请先执行 train.py 或运行 convert_weights.py")
        return

    checkpoint = safe_torch_load(model_path, map_location=DEVICE)
    ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    SEQ_LEN = int(ckpt_config.get("seq_len", COMMON_CONFIG.get("SEQ_LEN", 300)))
    TARGET_FPS = float(ckpt_config.get("fs", COMMON_CONFIG.get("FS", 30.0)))
    D_MODEL = _infer_d_model_from_checkpoint(checkpoint, COMMON_CONFIG.get("D_MODEL", 64))

    HR_BAND = PHYSIO_CONFIG.get("HR_BAND", (0.7, 3.0))
    RR_BAND = PHYSIO_CONFIG.get("RR_BAND", (0.1, 0.5))
    HR_RANGE = INFER_CONFIG.get("HR_RANGE", (40, 180))
    RR_RANGE = INFER_CONFIG.get("RR_RANGE", (6, 30))
    RR_BUFFER_LEN = int(TARGET_FPS * max(15, INFER_CONFIG.get("RR_MIN_SECONDS", 20.0)))
    FPS_WINDOW = int(INFER_CONFIG.get("FPS_WINDOW", 30))
    MAX_MISSED_FRAMES = int(INFER_CONFIG.get("MAX_FACE_LOST_FRAMES", 30))

    print(f"=== 🚀 启动 TitanMambaHR 实数推理引擎 (设备: {DEVICE}) ===")

    tracker = RobustFaceTracker()
    model = RealSSM(d_model=D_MODEL).to(DEVICE)
    model.eval()

    physio_extractor = DifferentiablePhysioExtractor(fs=TARGET_FPS).to(DEVICE)

    kalman_hr = KalmanFilter1D(
        process_variance=INFER_CONFIG.get("KALMAN_Q_HR", 0.01),
        measurement_variance=INFER_CONFIG.get("KALMAN_R_HR", 0.1),
    )
    kalman_rr = KalmanFilter1D(
        process_variance=INFER_CONFIG.get("KALMAN_Q_RR", 0.01),
        measurement_variance=INFER_CONFIG.get("KALMAN_R_RR", 0.1),
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    print(f"✅ 成功挂载实数模型: {model_path}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 摄像头打开失败，请检查摄像头是否被占用或设备是否可用。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    signal_buffer = deque(maxlen=SEQ_LEN)
    timestamp_buffer = deque(maxlen=SEQ_LEN)
    bvp_buffer = deque(maxlen=RR_BUFFER_LEN)
    fps_history = deque(maxlen=FPS_WINDOW)

    hr, rr = 0.0, 0.0
    last_valid_fused = np.array([127.0, 127.0, 127.0], dtype=np.float32)
    missed_frames = 0

    tick_freq = cv2.getTickFrequency()

    # 预热状态
    warmup_frames_collected = 0
    warmup_done = False
    warmup_template = []
    display_smooth_hr = 0.0
    display_smooth_rr = 0.0
    confidence = 0.0

    if not hasattr(run_realtime, 'real_frame_counter'):
        run_realtime.real_frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_tick = cv2.getTickCount()

        # 屏幕帧率显示
        current_time = time.time()
        time_diff = current_time - getattr(run_realtime, 'last_time', current_time)
        run_realtime.last_time = current_time
        if time_diff > 0.001:
            fps_history.append(1.0 / time_diff)
        avg_fps = sum(fps_history) / len(fps_history) if len(fps_history) > 0 else TARGET_FPS
        avg_fps = max(float(avg_fps), 1e-6)

        tracker_result = tracker.process_frame(frame)

        # 人脸丢失处理
        if tracker_result is None or tracker_result[0] is None:
            missed_frames += 1
            hr, rr = 0.0, 0.0
            if missed_frames > MAX_MISSED_FRAMES:
                cv2.putText(frame, "Resetting Context...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                signal_buffer.clear()
                timestamp_buffer.clear()
                bvp_buffer.clear()
                fps_history.clear()
                kalman_hr = KalmanFilter1D(
                    process_variance=INFER_CONFIG.get("KALMAN_Q_HR", 0.01),
                    measurement_variance=INFER_CONFIG.get("KALMAN_R_HR", 0.1),
                )
                kalman_rr = KalmanFilter1D(
                    process_variance=INFER_CONFIG.get("KALMAN_Q_RR", 0.01),
                    measurement_variance=INFER_CONFIG.get("KALMAN_R_RR", 0.1),
                )
                last_valid_fused = np.array([127.0, 127.0, 127.0], dtype=np.float32)
                missed_frames = 0
                warmup_done = False
                warmup_frames_collected = 0
                warmup_template.clear()
                run_realtime.real_frame_counter = 0
            else:
                cv2.putText(frame, "Hold Still...", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                signal_buffer.append(last_valid_fused)
                timestamp_buffer.append(current_tick)
            display_frame = frame
        else:
            missed_frames = 0
            roi_signals, display_frame = tracker_result
            fh_c, fh_l, fh_r, ch_l, ch_r = roi_signals

            fused_bgr = (
                0.4 * np.array(fh_c, dtype=np.float32) +
                0.1 * np.array(fh_l, dtype=np.float32) +
                0.1 * np.array(fh_r, dtype=np.float32) +
                0.2 * np.array(ch_l, dtype=np.float32) +
                0.2 * np.array(ch_r, dtype=np.float32)
            )
            last_valid_fused = fused_bgr

            # 预热：镜像延拓
            if not warmup_done:
                signal_buffer.append(fused_bgr)
                timestamp_buffer.append(current_tick)
                warmup_frames_collected += 1
                if warmup_frames_collected >= min(3, SEQ_LEN):
                    buffer_list = list(signal_buffer)
                    padded = buffer_list.copy()
                    while len(padded) < SEQ_LEN:
                        padded.extend(buffer_list[::-1])
                    padded = padded[:SEQ_LEN]
                    signal_buffer = deque(padded, maxlen=SEQ_LEN)

                    last_real_tick = timestamp_buffer[-1]
                    timestamp_buffer = deque(
                        list(timestamp_buffer) + [last_real_tick] * (SEQ_LEN - len(timestamp_buffer)),
                        maxlen=SEQ_LEN)
                    warmup_template = buffer_list.copy()
                    warmup_done = True
                    run_realtime.real_frame_counter = warmup_frames_collected
                else:
                    progress = int((len(signal_buffer) / SEQ_LEN) * 100)
                    cv2.putText(display_frame, f"Initializing... {progress}%", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                    cv2.putText(display_frame, f"FPS: {avg_fps:.1f}",
                                (10, display_frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("TitanMamba rPPG - Inference", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

            # 正常逐帧替换
            signal_buffer.append(fused_bgr)
            timestamp_buffer.append(current_tick)
            run_realtime.real_frame_counter = min(SEQ_LEN, run_realtime.real_frame_counter + 1)

            if len(signal_buffer) == SEQ_LEN:
                # 计算窗口真实帧率
                t_first = timestamp_buffer[0]
                t_last = timestamp_buffer[-1]
                elapsed = (t_last - t_first) / tick_freq
                real_fps = (SEQ_LEN - 1) / elapsed if elapsed > 0 else TARGET_FPS

                if run_realtime.real_frame_counter < SEQ_LEN:
                    dynamic_fs_hr = TARGET_FPS
                else:
                    dynamic_fs_hr = real_fps

                sig_array = np.array(signal_buffer, dtype=np.float32)
                channel_mean = np.mean(sig_array, axis=0)
                sig_array = sig_array / (channel_mean + 1e-8) - 1.0

                inp = torch.tensor(sig_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                # ========== 实数双通道输入 ==========
                x_real = inp[:, :, 1:2]                          # G 通道作为实部
                x_imag = inp[:, :, 0:1] - inp[:, :, 2:3]        # B - R 作为虚部
                # ================================

                with torch.inference_mode():
                    pred_bvp = model(x_real, x_imag)             # 实数模型推理
                    pred_bvp = pred_bvp.view(1, -1)

                    if abs(real_fps - TARGET_FPS) >= 1.0:
                        target_len = max(16,
                                         int(round(SEQ_LEN * (TARGET_FPS / max(real_fps, 1e-6)))))
                        uniform_bvp = F.interpolate(
                            pred_bvp.unsqueeze(1), size=target_len,
                            mode="linear", align_corners=False
                        ).squeeze(1)
                    else:
                        uniform_bvp = pred_bvp

                    current_bvp_val = uniform_bvp[0, -1].item()
                    bvp_buffer.append(current_bvp_val)

                    raw_hr = physio_extractor(
                        uniform_bvp, freq_range=HR_BAND, dynamic_fs=dynamic_fs_hr
                    )[0].item()
                    smoothed_hr = kalman_hr.update(raw_hr)
                    hr = np.clip(smoothed_hr, HR_RANGE[0], HR_RANGE[1])

                    # 呼吸率
                    min_rr_frames = int(TARGET_FPS * max(10.0, INFER_CONFIG.get("RR_MIN_SECONDS", 20.0)))
                    if len(bvp_buffer) >= min_rr_frames:
                        bvp_tensor = torch.tensor(list(bvp_buffer), dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        if abs(real_fps - TARGET_FPS) >= 1.0:
                            target_rr_len = max(16,
                                                int(round(len(bvp_buffer) * (TARGET_FPS / max(real_fps, 1e-6)))))
                            uniform_bvp_rr = F.interpolate(
                                bvp_tensor.unsqueeze(1), size=target_rr_len,
                                mode="linear", align_corners=False
                            ).squeeze(1)
                        else:
                            uniform_bvp_rr = bvp_tensor

                        raw_rr = physio_extractor(
                            uniform_bvp_rr, freq_range=RR_BAND, dynamic_fs=dynamic_fs_hr
                        )[0].item()
                        smoothed_rr = kalman_rr.update(raw_rr)
                        rr = np.clip(smoothed_rr, RR_RANGE[0], RR_RANGE[1])
                    else:
                        rr = 0.0

                # 置信度与平滑显示
                fill_ratio = run_realtime.real_frame_counter / SEQ_LEN
                confidence = min(1.0, 0.3 + 0.7 * fill_ratio)

                if display_smooth_hr == 0.0:
                    display_smooth_hr = hr
                    display_smooth_rr = rr
                else:
                    ema_alpha = 0.2
                    display_smooth_hr = ema_alpha * hr + (1 - ema_alpha) * display_smooth_hr
                    display_smooth_rr = ema_alpha * rr + (1 - ema_alpha) * display_smooth_rr

        # 界面绘制
        status_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
        status_text = "Tracking" if confidence > 0.8 else "Warming..."

        if hr > 0:
            cv2.putText(display_frame, f"{status_text} (conf: {confidence:.1f})",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, f"HR: {display_smooth_hr:.1f} BPM",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display_frame,
                        f"RR: {display_smooth_rr:.1f} RPM" if display_smooth_rr > 0 else "RR: Calculating...",
                        (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        else:
            progress = int((len(signal_buffer) / SEQ_LEN) * 100)
            cv2.putText(display_frame, f"Buffering... {progress}%",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}",
                    (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("TitanMamba rPPG - Inference", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()