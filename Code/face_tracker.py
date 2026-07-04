import cv2
import numpy as np
from config import INFER_CONFIG


class RobustFaceTracker:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"无法加载人脸检测模型: {cascade_path}")

        self.smoothed_box = None
        self.alpha = INFER_CONFIG.get("FACE_TRACKING_SMOOTH", 0.15)

        # 跳帧检测
        self.detect_interval = INFER_CONFIG.get("FACE_DETECT_INTERVAL", 1)
        self.frame_count = 0
        self.last_faces = None          # 存储上一次检测到的人脸列表

        # 肤色掩码开关
        self.enable_skin_mask = INFER_CONFIG.get("ENABLE_SKIN_MASK", True)
        # 调试叠加开关
        self.enable_debug = INFER_CONFIG.get("ENABLE_DEBUG_OVERLAY", False)

        # 动态肤色置信度历史（仅在掩码开启时使用）
        if self.enable_skin_mask:
            self.prev_coverages = [1.0] * 5
            self.history_valid_means = None
            self.skin_ema_alpha = 0.3
        else:
            self.prev_coverages = None
            self.history_valid_means = None

    @staticmethod
    def _compute_roi_skin_mask(roi_bgr):
        """基于局部统计和HSV的肤色掩码，返回掩码和有效像素均值"""
        h, w = roi_bgr.shape[:2]
        if h == 0 or w == 0:
            return None, None, 0.0

        bgr_f = roi_bgr.astype(np.float32).reshape(-1, 3)
        mean_rgb = bgr_f.mean(axis=0)
        std_rgb = bgr_f.std(axis=0) + 1e-6
        lower = mean_rgb - 2.0 * std_rgb
        upper = mean_rgb + 2.0 * std_rgb
        mask_rgb = np.all((bgr_f >= lower) & (bgr_f <= upper), axis=1)
        mask_rb = bgr_f[:, 2] > bgr_f[:, 0]   # BGR格式，R>B

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        sat = hsv[:, 1]
        val = hsv[:, 2]
        mask_hsv = (sat >= 30) & (val <= 240)

        final_mask = mask_rgb & mask_rb & mask_hsv
        coverage = np.mean(final_mask)
        if coverage < 0.05:
            return None, None, 0.0

        valid_mean = bgr_f[final_mask].mean(axis=0) if np.any(final_mask) else mean_rgb
        return final_mask.reshape(h, w).astype(np.uint8) * 255, valid_mean, coverage

    def process_frame(self, frame):
        h_img, w_img, _ = frame.shape

        # ---------- 人脸检测（支持跳帧）----------
        self.frame_count += 1
        do_detect = (self.detect_interval <= 1) or (self.frame_count % self.detect_interval == 1)

        if do_detect:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120)
            )
            self.last_faces = faces
        else:
            faces = self.last_faces

        if faces is None or len(faces) == 0:
            return None, frame

        # 选面积最大的人脸
        (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])

        # EMA 平滑人脸框
        if self.smoothed_box is None:
            self.smoothed_box = np.array([x, y, w, h], dtype=np.float32)
        else:
            self.smoothed_box = (
                self.alpha * np.array([x, y, w, h], dtype=np.float32)
                + (1 - self.alpha) * self.smoothed_box
            )

        sx, sy, sw, sh = [int(v) for v in self.smoothed_box]

        # ROI定义 (x1,y1,x2,y2)
        fh_c = (sx + int(sw * 0.30), sy + int(sh * 0.05), sx + int(sw * 0.70), sy + int(sh * 0.25))
        fh_l = (sx + int(sw * 0.10), sy + int(sh * 0.05), sx + int(sw * 0.30), sy + int(sh * 0.25))
        fh_r = (sx + int(sw * 0.70), sy + int(sh * 0.05), sx + int(sw * 0.90), sy + int(sh * 0.25))
        ch_l = (sx + int(sw * 0.10), sy + int(sh * 0.45), sx + int(sw * 0.40), sy + int(sh * 0.70))
        ch_r = (sx + int(sw * 0.60), sy + int(sh * 0.45), sx + int(sw * 0.90), sy + int(sh * 0.70))

        roi_boxes = [fh_c, fh_l, fh_r, ch_l, ch_r]

        def clip_coords(x1, y1, x2, y2):
            return (max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2))

        roi_boxes_clipped = [clip_coords(*box) for box in roi_boxes]

        if any(x2 <= x1 or y2 <= y1 for (x1, y1, x2, y2) in roi_boxes_clipped):
            return None, frame

        roi_means = []

        # 如果不启用调试叠加，就不复制帧，节省时间
        debug_frame = frame.copy() if self.enable_debug else frame

        for i, (x1, y1, x2, y2) in enumerate(roi_boxes_clipped):
            roi_img = frame[y1:y2, x1:x2]

            if self.enable_skin_mask:
                mask, valid_mean, coverage = self._compute_roi_skin_mask(roi_img)
                # 处理掩码面积骤降
                if mask is None:
                    if self.history_valid_means is not None:
                        fused_mean = self.history_valid_means[i]
                    else:
                        fused_mean = cv2.mean(roi_img)[:3]
                else:
                    prev_cov = self.prev_coverages[i]
                    if coverage < 0.5 and (prev_cov - coverage) > 0.3:
                        if self.history_valid_means is not None:
                            alpha_skin = 0.7
                            fused_mean = (alpha_skin * self.history_valid_means[i] +
                                          (1 - alpha_skin) * valid_mean)
                        else:
                            fused_mean = valid_mean
                    else:
                        fused_mean = valid_mean
                        if self.history_valid_means is None:
                            self.history_valid_means = np.zeros((5, 3), dtype=np.float32)
                        self.history_valid_means[i] = (self.skin_ema_alpha * valid_mean +
                                                       (1 - self.skin_ema_alpha) * self.history_valid_means[i])
                    self.prev_coverages[i] = coverage
                roi_means.append(fused_mean)

                # 调试叠加（仅在开启时绘制）
                if self.enable_debug and mask is not None:
                    mask_bgr = (cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * 0.5).astype(np.uint8)
                    debug_frame[y1:y2, x1:x2] = cv2.addWeighted(debug_frame[y1:y2, x1:x2], 0.5, mask_bgr, 0.5, 0)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # 不启用肤色掩码，直接取ROI均值
                fused_mean = cv2.mean(roi_img)[:3]
                roi_means.append(fused_mean)
                if self.enable_debug:
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 最后绘制人脸框
        if self.enable_debug:
            cv2.rectangle(debug_frame, (sx, sy), (sx + sw, sy + sh), (100, 100, 100), 1)

        return (
            roi_means[0], roi_means[1], roi_means[2], roi_means[3], roi_means[4]
        ), debug_frame