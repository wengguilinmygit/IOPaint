import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
import torch
import hashlib

# IOPaint-related modules
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest, HDStrategy
from iopaint.plugins.interactive_seg import InteractiveSeg, SEGMENT_ANYTHING_MODELS

# --- 配置部分 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置路径 ---
input_folder = "/Users/ringring/Downloads/weng/GoogleDrive/tmp"
output_folder = "/Users/ringring/Downloads/weng/GoogleDrive/output"
mask_folder = "/Users/ringring/Downloads/weng/GoogleDrive/masks"
compare_folder = "/Users/ringring/Downloads/weng/GoogleDrive/compare"

# --- 自动水印检测 ---
def auto_detect_watermark(image_rgb: np.ndarray, use_mobile_sam: bool = True):
    if image_rgb is None:
        logging.error("输入图像为空。")
        return None

    mask = None
    if use_mobile_sam and seg_model is not None:
        try:
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            lower = np.array([0, 0, 180])
            upper = np.array([180, 50, 255])
            initial_mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    img_md5 = hashlib.md5(image_rgb.tobytes()).hexdigest()
                    clicks = [[cx, cy, 1]]
                    pred_mask = seg_model.forward(image_rgb, clicks, img_md5=img_md5)
                    if pred_mask is not None and np.any(pred_mask):
                        mask = pred_mask
                        logging.info("使用 MobileSAM 成功生成遮罩。")
        except Exception as e:
            logging.error(f"MobileSAM 分割失败: {e}", exc_info=True)

    if mask is None:
        logging.info("回退到 HSV 方法生成遮罩。")
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 0, 180])
        upper = np.array([180, 50, 255])
        hsv_mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logo_mask = np.zeros_like(hsv_mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                cv2.drawContours(logo_mask, [cnt], -1, 255, -1)
        if np.any(logo_mask):
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(logo_mask, kernel, iterations=2)
    return mask

# --- 保存遮罩 ---
def save_mask(mask, mask_save_path):
    if mask is not None:
        cv2.imwrite(mask_save_path, mask)
    else:
        logging.warning(f"遮罩为空，无法保存至: {mask_save_path}")

# --- 主执行函数 ---
def main():
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(compare_folder, exist_ok=True)

    logging.info(f"OpenCV version: {cv2.__version__}")
    logging.info(f"支持的分割模型: {list(SEGMENT_ANYTHING_MODELS.keys())}")

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    logging.info(f"使用的设备: {device}")

    global model, seg_model
    logging.info("正在加载 LaMa 模型 (no_half=True)...")
    model = ModelManager(name='lama', device=device, no_half=True)
    logging.info("LaMa 模型加载成功。")

    seg_model = None
    try:
        seg_model = InteractiveSeg(model_name="mobile_sam", device=device)
        logging.info("MobileSAM 模型加载成功。")
    except Exception as e:
        logging.warning(f"MobileSAM 不可用: {e}，使用 HSV 回退模式。")

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    processed_count = 0

    for file in tqdm(image_files, desc="批量去水印中"):
        input_path = os.path.join(input_folder, file)
        mask_path = os.path.join(mask_folder, file)
        output_path = os.path.join(output_folder, file)
        compare_path = os.path.join(compare_folder, file)

        original_bgr = cv2.imread(input_path)
        if original_bgr is None:
            logging.error(f"加载图像失败: {file}")
            continue
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        mask = auto_detect_watermark(original_rgb, use_mobile_sam=(seg_model is not None))
        if mask is None:
            logging.warning(f"跳过文件 {file}: 未检测到水印。")
            continue

        save_mask(mask, mask_path)

        config = InpaintRequest(
            hd_strategy=HDStrategy.CROP,
            hd_strategy_crop_margin=128,
            hd_strategy_crop_trigger_size=800,
            hd_strategy_resize_limit=1280,
        )

        try:
            result_rgb = model(original_rgb, mask, config)
            if result_rgb is None:
                logging.error(f"修复失败: {file}")
                continue
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)

            compare_image = np.hstack((original_bgr, result_bgr))
            cv2.imwrite(compare_path, compare_image)

            logging.info(f"成功修复: {file}")
            processed_count += 1
        except Exception as e:
            logging.error(f"处理 {file} 时出错: {e}", exc_info=True)
            continue

    logging.info(f"批量去水印完成，共处理 {processed_count} 张图像。")
    logging.info(f"修复结果保存在: {output_folder}")
    logging.info(f"对比图像保存在: {compare_folder}")

if __name__ == "__main__":
    main()