import os
import sys
import shutil
from PIL import Image

def batch_remove_watermark(input_folder, mask_folder, mask_vertical_folder,masks_Horizontal_folder,output_folder):


    # 遍历 input_folder 中所有子目录及文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if not file.lower().endswith(('png', 'jpg', 'jpeg')):
                continue

            img_path = os.path.join(root, file)
            with Image.open(img_path) as img:
                width, height = img.size

            if (height > width):
                selected_mask_folder = mask_vertical_folder
                print(f"📐 {img_path} 为纵向图，使用 mask_vertical_folder")
            elif (height < width):
                selected_mask_folder = masks_Horizontal_folder
                print(f"📐 {img_path} 为横向向图，使用 masks_Horizontal_folder")
            else:
                selected_mask_folder = mask_folder
                print(f"📐 {img_path} 为正方形图，使用 mask_folder")
            
            # 从 masks 文件夹中随机选择一个 mask 文件
            mask_files = [f for f in os.listdir(selected_mask_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if not mask_files:
                print("未在 masks 文件夹中找到任何 mask 文件，退出。")
                return
            selected_mask = os.path.join(selected_mask_folder, mask_files[0])

            target_mask_path = os.path.join(selected_mask_folder, file)

            # 避免 SameFileError，复制mask文件到目标名
            if os.path.abspath(selected_mask) != os.path.abspath(target_mask_path):
                shutil.copy(selected_mask, target_mask_path)
                # 复制完成后，将 mask 的分辨率调整为与源图像一致
                try:
                    with Image.open(target_mask_path) as mask_img:
                        # 使用 NEAREST 保持掩码边界清晰（避免模糊）
                        resized_mask = mask_img.resize((width, height), resample=Image.NEAREST)
                        resized_mask.save(target_mask_path)
                        print(f"已将 mask 调整为 {width}x{height}: {target_mask_path}")
                except Exception as e:
                    print(f"调整 mask 大小失败 ({target_mask_path}): {e}")
            else:
                print(f"跳过复制，mask 已经是目标文件名: {target_mask_path}")

            # 生成与 input_folder 相对路径结构的输出目录
            relative_root = os.path.relpath(root, input_folder)
            output_subdir = os.path.join(output_folder, relative_root)
            os.makedirs(output_subdir, exist_ok=True)

            # 执行 iopaint CLI 命令
            cmd = (
                f"export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH && /opt/anaconda3/envs/myvideo/bin/iopaint run --model=lama --device=cpu "
                # f"/opt/anaconda3/envs/myvideo/bin/iopaint run --model=lama --device=cpu "
                f"--image=\"{img_path}\" "
                f"--mask=\"{target_mask_path}\" "
                f"--output=\"{output_subdir}\""
            )
            print(f"执行命令: {cmd}")
            os.system(cmd)

            # 使用完后删除临时mask文件
            if os.path.exists(target_mask_path):
                os.remove(target_mask_path)

    print("批量去水印处理完成。")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("用法: python eraserImageWithIopaint.py <input_folder> <mask_folder> <output_folder>")
        sys.exit(1)
    input_folder = sys.argv[1]
    mask_folder = sys.argv[2]
    mask_vertical_folder = sys.argv[3]
    masks_Horizontal_folder = sys.argv[4]
    output_folder = sys.argv[5]

    # input_folder = "/Users/ringring/Downloads/weng/GoogleDrive/inputTmp"
    # mask_folder = "/Users/ringring/Downloads/weng/GoogleDrive/masks"
    # mask_vertical_folder = "/Users/ringring/Downloads/weng/GoogleDrive/masks_vertical"
    # masks_Horizontal_folder = "/Users/ringring/Downloads/weng/GoogleDrive/masks_Horizontal"
    # output_folder = "/Users/ringring/Downloads/weng/GoogleDrive/input"
    # processed_folder = "/Users/ringring/Downloads/weng/GoogleDrive/processed"

    batch_remove_watermark(input_folder, mask_folder,mask_vertical_folder,masks_Horizontal_folder, output_folder)
