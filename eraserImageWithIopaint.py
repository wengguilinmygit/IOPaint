import os
import sys
import shutil
from PIL import Image

def batch_remove_watermark(input_folder, mask_folder, mask_vertical_folder,output_folder):


    # 遍历 input_folder 中所有子目录及文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if not file.lower().endswith(('png', 'jpg', 'jpeg')):
                continue

            img_path = os.path.join(root, file)
            with Image.open(img_path) as img:
                width, height = img.size

            if height > width:
                selected_mask_folder = mask_vertical_folder
                print(f"📐 {img_path} 为纵向图，使用 mask_vertical_folder")
            else:
                selected_mask_folder = mask_folder
                print(f"📐 {img_path} 为横向图，使用 mask_folder")
            
            # 从 masks 文件夹中随机选择一个 mask 文件
            mask_files = [f for f in os.listdir(selected_mask_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if not mask_files:
                print("未在 masks 文件夹中找到任何 mask 文件，退出。")
                return
            selected_mask = os.path.join(selected_mask_folder, mask_files[0])

            target_mask_path = os.path.join(selected_mask_folder, file)

            # 避免 SameFileError
            if os.path.abspath(selected_mask) != os.path.abspath(target_mask_path):
                os.rename(selected_mask, target_mask_path)
            else:
                print(f"跳过重命名，mask 已经是目标文件名: {target_mask_path}")

            # 生成与 input_folder 相对路径结构的输出目录
            relative_root = os.path.relpath(root, input_folder)
            output_subdir = os.path.join(output_folder, relative_root)
            os.makedirs(output_subdir, exist_ok=True)

            # 执行 iopaint CLI 命令
            cmd = (
                f"/Library/Frameworks/Python.framework/Versions/3.11/bin/iopaint run --model=lama --device=cpu "
                f"--image=\"{img_path}\" "
                f"--mask=\"{target_mask_path}\" "
                f"--output=\"{output_subdir}\""
            )
            print(f"执行命令: {cmd}")
            os.system(cmd)

            # 使用完后重命名回去以便循环使用
            os.rename(target_mask_path, selected_mask)

    print("批量去水印处理完成。")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("用法: python eraserImageWithIopaint.py <input_folder> <mask_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    mask_folder = sys.argv[2]
    mask_vertical_folder = sys.argv[3]
    output_folder = sys.argv[4]

    batch_remove_watermark(input_folder, mask_folder,mask_vertical_folder, output_folder)
