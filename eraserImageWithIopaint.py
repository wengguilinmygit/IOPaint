import os
import sys
import shutil

def batch_remove_watermark(input_folder, mask_folder, output_folder):
    # 从 masks 文件夹中随机选择一个 mask 文件
    mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    if not mask_files:
        print("未在 masks 文件夹中找到任何 mask 文件，退出。")
        return
    selected_mask = os.path.join(mask_folder, mask_files[0])

    # 遍历 tmp 中的所有图片，依次将 mask 重命名匹配，并执行 iopaint 命令
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    for img_name in image_files:
        target_mask_path = os.path.join(mask_folder, img_name)

        # 避免 SameFileError，如果 mask 文件名与目标相同则跳过重命名
        if os.path.abspath(selected_mask) != os.path.abspath(target_mask_path):
            os.rename(selected_mask, target_mask_path)
        else:
            print(f"跳过重命名，mask 已经是目标文件名: {target_mask_path}")

        # 执行 iopaint CLI 命令
        cmd = (
            f"iopaint run --model=lama --device=cpu "
            f"--image={os.path.join(input_folder, img_name)} "
            f"--mask={target_mask_path} "
            f"--output={output_folder}"
        )
        print(f"执行命令: {cmd}")
        os.system(cmd)

        # 使用完后重命名回去以便循环使用
        os.rename(target_mask_path, selected_mask)

    print("批量去水印处理完成。")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python eraserImageWithIopaint.py <input_folder> <mask_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    mask_folder = sys.argv[2]
    output_folder = sys.argv[3]

    batch_remove_watermark(input_folder, mask_folder, output_folder)
