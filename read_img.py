import os
import cv2
import numpy as np
from pathlib import Path
from natsort import natsorted


def parse_lasot_gt_file(gt_file, img_width, img_height):
    """解析LASOT数据集的GT文件"""
    bboxes = []

    if not os.path.exists(gt_file):
        print(f"警告: GT文件 '{gt_file}' 不存在，将不显示边界框")
        return bboxes

    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):  # 跳过空行和注释
                continue

            # LASOT格式: x,y,w,h (绝对坐标)
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    w = float(parts[2])
                    h = float(parts[3])

                    # 转换为xyxy格式
                    x1 = int(x)
                    y1 = int(y)
                    x2 = int(x + w)
                    y2 = int(y + h)

                    # 确保边界框坐标有效
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 >= img_width: x2 = img_width - 1
                    if y2 >= img_height: y2 = img_height - 1

                    # 类别ID默认为0（因为LASOT只有一个目标）
                    bboxes.append((x1, y1, x2, y2, 0))
                except ValueError:
                    print(f"警告: 无法解析边界框数据 '{line}'")

    return bboxes


def play_lasot_sequence(image_folder, gt_file, fps=30, image_extensions=('png', 'jpg', 'jpeg')):
    """
    播放LASOT数据集的图片序列，并显示Ground Truth边界框

    参数:
    image_folder (str): 包含图片序列的文件夹路径
    gt_file (str): 包含Ground Truth数据的TXT文件路径
    fps (int): 播放帧率，默认为30
    image_extensions (tuple): 支持的图片文件扩展名，默认为('png', 'jpg', 'jpeg')
    """
    # 获取所有图片文件并按自然顺序排序
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_folder).glob(f'*.{ext}'))

    if not image_files:
        raise ValueError(f"在文件夹 {image_folder} 中未找到支持的图片文件")

    # 使用自然排序确保正确的顺序
    image_files = natsorted(image_files)

    # 定义边界框颜色
    color = (0, 255, 0)  # 绿色

    # 计算每张图片的显示时间(毫秒)
    delay = int(1000 / fps)

    # 创建窗口
    cv2.namedWindow('LASOT Sequence Player', cv2.WINDOW_NORMAL)

    # 播放图片序列
    current_frame = 0
    total_frames = len(image_files)

    while current_frame < total_frames:
        image_path = image_files[current_frame]

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"警告: 无法读取图片 {image_path}，跳过")
            current_frame += 1
            continue

        img_height, img_width = img.shape[:2]

        # 解析当前帧的GT边界框
        frame_bboxes = parse_lasot_gt_file(gt_file, img_width, img_height)

        # 如果有多个边界框，选择对应帧的
        if len(frame_bboxes) > current_frame:
            bbox = frame_bboxes[current_frame]
            x1, y1, x2, y2, class_id = bbox

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = "Target"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 显示当前帧号和文件名
        info_text = f"Frame: {current_frame + 1}/{total_frames} - {os.path.basename(str(image_path))}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示图片
        cv2.imshow('LASOT Sequence Player', img)

        # 等待按键事件
        key = cv2.waitKey(delay) & 0xFF

        # 按ESC或q键退出播放
        if key == 27 or key == ord('q'):
            break
        # 按空格暂停/继续
        elif key == ord(' '):
            print("已暂停，按空格继续...")
            while True:
                pause_key = cv2.waitKey(0) & 0xFF
                if pause_key == ord(' '):
                    break
        # 按右箭头键下一帧
        elif key == 83:  # 右箭头键
            current_frame = min(current_frame + 1, total_frames - 1)
            print(f"跳转到帧 {current_frame + 1}/{total_frames}")
        # 按左箭头键上一帧
        elif key == 81:  # 左箭头键
            current_frame = max(current_frame - 1, 0)
            print(f"跳转到帧 {current_frame + 1}/{total_frames}")
        # 按数字键快速跳转到指定百分比的帧
        elif ord('0') <= key <= ord('9'):
            percent = (key - ord('0')) * 10
            current_frame = int(total_frames * percent / 100)
            print(f"跳转到 {percent}% 位置: 帧 {current_frame + 1}/{total_frames}")
        else:
            current_frame += 1

    # 关闭所有窗口
    cv2.destroyAllWindows()

    print(f"图片序列播放完成")
    print(f"播放帧率: {fps}fps")
    print(f"总帧数: {total_frames}")
    print(f"序列时长: {total_frames / fps:.2f}秒")


if __name__ == "__main__":
    # 直接在代码中设置路径和参数
    IMAGE_FOLDER = "./data/lasot/helmet/helmet-19/img"  # 替换为LASOT序列图片文件夹路径
    GT_FILE = "./data/lasot/helmet/helmet-19/groundtruth.txt"  # 替换为LASOT序列的GT文件路径
    FPS = 60  # 播放帧率

    # 检查路径是否存在
    if not os.path.exists(IMAGE_FOLDER):
        print(f"错误: 图片文件夹 '{IMAGE_FOLDER}' 不存在")
    elif not os.path.exists(GT_FILE):
        print(f"错误: GT文件 '{GT_FILE}' 不存在")
    else:
        try:
            play_lasot_sequence(IMAGE_FOLDER, GT_FILE, FPS)
        except Exception as e:
            print(f"错误: 处理过程中发生异常 - {str(e)}")