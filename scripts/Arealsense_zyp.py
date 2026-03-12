import os  # 用于文件路径操作
import numpy as np  # 用于数值计算
import pyrealsense2 as rs  # 用于Intel RealSense相机数据采集
import cv2  # 用于图像处理和显示


def input1():
    """
    使用Intel RealSense相机采集彩色、深度图像，并生成mask（中间80%为1，其他为255）
    操作：按's'保存当前帧及mask；按'q'退出
    """
    # 创建相机管道
    print("当前工作目录：", os.getcwd())  # 打印当前运行脚本的目录

    pipeline = rs.pipeline()

    # 配置相机流（分辨率、格式、帧率）
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 深度流：640x480，16位深度值，30fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 彩色流：640x480，BGR格式，30fps

    # 启动相机
    pipeline.start(config)
    align = rs.align(rs.stream.color)  # 创建深度-彩色对齐器（确保深度图与彩色图像素对齐）

    try:
        while True:
            # 等待一组同步帧（彩色+深度）
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)  # 对齐深度帧到彩色帧

            if not aligned_frames:
                continue  # 对齐失败则跳过

            # 提取对齐后的帧
            color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not aligned_depth_frame:
                continue  # 帧获取失败则跳过

            # 转换为numpy数组
            aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图像（单位：毫米）
            color_image = np.asanyarray(color_frame.get_data())  # 彩色图像（BGR格式）

            # 显示深度图（伪彩色）和原始深度图
            aligned_depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(aligned_depth_image, alpha=0.03),  # 缩放深度值以适应显示
                cv2.COLORMAP_JET
            )
            cv2.imshow("Aligned Depth colormap", aligned_depth_colormap)  # 显示伪彩色深度图
            cv2.imshow("Aligned Depth Image", aligned_depth_image)  # 显示原始深度图

            # 显示彩色图像
            cv2.imshow("Color Image", color_image)

            # 按键处理
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):  # 按's'保存当前帧及mask到data1目录
                # 确保data1目录存在
                save_dir = 'data1'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 保存深度图和彩色图
                cv2.imwrite(os.path.join(save_dir, 'depth.png'), aligned_depth_image)
                cv2.imwrite(os.path.join(save_dir, 'color.png'), color_image)

                # 生成mask：中间80%区域为1，其他区域为255
                height, width = color_image.shape[:2]
                # 计算中间80%区域的边界（宽高各留10%作为边缘）
                width_start = int(width * 0.1)  # 左边界（宽度的10%处）
                width_end = int(width * 0.9)    # 右边界（宽度的90%处）
                height_start = int(height * 0.1)  # 上边界（高度的10%处）
                height_end = int(height * 0.9)    # 下边界（高度的90%处）

                # 初始化mask为全0
                mask = np.ones((height, width), dtype=np.uint8) * 0
                # 将中间80%区域设为1
                mask[height_start:height_end, width_start:width_end] = 1

                cv2.imwrite(os.path.join(save_dir, 'segmentation_mask.png'), mask)

                print(f"数据已保存到{save_dir}目录：")
                print(f"- 深度图：{os.path.join(save_dir, 'depth.png')}")
                print(f"- 彩色图：{os.path.join(save_dir, 'color.png')}")
                print(f"- mask：{os.path.join(save_dir, 'segmentation_mask.png')}（中间80%区域为1，其他为255）")
                break
            elif key & 0xFF == ord('q'):  # 按'q'退出程序
                break

    finally:
        # 停止相机并关闭所有窗口
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    input1()  # 采集相机数据
    data_dir = 'data1'  # 数据目录
