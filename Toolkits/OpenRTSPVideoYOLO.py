import cv2
from ultralytics import YOLO

from ShutdownYOLOLogger import ShutdownYOLOLogger

ShutdownYOLOLogger()

# RTSP 流地址
rtsp_url = 'rtsp://192.168.43.112:8554/live/stream'

# 创建视频捕捉对象
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


if not cap.isOpened():
    print("无法打开视频流")
    exit()

# 加载 YOLO 模型（请确保你有合适的 YOLO 模型文件）
model = YOLO('../Models/yolo11n.pt')  # 替换为你的 YOLO 模型路径

while True:
    # 逐帧读取视频流
    ret, frame = cap.read()

    if not ret:
        print("无法读取帧")
        break

    # 使用 YOLO 进行推理
    results = model(frame)

    # 处理 YOLO 结果并绘制边界框
    for result in results:
        boxes = result.boxes.xyxy  # 获取边界框
        confidences = result.boxes.conf  # 获取置信度
        class_ids = result.boxes.cls  # 获取类别 ID

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f'Class {int(class_id)}: {confidence:.2f}'  # 根据需要修改标签内容
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示当前帧
    cv2.imshow('RTSP Stream with YOLO Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放视频捕捉对象
cap.release()
cv2.destroyAllWindows()
