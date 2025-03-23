import cv2

# RTSP 流地址
rtsp_url = 'rtsp://192.168.43.112:8554/live/stream'

# 创建视频捕捉对象
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("无法打开视频流")
    exit()

while True:
    # 逐帧读取视频流
    ret, frame = cap.read()

    if not ret:
        print("无法读取帧")
        break

    # 显示当前帧
    cv2.imshow('RTSP Stream', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕捉对象
cap.release()
cv2.destroyAllWindows()
