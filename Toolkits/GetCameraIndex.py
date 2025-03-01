import cv2

'''
系统会自动给插入的摄像头（或虚拟摄像头）进行标号
用于测试摄像头的序号可用性（默认测试前十个摄像头序号）
'''
TEST_Camera_Count = 10


def test_cameras():
    available_cameras = []
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"摄像头索引 {index} 无法打开")
            continue
        print(f"摄像头索引 {index} 已打开")
        ret, frame = cap.read()
        if not ret:
            print(f"无法读取摄像头索引 {index} 的画面")
        else:
            print(f"成功读取摄像头索引 {index} 的画面")
            cv2.imshow(f'{index}', frame)
            cap.release()
            available_cameras.append(index)

    return available_cameras


def main():
    print("正在测试摄像头...")
    available_cameras = test_cameras()

    if not available_cameras:
        print("没有检测到可用的摄像头。")
    else:
        print(f"可用摄像头列表: {available_cameras}")

    return available_cameras


if __name__ == "__main__":
    available_cameras = main()
    cv2.waitKey(0)
