from twisted.internet import reactor
from twisted.web import server, resource
from ultralytics import YOLO
import cv2
import threading


from Toolkits.ShutdownYOLOLogger import ShutdownYOLOLogger

ShutdownYOLOLogger()


class VideoStreamResource(resource.Resource):
    isLeaf = True

    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.yolo_model = YOLO('./Models/yolo11n.pt')  # 使用合适的 YOLO 模型
        self.capture_thread = threading.Thread(target=self.capture_video)
        self.capture_thread.start()

    def capture_video(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print("无法打开视频流")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取帧")
                break

            # 使用 YOLO 进行推理
            results = self.yolo_model(frame)
            # 处理结果（可以在这里绘制边界框等）
            frame = self.process_results(frame, results)

            # 将帧编码为 JPEG 格式
            _, jpeg = cv2.imencode('.jpg', frame)
            self.frame = jpeg.tobytes()

        cap.release()

    def process_results(self, frame, results):
        # 在这里处理 YOLO 结果，例如绘制边界框
        for result in results:
            boxes = result.boxes.xyxy  # 获取边界框
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def render(self, request):
        request.setHeader(b'Access-Control-Allow-Origin', b'*')
        request.setHeader(b'Access-Control-Allow-Methods', b'GET, OPTIONS')
        request.setHeader(b'Access-Control-Allow-Headers', b'Content-Type')
        request.setHeader(
            b'Content-Type', b'multipart/x-mixed-replace; boundary=frame')
        if hasattr(self, 'frame'):
            request.write(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + self.frame + b'\r\n'
            )
        else:
            request.write(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n'
            )
        return server.NOT_DONE_YET


def main():
    # rtsp_url = 'rtsp://192.168.43.112:8554/live/stream'  # 替换为你的 RTSP 流地址
    # rtsp_url = '0'  # 替换为你的 RTSP 流地址
    rtsp_url=0
    site = server.Site(VideoStreamResource(rtsp_url))
    reactor.listenTCP(8221, site)  # 在8080端口启动服务器
    print("服务器运行在 http://localhost:8221")
    reactor.run()


if __name__ == '__main__':
    main()
