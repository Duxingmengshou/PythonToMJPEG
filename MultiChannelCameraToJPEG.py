from twisted.internet import reactor
from twisted.web import server, resource
from ultralytics import YOLO
import cv2
import threading

from Toolkits.ShutdownYOLOLogger import ShutdownYOLOLogger

ShutdownYOLOLogger()

isUseModel = True
# isUseModel = False


class ImageProcesCore:
    def __init__(self, ModelPath, VisualType=1):
        self.Model = YOLO(ModelPath, verbose=False)
        self.VisualType = VisualType

    def InferFrame(self, frame):
        if isUseModel:
            results = self.Model.predict(frame)
            self.AnalyzeResults(results)
            return results[0].plot()
        else:
            return frame

    def AnalyzeResults(self, results):
        if self.VisualType == 1:
            for result in results:
                modified_data = [row[:-2] + row[-1:]
                                 for row in result.boxes.data.numpy().tolist()]
                int_data = [[int(value) for value in sublist]
                            for sublist in modified_data]
                # 此处可以存储到DB


class MJPEGStreamCore(resource.Resource):
    isLeaf = True

    def __init__(self, CameraList):
        super().__init__()
        self.Frames = {}

        self.CameraList = CameraList
        self.CameraAbleList = []
        self.InitializeCameras()

    def InitializeCameras(self):
        for CameraIndex in self.CameraList:
            cap = cv2.VideoCapture(CameraIndex)
            if not cap.isOpened():
                print(f"Failed to open camera {CameraIndex}")
                continue
            self.CameraAbleList.append(CameraIndex)
            self.Frames[CameraIndex] = None
            # 启动一个线程来持续从摄像头读取帧
            threading.Thread(target=self.CaptureFrame, args=(
                CameraIndex, cap), daemon=True).start()

    def CaptureFrame(self, CameraIndex, Cap):
        if CameraIndex == 0:
            model = ImageProcesCore("./Models/yolo11n.pt", 1)
            pass
        elif CameraIndex == 1:
            # model = ImageProcesCore("./models/RestaurantSimulatorYOLOn.pt", 2)
            model = None
            pass
        else:
            model = None
        while True:
            ret, frame = Cap.read()
            if ret:
                if model is not None:
                    frame = model.InferFrame(frame)
                self.Frames[CameraIndex] = frame
            else:
                print(f"Camera {CameraIndex} failed to grab frame.")
                break

    def render_GET(self, request):
        request.setHeader(b'Access-Control-Allow-Origin', b'*')
        request.setHeader(b'Access-Control-Allow-Methods', b'GET, OPTIONS')
        request.setHeader(b'Access-Control-Allow-Headers', b'Content-Type')
        try:
            CameraIndex = int(request.path.decode().strip('/'))
        except ValueError:
            devices = "<p>Devices</p>"
            for device in self.Frames.keys():
                devices += f"<br><a href=\"http://127.0.0.1:18080/{device}\">device:{device}</a></br>"
            return devices.encode()

        if CameraIndex in self.CameraAbleList:
            request.setHeader(
                b'Content-Type', b'multipart/x-mixed-replace; boundary=frame')
            self.WriteFrame(request, CameraIndex)
            return server.NOT_DONE_YET
        else:
            devices = "<p>Devices</p>"
            for device in self.Frames.keys():
                devices += f"<br><a href=\"http://127.0.0.1:18080/{device}\">device:{device}</a></br>"
            return devices.encode()

    def WriteFrame(self, request, CameraIndex):
        if CameraIndex in self.Frames and self.Frames[CameraIndex] is not None:
            _, jpeg = cv2.imencode('.jpg', self.Frames[CameraIndex])
            frame = jpeg.tobytes()
            request.write(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
        reactor.callLater(0.04, self.WriteFrame, request,
                          CameraIndex)  # 每 100ms 发送一帧


if __name__ == "__main__":
    camera_ids = [0, 1, 2]
    stream = MJPEGStreamCore(camera_ids)
    site = server.Site(stream)
    reactor.listenTCP(18080, site)
    print(f"服务启动！Listening on port 18080.")
    reactor.run()
