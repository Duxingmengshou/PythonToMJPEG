import json
from twisted.internet import reactor, protocol
from twisted.web import server, resource
from twisted.internet import reactor
import time
import cv2
import numpy as np
import uuid
from ultralytics import YOLO

from Toolkits.ShutdownYOLOLogger import ShutdownYOLOLogger

ShutdownYOLOLogger()

# Load a model
# isUseModel = True
isUseModel = False
model = YOLO("./Models/yolov8n.pt", verbose=False)  # pretrained YOLOv8n model


class CoordinateTools():
    def __init__(self):
        pass

    def CenterPointJudgment(self, centerX, centerY):
        if centerX < 0:
            centerX = 0
        if centerX > 1270:
            centerX = 1270
        if centerY < 0:
            centerY = 0
        if centerY > 710:
            centerY = 710
        return (centerX, centerY)

    def CategoryColor(self, classID):
        color = (255, 255, 0)
        if classID == 0:
            color = (100, 255, 30)
        if classID == 1:
            color = (255, 255, 0)
        if classID == 2:
            color = (255, 0, 255)
        if classID == 3:
            color = (0, 255, 255)
        if classID == 4:
            color = (0, 255, 0)
        if classID == 5:
            color = (0, 0, 255)
        return color


class FrameDataServer(protocol.Protocol):
    def __init__(self, HttpServer, ClientID):
        self.HttpServer = HttpServer
        self.ClientID = ClientID
        self.Buffer = b''
        self.DepthRaw = None
        self.StartTime = time.time()
        self.EndTime = time.time()
        self.FrameCount = 0
        self.CoordinateTool = CoordinateTools()
        self.PersonInfo = []

    def dataReceived(self, data):
        self.Buffer += data
        while True:
            if len(self.Buffer) < 4:
                break
            length = int.from_bytes(self.Buffer[:4], byteorder='big')
            if len(self.Buffer) < 4 + length:
                break
            frame_data = self.Buffer[4:4 + length]
            self.Buffer = self.Buffer[4 + length:]

            frame_type = frame_data[0]
            frame_data = frame_data[1:]

            if frame_type == 1:
                self.EndTime = time.time()
                self.FrameCount += 1
                fps = self.FrameCount / (self.EndTime - self.StartTime)
                color_img = cv2.imdecode(np.frombuffer(
                    frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if isUseModel:
                    results = model(color_img)
                    # 打印每个标注框的坐标信息、类别和置信度，并在图像上绘制
                    self.PersonInfo = []
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0]  # 获取框的坐标
                        center_x = int((x1 + x2) / 2)  # 计算中心点的 x 坐标
                        center_y = int((y1 + y2) / 2)  # 计算中心点的 y 坐标
                        class_id = int(box.cls[0])  # 获取类别 ID
                        confidence = box.conf[0]  # 获取置信度
                        if confidence < 0.5:
                            continue

                        # 绘制标注框
                        color = self.CoordinateTool.CategoryColor(class_id)
                        cv2.rectangle(color_img, (int(x1), int(y1)),
                                      (int(x2), int(y2)), color, 1)

                        # 绘制中心点的十字线
                        cv2.line(color_img, (center_x - 5, center_y),
                                 (center_x + 5, center_y), color, 1)
                        cv2.line(color_img, (center_x, center_y - 5),
                                 (center_x, center_y + 5), color, 1)
                        center_x, center_y = self.CoordinateTool.CenterPointJudgment(
                            center_x, center_y)

                        # 将识别信息存储到字典中
                        self.PersonInfo.append({
                            'class_id': class_id,
                            'confidence': f"{confidence:.2f}",
                            'center': (center_x, center_y)
                        })

                    cv2.putText(color_img,
                                f"fps: {fps:.1f}",
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 0), 1,
                                cv2.LINE_AA)

                self.HttpServer.PersonInfo[self.ClientID] = self.PersonInfo
                self.HttpServer.Frames[self.ClientID] = color_img

    def connectionLost(self, reason):
        print(f"Connection lost: {reason.getErrorMessage()}")
        if self.ClientID in self.HttpServer.Frames:
            del self.HttpServer.Frames[self.ClientID]


class FrameDataServerFactory(protocol.Factory):
    def __init__(self, http_server):
        self.http_server = http_server

    def buildProtocol(self, addr):
        client_id = str('check')  # Generate a unique client ID
        print(f"New connection with ID: {client_id}")
        return FrameDataServer(self.http_server, client_id)


class MJPEGStream(resource.Resource):
    isLeaf = True

    def __init__(self):
        super().__init__()
        self.Frames = {}
        self.PersonInfo = {}

    def render_GET(self, request):
        # 添加 CORS 头部
        request.setHeader(b'Access-Control-Allow-Origin', b'*')
        request.setHeader(b'Access-Control-Allow-Methods', b'GET, OPTIONS')
        request.setHeader(b'Access-Control-Allow-Headers', b'Content-Type')
        clientID = request.path.decode().strip('/').split('/')[0]
        action = request.path.decode().strip('/').split('/')[1] if len(
            request.path.decode().strip('/').split('/')) > 1 else ''

        if clientID in self.Frames:
            if action == 'video':
                request.setHeader(
                    b'Content-Type', b'multipart/x-mixed-replace; boundary=frame')
                self._write_frame(request, clientID)
                return server.NOT_DONE_YET
            elif action == 'info':
                # 返回 JSON 格式的 PersonInfo
                if clientID in self.PersonInfo:
                    print(self.PersonInfo[clientID])
                    request.setHeader(b'Content-Type', b'application/json')
                    return json.dumps(self.PersonInfo[clientID], ensure_ascii=False).encode('utf-8')
                else:
                    request.setResponseCode(404)
                    return b'No information found for this client'
        else:
            request.setResponseCode(404)
            return b'Client not found'

    def _write_frame(self, request, client_id):
        if client_id in self.Frames:
            _, jpeg = cv2.imencode('.jpg', self.Frames[client_id])
            frame = jpeg.tobytes()

            request.write(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )

        # Adjust the frame rate as needed
        reactor.callLater(0.01, self._write_frame, request, client_id)

    def _update_person_info(self, client_id, person_info):
        self.PersonInfo[client_id] = person_info


if __name__ == "__main__":
    stream = MJPEGStream()

    site = server.Site(stream)
    reactor.listenTCP(8080, site)

    factory = FrameDataServerFactory(stream)
    reactor.listenTCP(8000, factory)

    print('服务启动成功！')

    reactor.run()
