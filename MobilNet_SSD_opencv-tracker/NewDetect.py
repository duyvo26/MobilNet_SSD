import cv2
import imutils
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
from centroidtracker import CentroidTracker
import datetime

# Load mô hình MobileNetSSD
prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
Sizeou = 1200
blobSizepercent = 0.4
patch = 'http://duyvo26.xyz:5000/webapi/entry.cgi/TraOnTruocRong-20230418-135145-1681800705.mp4?api=SYNO.SurveillanceStation.Recording.ShareRecording&version=1&method=Download&evtHash="1Yi6dqEd0"'




net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Khởi tạo Centroid Tracker
tracker = CentroidTracker(maxDisappeared=35, maxDistance=50)


# Đọc video từ file hoặc camera
# nếu đọc từ camera, thay đổi đường dẫn thành số camera
vs = cv2.VideoCapture(patch)
# vs = cv2.VideoCapture('rtsp://syno:8485a383b59958a8f87902a4d6ebd124@192.168.1.37:554/Sms=14.unicast')

# Tính FPS
fps = FPS().start()

# Lặp qua các frame trong video
while True:

    # Đọc frame từ video
    ret, frame = vs.read()
    
    # Kiểm tra xem còn frame nào để đọc không
    if not ret:
        break

    # Điều chỉnh kích thước của frame để tăng tốc độ xử lý
    frame = imutils.resize(frame, width=Sizeou)

    # Lấy chiều cao và chiều rộng của frame
    (h, w) = frame.shape[:2]


    blobSize = int(min(w, h) * blobSizepercent)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (blobSize, blobSize)), 0.007843, (blobSize, blobSize), 127.5)

    # Truyền blob qua mạng và lấy các detections và predictions
    net.setInput(blob)
    detections = net.forward()

    # Xử lý các detections
    rects = []



    for i in np.arange(0, detections.shape[2]):
        confidence = round(detections[0, 0, i, 2], 2)
        # ty le vat
        if confidence > 0.70:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] not in ["person", "motorbike", "car"]:
                continue
            person_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = person_box.astype("int")
            person_box = [person_box[0], person_box[1], person_box[2], person_box[3], idx, int(confidence * 100)]
            rects.append(person_box)


    # Cập nhật tracker với các bounding box mới

    objects = tracker.update(rects)

    # Vẽ bounding box và đánh số đối tượng trên frame
    for (objectId, bbox) in objects.items():
        # Lấy bounding box của đối tượng từ tracker

        # Lấy tọa độ của bounding box
        startX, startY, endX, endY, ClasID, confidence = map(int, bbox)

        # Tính toán tâm của bounding box
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)


        # Vẽ bounding box và đánh số đối tượng trên frame
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, "ID {} {}".format(objectId, CLASSES[ClasID]), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    fps.stop()
    cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)


    # Hiển thị frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Nếu nhấn q thì thoát vòng lặp
    if key == ord("q"):
        break

    # Cập nhật số khung hình đã xử lý và tính FPS
    fps.update()

