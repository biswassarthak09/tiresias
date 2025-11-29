from ultralytics import YOLO
import cv2

def detect_objects_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, 30,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, conf, label in zip(boxes, confs, labels):
            if conf >= 0.2:
                x1, y1, x2, y2 = map(int, box)
                label_name = model.names[label]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f'{label_name} , {conf:.2f}',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

model = YOLO("yolo11n.pt")
detect_objects_in_video("live_capture.mp4")
