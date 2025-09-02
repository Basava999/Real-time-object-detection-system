from flask import Flask, render_template, Response
from flask_sqlalchemy import SQLAlchemy
import cv2
from ultralytics import YOLO

# Flask setup
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'
db = SQLAlchemy(app)

# Database model
class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    object_name = db.Column(db.String(50))
    confidence = db.Column(db.Float)

# YOLO model
model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)

def blur_face(frame, x1, y1, x2, y2):
    face = frame[y1:y2, x1:x2]
    if face.size != 0:
        face = cv2.GaussianBlur(face, (99, 99), 30)
        frame[y1:y2, x1:x2] = face
    return frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model(frame)

            for r in results:
                for box in r.boxes:
                    cls = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])

                    # Save in DB
                    detection = Detection(object_name=cls, confidence=conf)
                    db.session.add(detection)
                    db.session.commit()

                    # Coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Blur person for privacy
                    if cls == "person":
                        frame = blur_face(frame, x1, y1, x2, y2)

                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
