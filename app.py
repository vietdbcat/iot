from flask import Flask,render_template,session,request,redirect, url_for, Response
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import math
from face import FaceDetection

app = Flask(__name__)

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/')
def index():
    if 'username' in session:
        return render_template('home.html')
    return redirect(url_for('login'))

@app.route('/camera')
def camera():
    if session['username']:
        return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            session['username'] = request.form['username']
            return redirect(url_for('index'))   
    if request.method == 'GET': 
        return render_template('login.html')

@app.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run()
    
    
# Tải lên mô hình nhận diện đối tượng
model = YOLO('yolov8n.pt') # model này dùng để nhận diện đối tượng Person
model2 = YOLO('best.pt') # model này dùng để nhận diện đối tượng Garbage

# Tải lên mô hình nhận diện khuôn mặt
face = FaceDetection(r"D:/yolov8/static/faceData", r"D:/yolov8/encodeImage.pkl")
face.KhoiTao()

# Lấy video đầu vào, truyền vào biến cap
cap = cv2.VideoCapture(1)

# Lấy kích thước video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tạo biến lưu trữ video
vid_writer = cv2.VideoWriter(
	'result.mp4',
	cv2.VideoWriter_fourcc(*'mp4v'), 
	30, (width, height))

# Khởi tạo khoảng cách để xác định đối tượng có hành vi xả rác
warning_distance = 150
change = 50

# Hàm tính khoảng cách đối tượng
# Ở đây sử dụng khoảng cách Mahattan
def mahattan(box1, box2): # box là bounding-box của đối tượng được nhận diện
    x1, y1, z = box1
    x2, y2 = box2
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

# Hàm vẽ bounding-box cho đối tượng
def detection_box(frame, box, label, color, thickness=3):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
    y_label = max(y1, label_size[1])
    cv2.putText(frame, label, (x1, y_label - baseline), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
    return frame

# Khởi tạo một số danh sách cần thiết
person = {} # Lưu trữ đối tượng người
garbage = {} # Lưu trữ đối tượng rác thải
warning = {} # Lưu trữ đối tượng nằm trong diện cảnh báo, nghi ngờ, cần được theo dõi
classes = ["Person","Garbage"] # Label

crop = [] # Ảnh của đối tượng xả rác sẽ được trích xuất
dumped = [] # Bounding box của đối tượng xả rác

def generate_frames():
    vid_writer = cv2.VideoWriter(
        'result.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'), 
        30, (width, height))
    warning_distance = 150
    change = 50
    # Khởi tạo một số danh sách cần thiết
    person = {} # Lưu trữ đối tượng người
    garbage = {} # Lưu trữ đối tượng rác thải
    warning = {} # Lưu trữ đối tượng nằm trong diện cảnh báo, nghi ngờ, cần được theo dõi
    classes = ["Person","Garbage"] # Label

    crop = [] # Ảnh của đối tượng xả rác sẽ được trích xuất
    dumped = [] # Bounding box của đối tượng xả rác
    
    while True:
        success, frame = cap.read() # đọc dữ liệu đầu vào từ camera
        annotated_frame = frame

        if success:
            # nhận diện đối tượng có nhãn Person
            results = model.track(frame, classes=0, persist=True)
            boxes = results[0].boxes.xyxy.cpu() # Trích xuất bounding-box
            ids = []
            if results[0].boxes.id is not None: # trích xuất id ( mỗi đối tượng được theo vết sẽ có 1 id riêng biệt)
                ids = results[0].boxes.id.int().cpu().tolist()
            
            # Thực hiện vẽ bounding box và thêm đối tượng vào danh sách Person    
            for box, id in zip (boxes, ids):
                if id in dumped:
                    annotated_frame = detection_box(frame=annotated_frame, box=box, label=str(id)+" DOI TUONG XA RAC", color=(0,0,255))
                else:
                    annotated_frame = detection_box(frame=annotated_frame, box=box, label=str(id)+" PERSON", color=(0,255,0))
                person[id] = [(box[0]+box[2])/2,(box[1]+box[3])/2, box]
            
            # Nhận diện đối tượng rác thải
            # Quy trình tương tự nhận diện Người
            results = model2.track(frame, persist=True)
            boxes = results[0].boxes.xyxy.cpu()
            ids = []
            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().cpu().tolist()
            for box, id in zip (boxes, ids):
                annotated_frame = detection_box(frame=annotated_frame, box=box, label=str(id)+" GARBAGE", color=(51, 153, 255))
                garbage[id] = [(box[0]+box[2])/2,(box[1]+box[3])/2] 
                
            # Nếu khoảng cách quá gần, đưa vào danh sách đối tượng cảnh báo
            for i, ps in person.items():
                for j, gb in garbage.items():
                    if mahattan(ps, gb) < warning_distance:
                        warning[i] = j

            # Thực hiện kiểm tra các đối tượng trong danh sách cảnh báo
            pop=[] # pop là danh sách các đối tượng đã bị phát hiện, sẽ được loại bỏ khỏi danh sách Person để giảm thiểu thời gian xử lí
            for i, j in warning.items():
                if mahattan(person[i], garbage[j]) > warning_distance + change:
                    box = person[i][2]
                    x1, y1, x2, y2 = map(int, box)
                    
                    crop_img = frame[y1-50:y2+50,x1-50:x2+50] # lưu lại ảnh đối tượng
                    crop.append(crop_img)
                    
                    info = face.detect(crop_img)
                    name = "Unknown"
                    if len(info) > 0:
                        name = info[0]
                    cv2.imwrite(f"static/crop/{name}id{len(crop)}.jpg", crop_img)
                    
                    dumped.append(i)
                    pop.append(i)    
            for i in pop:
                warning.pop(i)
            
            vid_writer.write(annotated_frame)
                
            ret, tmp_frame = cv2.imencode('.jpg', annotated_frame)
            frame = tmp_frame.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # ngăn cách các frame bằng boundary
    vid_writer.release()
    cap.release()