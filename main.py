from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

# Khởi tạo đối tượng nhận diện tay
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

camera = cv2.VideoCapture(0) 

def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Chuyển đổi frame sang màu RGB (Mediapipe yêu cầu RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý detect tay
        results = hands.process(frame_rgb)
        
        # Vẽ các điểm nhận diện lên frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Encode frame thành định dạng JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()  
        
        # Trả về frame để hiển thị trên web
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
