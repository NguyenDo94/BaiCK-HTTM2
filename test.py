import cv2 
import mediapipe as mp 



mp_drawing_util = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

# Mở máy ảnh
cap = cv2.VideoCapture(0)


while cap.isOpened():
    success, img = cap.read()

    img=cv2.flip(img,1)
    if not success:
        break

    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    count=''
    if result.multi_hand_landmarks:
        myHand = []
        count=0
        for idx, hand in enumerate(result.multi_hand_landmarks):
            # Vẽ tọa độ khung xương bàn tay
            mp_drawing_util.draw_landmarks(img, 
                                           hand, 
                                           mp_hand.HAND_CONNECTIONS,
                                           mp_drawing_style.get_default_hand_landmarks_style(),
                                           mp_drawing_style.get_default_hand_connections_style(),
                                           )
            # lbl = result.multi_handedness[idx].classification[0].label
            for id, lm in enumerate(hand.landmark):
                # Lấy các tọa độ
                h, w, _ = img.shape
                myHand.append([int(lm.x * w), int(lm.y * h)])
            # Từ các tọa độ => chữ cái muốn hiển thị
            if myHand[8][1]< myHand[5][1]:
                count=count+1
            if myHand[12][1]< myHand[9][1]:
                count=count+1
            if myHand[16][1]< myHand[13][1]:
                count=count+1
            if myHand[20][1]< myHand[17][1]:
                count=count+1
            if myHand[4][0]< myHand[2][0]:
                count=count+1

    # Hiểm thị chữ đã nhận dạng ra màn hình
    cv2.putText(img, str(count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)
    # Hiển thị hình ảnh bàn tay
    cv2.imshow("Nhan dang bang chu cai bang cu chi tay", img)
    
    if cv2.waitKey(1) == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()