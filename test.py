#Chương trình nhận diện cử chỉ bàn tay hiển thị chữ cái
import cv2 
import mediapipe as mp 
import time
from google.protobuf.json_format import MessageToDict 

# Cấu hình và khái báo đối tượng
mp_drawing_util = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
    model_complexity = 0,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
    static_image_mode=False, 
    model_complexity=1,
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=2) 

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# Mở máy ảnh
cap = cv2.VideoCapture(0)
cap.set(3,840) 
cap.set(4,680)
class Handtracking():
    while cap.isOpened():
        success, img = cap.read()

        img=cv2.flip(img,1)
        # Đọc hình ảnh thành công
        if not success:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        results = hands.process(imgRGB) 

        alphabet = ''
        #Phân biệt tay trái tay phải và hai tay        
        if results.multi_hand_landmarks: 

            # Cả hai bàn tay đều có trong hình ảnh (khung) 
            if len(results.multi_handedness) == 2: 
                # Hiển thị “Cả hai tay” trên hình ảnh
                cv2.putText(img, 'Both Hands', (250, 50), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.9, 
                            (0, 255, 0), 2) 

            # Nếu có bàn tay nào hiện diện 
            else: 
                for i in results.multi_handedness: 
                    # Return whether it is Right or Left Hand 
                    label = MessageToDict(i)[ 'classification'][0]['label'] 

                    if label == 'Left':
                        # Trả về dù là Tay Phải hay Tay Trái
                        cv2.putText(img, label+' Hand', (20, 50), 
                                    cv2.FONT_HERSHEY_COMPLEX, 0.9, 
                                    (0, 255, 0), 2) 
                    if label == 'Right':
                        # Hiển thị 'Tay trái' ở bên trái cửa sổ 
                        cv2.putText(img, label+' Hand', (460, 50), 
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.9, (0, 255, 0), 2) 
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    #print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    #if id ==0:
                    cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        if result.multi_hand_landmarks:
            myHand = []

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
    
                # Nhận diện chữ A
                if myHand[4][0] > myHand[2][0] and myHand[8][1] > myHand[5][1] and myHand[12][1] > myHand[9][1] and myHand[16][1] > myHand[13][1] and myHand[20][1] > myHand[17][1]:
                    alphabet = 'A'
                # Nhận diện chữ B
                elif myHand[4][0] < myHand[2][0] and myHand[8][1] < myHand[5][1] and myHand[12][1] < myHand[9][1] and myHand[16][1] < myHand[13][1] and myHand[20][1] < myHand[17][1]:
                    alphabet = 'B'
                # Nhận diện chữ C
                elif myHand[4][0] > myHand[5][0] and myHand[8][0] > myHand[5][0] and myHand[12][0] > myHand[9][0] and myHand[16][0] > myHand[13][0] and myHand[20][1] > myHand[18][1]:
                    alphabet = 'C'
                # Nhận diện chữ D
                elif myHand[4][0] > myHand[12][0] and myHand[8][0] > myHand[5][0] and myHand[12][0] > myHand[9][0] and myHand[16][0] > myHand[13][0] and myHand[20][1] < myHand[18][1]:
                    alphabet = 'D'
                # Nhận diện chữ E
                elif myHand[4][0] < myHand[11][0] and myHand[8][1] > myHand[5][1] and myHand[12][1] > myHand[9][1] and myHand[16][1] > myHand[13][1] and myHand[20][1] > myHand[17][1]:
                    alphabet = 'E'
                # Nhận diện chữ F
                elif myHand[4][0] > myHand[2][0] and myHand[8][1] > myHand[5][1] and myHand[12][1] < myHand[9][1] and myHand[16][1] < myHand[13][1] and myHand[20][1] < myHand[17][1]:
                    alphabet = 'F'
                # Nhận diện chữ I
                elif myHand[4][0] < myHand[10][0] and myHand[8][1] > myHand[5][1] and myHand[12][1] > myHand[9][1] and myHand[16][1] > myHand[13][1] and myHand[20][1] < myHand[17][1]: 
                    alphabet = 'I'
                # Nhận diện chữ K
                elif myHand[4][0] > myHand[2][0] and myHand[8][1] < myHand[5][1] and myHand[12][1] < myHand[9][1] and myHand[16][1] > myHand[13][1] and myHand[20][1] > myHand[17][1]: 
                    alphabet = 'K'
                # Nhận diện chữ L
                elif myHand[4][0] > myHand[2][0] and myHand[4][0] > myHand[12][0] and myHand[8][1] < myHand[5][1] and myHand[12][1] > myHand[9][1] and myHand[16][1] > myHand[13][1] and myHand[20][1] > myHand[17][1]: 
                    alphabet = 'L'
                # Nhận diện chữ P
                elif myHand[4][0] > myHand[11][0] and myHand[8][0] < myHand[6][0] and myHand[12][0] < myHand[9][0] and myHand[16][0] < myHand[13][0] and myHand[20][0] < myHand[17][0]:
                    alphabet = 'P'
                # Nhận diện chữ S
                elif myHand[4][0] > myHand[11][0] and myHand[8][1] > myHand[5][1] and myHand[12][1] > myHand[9][1] and myHand[16][1] > myHand[13][1] and myHand[20][1] > myHand[17][1]:
                    alphabet = 'S'
                # Nhận diện chữ V
                elif myHand[4][0] < myHand[2][0] and myHand[8][1] < myHand[5][1] and myHand[12][1] < myHand[9][1] and myHand[16][1] > myHand[13][1] and myHand[20][1] > myHand[17][1]: 
                    alphabet = 'V'
                # Nhận diện chữ W
                elif myHand[4][0] < myHand[2][0] and myHand[8][1] < myHand[5][1] and myHand[12][1] < myHand[9][1] and myHand[16][1] < myHand[13][1] and myHand[20][1] > myHand[17][1]: 
                    alphabet = 'W'
                # Nhận diện chữ Y
                elif myHand[4][0] > myHand[10][0] and myHand[8][1] > myHand[5][1] and myHand[12][1] > myHand[9][1] and myHand[16][1] > myHand[13][1] and myHand[20][1] < myHand[17][1]: 
                    alphabet = 'Y'
                # Các trường hợp cử chi sai
                else:
                    alphabet = 'Unknow'

        # Hiểm thị chữ đã nhận dạng ra màn hình
        cv2.putText(img, str(alphabet), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # Hiển thị hình ảnh bàn tay
        cv2.imshow("Nhan dang bang chu cai bang cu chi tay", img)
        # Gán 1 key để thoát khỏi chương trình
        if cv2.waitKey(1) == ord('q'):
            break
# Đóng máy ảnh
cap.release()
cv2.destroyAllWindows()