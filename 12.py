#Nhận dạng tay 1


# Importing Libraries 
import cv2 
import mediapipe as mp 
import time

# Used to convert protobuf message 
# to a dictionary. 
from google.protobuf.json_format import MessageToDict 
#Nhận dạng tay 2
# Initializing the Model 
mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
    static_image_mode=False, 
    model_complexity=1,
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=2) 

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
#Nhận dạng tay 3

# Start capturing video from webcam 
cap = cv2.VideoCapture(0) 

while True:
    # Read video frame by frame 
    success, img = cap.read() 

    # Flip the image(frame) 
    img = cv2.flip(img, 1) 
    # Convert BGR image to RGB image 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    # Process the RGB image 
    results = hands.process(imgRGB) 

    # If hands are present in image(frame) 
    if results.multi_hand_landmarks: 

        # Both Hands are present in image(frame) 
        if len(results.multi_handedness) == 2: 
            # Display 'Both Hands' on the image 
            cv2.putText(img, 'Both Hands', (250, 50), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, 
                        (0, 255, 0), 2) 

        # If any hand present 
        else: 
            for i in results.multi_handedness: 
                # Return whether it is Right or Left Hand 
                label = MessageToDict(i)[ 'classification'][0]['label'] 

                if label == 'Left':
                    # Display 'Left Hand' on left side of window 
                    cv2.putText(img, label+' Hand', (20, 50), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.9, 
                                (0, 255, 0), 2) 
                if label == 'Right':
                    # Display 'Left Hand' on left side of window 
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
    
    
    
    
    # Display Video and when 'q' is entered, destroy the window 
    cv2.imshow('Image', img) 
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break
