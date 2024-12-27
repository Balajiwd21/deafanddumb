# Import necessary packages
import mediapipe as mp 
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize mediapipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model(r'checkpoints\gesture_15_model')

# Load class names
actions = np.array(['Hello', 'Love You', 'Understand', 'Thanks', 'Some', 'Home', 'name', 'my', 'how', 'Sorry', "Help me", "Yes", "No", "eat", "friend"])
print(actions)

sentence = []
predictions = []

def mediapipe_detection(image, model):
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    results = model.process(image)                 # Make prediction
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

colors = [(245,117,16), (117,245,16), (16,117,245)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*28), (int(prob*100), 90+num*28), colors[1], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:

    # Read each frame from the webcam
    _, frame = cap.read()
    x, y, c = frame.shape
    frame, result = mediapipe_detection(frame, holistic)
    className = ''

    # Process the result
    if result.left_hand_landmarks or result.right_hand_landmarks:
        landmarks = []

        lh = [[res.x, res.y] for res in result.left_hand_landmarks.landmark] if result.left_hand_landmarks else np.zeros(21*2).reshape(-1,2).tolist()
        rh = [[res.x, res.y] for res in result.right_hand_landmarks.landmark] if result.right_hand_landmarks else np.zeros(21*2).reshape(-1,2).tolist()
        
        for i in range(len(lh)):
            landmarks.append([lh[i][0], lh[i][1], rh[i][0], rh[i][1]])
        
        # Drawing landmarks on frames
        mpDraw.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
        mpDraw.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


        # Predict gesture
        prediction = model.predict([landmarks])
        classID = np.argmax(prediction)
        predictions.append(classID)
        className = actions[classID]

        if np.unique(predictions[-20:])[0] == classID: 
                if prediction[0][classID] > 0.7: 
                    
                    if len(sentence) > 0: 
                        if actions[classID] != sentence[-1]:
                            sentence.append(actions[classID])
                    else:
                        sentence.append(actions[classID])

        if len(sentence) > 4: 
            sentence = sentence[-4:]
        

        frame = prob_viz(prediction[0], actions, frame, colors)

    if cv2.waitKey(1) == ord('r'):
            if(len(sentence)!=0):
                sentence.pop()
            # print("Popped: ", sentence)
            
    # show the prediction on the frame
    cv2.rectangle(frame, (0,0), (640, 40), (255, 140, 51), -1)
    cv2.putText(frame, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()