import cv2
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    #model_selection=1,
    min_detection_confidence=0.5) as face_detection:
    
    while True:
        ban, frame = cam.read()
        if ban == False:
            break

        #var para el alto y ancho de la cara
        height, width, _ = frame.shape

        #Invertir el video
        frame = cv2.flip(frame,1)

        #Cambiar frame a RGB para todas las librerias de mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Tomar datos
        result = face_detection.process(frame_rgb)

        for detection in result.detections:
            mp_drawing.draw_detection(frame, detection)
        
        cv2.imshow("Cam Live", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cam.release()
cv2.destroyAllWindows()


        

