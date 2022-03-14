import cv2
import numpy as np
import mediapipe as mp

#Librerias
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


cam = cv2.VideoCapture(1)

#Configuracion de mediapipe
with mp_pose.Pose(
    static_image_mode = False,
    min_detection_confidence = 0.5) as pose:

    #leer video
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
        result = pose.process(frame_rgb)
        #Crear una ventana en negro
        negro = np.zeros(frame.shape, np.uint8)

        if result.pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                negro, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2))

        #Mostrar video procesado
        cv2.imshow("Cam Live", frame)
        cv2.imshow("Cam MediaPipe", negro)
        
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cam.release()
cv2.destroyAllWindows()



