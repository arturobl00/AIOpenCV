import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

cam = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces = 1,
    min_detection_confidence=0.5) as face_mesh:
    
    while True:
        ret, frame = cam.read()
        if ret == False:
            break
        
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = face_mesh.process(frame_rgb)

        if result.multi_face_landmarks is not None:
            #Si tenemos informacion entonces con un form recorremos los datos y dibujamos
            for face_landmark in result.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                frame, face_landmark, mp_face_mesh.FACE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=1))

        cv2.imshow("Detectando Rostros",frame)
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cam.release()
cv2.destroyAllWindows()
