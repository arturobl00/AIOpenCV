import cv2
import numpy as np

cam = cv2.VideoCapture(0)

dectaini = np.array([85,150,20])
dectafin = np.array([130,255,255])

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)

    if ret == True:
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sensor = cv2.inRange(frameHSV, dectaini, dectafin)

        #Porceso para dibujar contornos de deteccion
        #PASO 1 DEDECTAR LOS CONTONORNOS BASADO EN COLOR
        contono,_ = cv2.findContours(sensor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #Paso 2 Crear un ciclo para dibujar todos los contornos detectados
        for c in contono:
            area = cv2.contourArea(c)
            if area > 500:
                cv2.drawContours(frame, [c], -1, (255,0,0), 2)

        cv2.imshow("Camara Sensor Azul", sensor)
        cv2.imshow("WebCam On",frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cam.release()
cam.destroyAllWindows()
