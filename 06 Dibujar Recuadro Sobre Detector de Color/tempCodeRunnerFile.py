import cv2
import numpy as np

cam = cv2.VideoCapture(0);
colorini = np.array([170,100,20], np.uint8)
colorfin = np.array([180,255,255], np.uint8)

while True:
    ret, frame = cam.read()
    if ret==True:
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detector = cv2.inRange(frameHSV,colorini,colorfin)
        #Deteccion en contorno de la imagen
        contono,_ = cv2.findContours(detector, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #Ciclo para dibujar el contorno
        for c in contono:
            area = cv2.contourArea(c)
            if area > 3000:
                x1, y1, x2, y2 = cv2.boundingRect(c)
                cv2.rectangle(frame, (x1,y1), (x1+x2,y1+y2), (0,255,0), 2)
                cv2.putText(frame, 'Rojo', (x1,y1-20), 0, 1, (0,255,0), 2)
            cv2.imshow("Detectando Color",frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
cam.release()
cv2.destroyAllWindows(0)


    


