import cv2
import numpy as np

#este programa detectara solo la precencia del rojo usando HSV que se refiere a
#Hue Matiz, Saturation Saturación y Value Brillo
#valores son hue de 0 a 179, saturation de 0 a 255 y value de 0 a 255

redBajo1 = np.array([0, 100, 20], np.uint8)
redAlto1 = np.array([8, 255, 255], np.uint8)

redBajo2=np.array([175, 100, 20], np.uint8)
redAlto2=np.array([178, 255, 255], np.uint8)


#Primero decaramos una variable para iniciar en una ventana captura de video sin detenerce
cap = cv2.VideoCapture(0)

#con un ciclo mostramos la ventana e interrumpimos hasta precionar la s
while True:
  ret,frame = cap.read()
  if ret==True:
    #convertimos el frame a un frameHSV
    frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    maskRed1 = cv2.inRange(frameHSV,redBajo1,redAlto1)
    maskRed2 = cv2.inRange(frameHSV,redBajo1,redAlto2)
    maskRed = cv2.add(maskRed1,maskRed2)
  
    cv2.imshow('frame', frame)
    cv2.imshow('Mask', maskRed)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break

#refrescamos la memoria
cap.release()
#cerramos ventana
cv2.destroyAllWindows()
