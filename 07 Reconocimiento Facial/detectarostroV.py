import cv2
import numpy as nps

detectarostro = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

##Detecta video
cam = cv2.VideoCapture(0)

#Cambiar a escala de gris
while True:
    ret,frame = cam.read()
    if ret == True:
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Variable y Funicon para deteccion de rostros
        caras = detectarostro.detectMultiScale(gris,
        scaleFactor = 1.2,
        minNeighbors = 5, 
        minSize=(30,30), 
        maxSize=(0,200))
    
        #Dibujar los rostros detectados
        for (x1,y1,x2,y2) in caras:
            cv2.rectangle(frame, (x1,y1), (x1+x2, y1+y2), (255,255,255), 2)
            #Mostrar la imagen
            cv2.imshow('Imagen Detectada', frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cam.release()
cv2.destroyAllWindows()