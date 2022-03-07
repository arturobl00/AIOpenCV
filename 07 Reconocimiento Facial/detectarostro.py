import cv2
import numpy as np

detectarostro = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

##Detecta sobre imagen o foto
foto = cv2.imread('gays.jpg')

#Cambiar a escala de gris
gris = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)

#Variable y Funicon para deteccion de rostros
caras = detectarostro.detectMultiScale(gris,
scaleFactor = 1.2,
minNeighbors = 5, 
minSize=(30,30), 
maxSize=(0,200))

#Dibujar los rostros detectados
for (x1,y1,x2,y2) in caras:
    cv2.rectangle(foto, (x1,y1), (x1+x2, y1+y2), (255,255,255), 2)

#Mostrar la imagen
cv2.imshow('Imagen Detectada', foto)
cv2.waitKey(0)
cv2.destroyAllWindows()