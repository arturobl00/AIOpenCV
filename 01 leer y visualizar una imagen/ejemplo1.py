#Importar la libreria o librerias
import cv2

#Instrucion imread para leer una imagen o un video
imagen = cv2.imread('opencv.png')

#Instruccion para mostrar el contenido de una variable en una ventana
cv2.imshow('AI Primera Practica', imagen)

#Instruccion para realizar una pausa o press any key to continue
cv2.waitKey(0)

#Destruir o terminar una aplicacion 
cv2.destroyAllWindows()

