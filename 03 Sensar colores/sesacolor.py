import  cv2
import numpy as np

#Sesnsar colores mediante la camara
#RAngo y color para sensar
#                    H  S  V
#Hue 0 al 180
#Saturacion 0 a 255
#Brillo 0 a 255

colorin = np.array([0,150,20], np.uint8)
colorfi = np.array([8,255,255], np.uint8)

colorin2 = np.array([175,150,20], np.uint8)
colorfi2 = np.array([180,255,255], np.uint8)

#Activar camara
cam = cv2.VideoCapture(0)

while True:
    ban, frameBGR = cam.read()
    frameBGR = cv2.flip(frameBGR,1)

    #Convertir imagen de BGR a HSV 
    frameHSV = cv2.cvtColor(frameBGR,cv2.COLOR_BGR2HSV)

    #Detaactar colores
    detecta = cv2.inRange(frameHSV, colorin, colorfi)
    
    detecta2 = cv2.inRange(frameHSV, colorin2, colorfi2)
    mix = cv2.add(detecta,detecta2)

    #Usando Bitwise para mostrar imagen con color detectado como mascara

    imagenBitwise = cv2.bitwise_and(frameBGR, frameBGR, mask = mix)

    #Condicion para mostrar los fotogramas
    if ban == True:
        cv2.imshow("Camara On",frameBGR)
        #cv2.imshow("Camara HSV",frameHSV)
        cv2.imshow("Camara Detecta Rojo con Bitwise",imagenBitwise)
    
    #Condicion para apagar la camara
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

cam.release()
cv2.destroyAllWindows()


