import cv2

#Proceso para activar o encender una camara y mandar a una varibale su captura
cap = cv2.VideoCapture(0)

#Escribir el video que queremos guardar
out = cv2.VideoWriter('MiVideo.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))

#Crear un ciclo para la captura en tiempo del video
while(cap.isOpened()):
    #Parametros que detectan la camara y extraen el fotograma de la misma
    ret,frame=cap.read()
    if ret == True:
        cv2.imshow('Video online',frame)
        out.write(frame)

        #Condicion para que termine la captura de video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#Regresacar y apagar la camara
cap.realise()
out.realise()
cv2.destroyAllWindows()


