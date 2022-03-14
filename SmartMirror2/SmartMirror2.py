import  cv2
import numpy as np
import mediapipe as mp

colorin = np.array([0,150,20], np.uint8)
colorfi = np.array([8,255,255], np.uint8)

colorin2 = np.array([175,150,20], np.uint8)
colorfi2 = np.array([180,255,255], np.uint8)

#Activar camara
cam = cv2.VideoCapture(1)

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

cam = cv2.VideoCapture(1)

colorin = np.array([0,150,20], np.uint8)
colorfi = np.array([8,255,255], np.uint8)

colorin2 = np.array([175,150,20], np.uint8)
colorfi2 = np.array([180,255,255], np.uint8)

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)

    if ret == True:
        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detecta = cv2.inRange(frameHSV, colorin, colorfi)
        detecta2 = cv2.inRange(frameHSV, colorin2, colorfi2)
        sensor = cv2.add(detecta,detecta2)

        #Porceso para dibujar contornos de deteccion
        #PASO 1 DEDECTAR LOS CONTONORNOS BASADO EN COLOR
        contono,_ = cv2.findContours(sensor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #Paso 2 Crear un ciclo para dibujar todos los contornos detectados
        for c in contono:
            area = cv2.contourArea(c)
            if area > 500:
                cv2.drawContours(frame, [c], -1, (0,255,0), 2)

        cv2.imshow("Camara Sensor Azul", sensor)
        cv2.imshow("WebCam On",frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cam.release()
cv2.destroyAllWindows()

cam = cv2.VideoCapture(1);
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
cv2.destroyAllWindows()

rostros = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

imagen = cv2.imread('rostros.jpg')
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

#Funcion de deteccion de rostos
caras = rostros.detectMultiScale(gray,
    scaleFactor=1.1,
    minNeighbors=6,
    minSize=(30,30),
    maxSize=(0,200))

#Ciclo para crear los rectangulos de rostros detectados
for(x,y,w,h) in caras:
    cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('Diversidad Personas',imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

image2 = cv2.imread("rostro1.jpg")
cv2.imshow("Analitica de Rostros", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Importamos las librerias de dibujo y deteccion esto es de lo mas importate
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

#Para la deteccion le damos los parametros de tamañao y cantidad de manos
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces = 3,
    min_detection_confidence=0.5) as face_mesh:

    #Leemos una imagen estatica como pusimos en static_image_mode
    image = cv2.imread("rostro.jpg")
    
    #Tomamos sus propiedades de alto y ancho
    height, width, _ = image.shape
    
    #Cobnvertimos la imagen a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Vamos a ver la informaciond e la imagen
    result = face_mesh.process(image_rgb)
    
    #imprimimos en consola los datos nos da los puntos y que mano es
    print("Face Landmarks", result.multi_face_landmarks)

    #Corremos la app 1

    #Parte 2
    #Ponemos una condicion si es que no encontramos manos
    if result.multi_face_landmarks is not None:
        #Si tenemos informacion entonces con un form recorremos los datos y dibujamos
        for face_landmark in result.multi_face_landmarks:
            
            #Dibujar los 21 puntos de deteccion
            #mp_drawing.draw_landmarks(
             #   image, face_landmark, mp_face_mesh.FACE_CONNECTIONS)
            
            #podemos cambiar el color de las conecciones
            mp_drawing.draw_landmarks(
                image, face_landmark, mp_face_mesh.FACE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=1))

    #Mostramos la imagen en una ventana
    cv2.imshow("Imagen",image)

cv2.waitKey(0)
cam.release()
cv2.destroyAllWindows()

#Ahora usaremos este codigo y lo pondremos en un video
cap = cv2.VideoCapture(1)
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces = 2,
    min_detection_confidence=0.5) as face_mesh:

    #Ciclo de video
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = face_mesh.process(frame_rgb)
        
        if result.multi_face_landmarks is not None:
        #Si tenemos informacion entonces con un form recorremos los datos y dibujamos
            for face_landmark in result.multi_face_landmarks:
                #Dibujar los 21 puntos de deteccion
                mp_drawing.draw_landmarks(
                frame, face_landmark, mp_face_mesh.FACE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=1))

              
        #Mostramos
        cv2.imshow("Video",frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cam.release()
cv2.destroyAllWindows()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(1)

#Configuracion de mediapipe
with mp_hands.Hands(
    static_image_mode = True,
    max_num_hands = 2,
    min_detection_confidence = 0.5) as hands:

    #leer video
    while True:
        ban, frame = cam.read()
        if ban == False:
            break

        #var para el alto y ancho de la cara
        height, width, _ = frame.shape

        #Invertir el video
        frame = cv2.flip(frame,1)

        #Cambiar frame a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #TOmar datos
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks is not None:

            index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                #Ciclo para mostrar los puntos de la mano
                for (i, points) in enumerate(hand_landmarks.landmark):
                    if i in index:
                        x = int(points.x * width)
                        y = int(points.y * height)
                        cv2.circle(frame, (x,y), 10, (0,255,0), -4)
                        cv2.circle(frame, (x,y), 5, (0,0,255), -2)

        #Mostrar video procesado
        cv2.imshow("Cam MediaPipe", frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cam.release()
cv2.destroyAllWindows()


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(1)

#Configuracion de mediapipe
with mp_pose.Pose(
    static_image_mode = True,
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

        #Cambiar frame a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #TOmar datos
        result = pose.process(frame_rgb)

        negro = np.zeros(frame.shape, np.uint8)

        if result.pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                negro, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=6))
            
        #Mostrar video procesado
        cv2.imshow("Cam Live", frame)
        cv2.imshow("Cam MediaPipe", negro)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cam.release()
cv2.destroyAllWindows()




