import cv2
import mediapipe as mp

#Importamos las librerias de dibujo y deteccion esto es de lo mas importate
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

#Para la deteccion le damos los parametros de tamañao y cantidad de manos
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces = 3,
    min_detection_confidence=0.5) as face_mesh:

    #Leemos una imagen estatica como pusimos en static_image_mode
    image = cv2.imread("rostros.jpg")
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
cv2.destroyAllWindows()

#Ahora usaremos este codigo y lo pondremos en un video
cap = cv2.VideoCapture(0)
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
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=1))

              
        #Mostramos
        cv2.imshow("Video",frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

cap.realice()
cv2.destroyAllWindows()