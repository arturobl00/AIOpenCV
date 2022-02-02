import cv2
imagen = cv2.imread('opencv.png')
imagen2 = cv2.imread('opencv.png',0)

#Guardar una imagen
cv2.imwrite('opengris.jpg',imagen2)

cv2.imshow("Imagen original", imagen)
cv2.waitKey(0)

cv2.imshow("Imagen Procesada", imagen2)
cv2.waitKey(0)

cv2.destroyAllWindows()