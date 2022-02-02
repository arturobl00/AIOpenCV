import cv2
scImagen = cv2.imread('opencv.png',0)
cv2.imshow('Imagen sin color',scImagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

