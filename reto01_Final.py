
# Importamos las librerias necesarias

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Act 1.1 -> Usar lo de leer los frames de un video e irlos guardando en un video nuevo

# Act 1.2 -> Vimos segmentacion y los diferentes thresholds (umbralizacion)

# NOTA : Considerar usar MediamBlur a la imagen de color para usar el edge detector (checar pagina 61 - 70)

# --------------------------------- DECLARACION DE VARIABLES ----------------------------------------
offsetRefLines = 175  #

minContourArea = 100  # Ajusta su valor de acuerdo al area minima que desees detectar
entranceCounter = 0
coordXEntranceLine = 0
pixelTolerance = 50  # Tolerancia

# --------------------------------- DECLARACION DE FUNCIONES -----------------------------------------

# verifica si la coordenada 'x' del centroide de los bordes esta pasando la linea de entrada
def check_entranceline(x, coordx_entranceline):
    absdistance = abs(x - coordx_entranceline)

    # como el flujo del video va de derecha izquierda, usamos 'menor que' donde esta la linea. Si fuera en el sentido, contrario seria 'mayor que'
    if ((absdistance <= pixelTolerance) and (x < coordx_entranceline)):
        return 1
    else:
        return 0

# --------------------------------- CONFIGURACION DE LA CAMARA --------------------------------------
camera = cv2.VideoCapture('bloques.mp4')

# resolucion de la camara
camera.set(3, 640)
camera.set(4, 480)

for i in range(0, 10):
    grabbed, frame = camera.read()
    height = np.size(frame, 0)
    width = np.size(frame, 1)

    # linea de referencia
    coordXEntranceLine = round((width / 2) - offsetRefLines)

# -------------------------------------- INICIO ------------------------------------------------------

# Lineas de codigo para poder grabar el resultado en un nuevo video
fps = camera.get(cv2.CAP_PROP_FPS)
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('VideoDeResultados.mp4', cv2.VideoWriter_fourcc('I','4','2','0'), fps, size)

while cv2.waitKey(1) == -1:
    
    # Obtenemos un frame del video
    grabbed, frame = camera.read()

    # Si ya no encontramos frames, salimos del while
    if not grabbed:
        break

    # transformacion a escala de grises
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # METODOS PARA AL DETECCION DE LOS CONTORNOS

    # binarizamos la imagen 'grayFrame'
    _, th = cv2.threshold(grayFrame, 80, 255, cv2.THRESH_BINARY)

    # creamos el kernel (5x5) de puros 1s para la morfologia
    kernel  =  np.ones (( 5, 5), np.uint8 )  ## construimos un kernel de 5x5 de solo unos

    # aplicamos la función de apertura a la imagen binarizada 'th' (1 erosion y luego 1 dilatacion)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)  

    # aplicamos el filtro canny para la detección de bordes. Seleccionamos un umbral minimo de 135 y un maximo de 255
    bordes = cv2.Canny(opening,135,255)

    # En la jerarquia guardamos los contornos detectados en la imagen 'bordes'
    cnts, hierarchy = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    qtyOfContours = 0

    # Guardamos el frame en otra variable, la cual modificaremos para desplegar lo detectado
    disp_frame = frame

    # Ciclo for para marcar los contornos en el frame
    for c in cnts:

        print("area ", cv2.contourArea(c))
        
        # si el contour tiene una area menor se descarta y se sigue al siguiente contorno
        if cv2.contourArea(c) < minContourArea:
            continue

        qtyOfContours += 1

        # dibuja un bounding box alrededor del objeto
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(disp_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        # centroides --> Estos te ayudaran a saber donde esta el centro de los contornos que detecto 'c'
        coordXCentroid = round(x + w / 2)
        coordYCentroid = round(y + h / 2)
        cv2.circle(disp_frame, (coordXCentroid, coordYCentroid), 1, (0, 0, 255), 5)

        if check_entranceline(coordXCentroid, coordXEntranceLine):
             entranceCounter += 1
    
    # Pasamos al ultimo paso : mostrar en la pantalla los resultados
    print("Contornos encontrados: " + str(qtyOfContours))

    # letrero con objetos encontrados
    cv2.putText(disp_frame, "Objetos: {}".format(str(entranceCounter)), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 1), 2)

    cv2.imshow("Imagen binaria", th)
    cv2.imshow("Imagen con detecciones", disp_frame)
    cv2.waitKey(1);

    # Tambien guardamos el resultado en un archivo de video .mp4
    videoWriter.write(disp_frame)
    cv2.waitKey(1);

# cierra ventanas
camera.release()
cv2.destroyAllWindows()

