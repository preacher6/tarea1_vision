#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example on homography estimation using OpenCV

Created on Tue Sep 12 21:01:53 2017
@author: gholguin
"""

# Imports
import cv2
import numpy as np

# Variable global que se pueda compartir con el callback
puntos_click = list()

# ----------------------------------------------------------------------
def click_and_count(event, x, y, flags, param):
    """Definicion del callback para captura del raton"""

    global puntos_click

    # Si se hace click, mientras el boton baja, guardar las coordenadas
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos_click.append((x, y))

# =======================================================================
class MiHomografia():
    """Clase para solucionar problemas relacionados con homografias"""

    # Atributos de la clase
    reprojThresh = 0.01

    def __init__(self):
        """Inicializador del objeto miembro de la clase"""

        # Atributos del objeto
        self.imagen_original = list()
        self.rectificada = list()

        self.pts_x = np.array(list())
        self.pts_xp = np.array(list())

        self.H = list()

    def load_image(self, image_path):
        """Funcion para cargar una imagen desde el disco duro"""

        self.imagen_original = cv2.imread(image_path)

    def grab_four_points(self):
        """Capturar Puntos en la imagen"""

        global puntos_click

        # Clonar la imagen original para no modificarla
        imagen_conpuntos = self.imagen_original.copy()

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", click_and_count)

        while True:
            # Muestre la imagen hasta que se presione 'q'
            cv2.imshow("image", imagen_conpuntos)
            key = cv2.waitKey(1) & 0xFF

            # Si se presiona 'r', resetear los puntos
            if key == ord("r"):
                imagen_conpuntos = self.imagen_original.copy()
                puntos_click = list()

            # Si se presiona 'q' termine
            elif key == ord("q"):
                break

            # Mostrar los puntos en la imagen
            if puntos_click:
                for pt, coords in enumerate(puntos_click):
                    x, y = coords[0], coords[1]
                    cv2.circle(imagen_conpuntos, (x, y), 5, (0, 0, 255), 5, 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(imagen_conpuntos, str(pt+1), (x, y), font, 4, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("image", imagen_conpuntos)

    def encontrar_h(self):
        """Calculo robusto de H"""

        self.H, status = cv2.findHomography(self.pts_xp, self.pts_x, cv2.RANSAC, self.reprojThresh)
        return self.H, status

    def remover_proyectividad(self):
        """Basado en el H encontrado, remover la proyectividad"""

        self.rectificada = cv2.warpPerspective(self.imagen_original, self.H,
                                               (self.imagen_original.shape[1], self.imagen_original.shape[0]))
        cv2.namedWindow("Rectificada")
        cv2.imshow("Rectificada", self.rectificada)
        cv2.waitKey(0)


if __name__ == '__main__':

    # Crear un objeto de la clase MiHomografia
    hproblem = MiHomografia()

    # Usar el metodo load_image
    hproblem.load_image("capilla60.jpg")

    # Llamar el metodo que toma 4 puntos de la imagen
    hproblem.grab_four_points()
    print("\nEsquinas seleccionadas:")
    print(puntos_click)

    hproblem.pts_xp = np.array(puntos_click)
    hproblem.pts_x  = np.array([(200, 300), (700, 300), (700, 700), (200, 700)])

    H, status = hproblem.encontrar_h()
    print("\nMatriz H:")
    print(H)

    hproblem.remover_proyectividad()







