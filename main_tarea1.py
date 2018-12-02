import math
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

#puntos_p = [(370, 19), (1202, 48), (326, 1126), (1217, 1134)]
puntos_p = [(1234.538745387454, 60.3947368421052), (2872.560885608856, 247.61842105263145),
            (1294.760147601476, 1636.6973684210527), (2932.7822878228785, 1370.9605263157896)]
escala = (1000, 700)
def aplicar_homografia(puntos_p):
    """Estimar homografia"""
    puntos = [(0, 0), (escala[0], 0), (0, escala[1]), (escala[0], escala[1])]
    lista_a = []
    lista_b = []
    for punto_p, punto in zip(puntos_p, puntos):
        lista_a.append([punto[0], punto[1], 1, 0, 0, 0, -punto_p[0] * punto[0],
                        -punto_p[0] * punto[1]])
        lista_a.append([0, 0, 0, punto[0], punto[1], 1, -punto_p[1] * punto[0],
                        -punto_p[1] * punto[1]])
        lista_b.append(punto_p[0])
        lista_b.append(punto_p[1])

    matriz_a = np.matrix(lista_a)
    matriz_b = np.matrix(lista_b)
    out_u, out_s, out_vh = np.linalg.svd(matriz_a)
    out_s = np.diag(out_s)
    # Calcular la homografia
    homografia_inicial = out_vh.T*np.linalg.solve(out_s, (out_u.T*matriz_b.T))
    lista_homo = [elemento[0] for elemento in homografia_inicial.tolist()]
    # Reacomodar la homografia con h33 = 1
    lista_homo.append(1)
    homografia_final = np.matrix([lista_homo[:3], lista_homo[3:6], lista_homo[6:9]])
    print(homografia_final)
    return homografia_final

def interpolacion_bilineal(ruta_imagen, homografia):
    """Aplicar interpolación bilineal"""
    imagen = io.imread(ruta_imagen)
    filas, columnas, _ = imagen.shape
    vertices = [{'x': 0, 'y': 0}, {'x': 0, 'y': 0}, {'x': 0, 'y': 0}, {'x': 0, 'y': 0}]
    punto = {'x': 0, 'y': 0}
    esquina1 = np.linalg.solve(homografia, np.matrix([0, 0, 1]).T)
    print(esquina1)
    esquina1 = esquina1/esquina1[2]
    print(esquina1)
    esquina1 = [elemento[0] for elemento in esquina1.tolist()]
    print(esquina1)
    esquina2 = np.linalg.solve(homografia, np.matrix([columnas, 0, 1]).T)
    esquina2 = esquina2/esquina2[2]
    esquina2 = [elemento[0] for elemento in esquina2.tolist()]
    esquina3 = np.linalg.solve(homografia, np.matrix([0, filas, 1]).T)
    esquina3 = esquina3/esquina3[2]
    esquina3 = [elemento[0] for elemento in esquina3.tolist()]
    esquina4 = np.linalg.solve(homografia, np.matrix([columnas, filas, 1]).T)
    esquina4 = esquina4/esquina4[2]
    esquina4 = [elemento[0] for elemento in esquina4.tolist()]
    xminimo = min(esquina1[0], esquina2[0], esquina3[0], esquina4[0])
    xmaximo = max([esquina1[0], esquina2[0], esquina3[0], esquina4[0]])
    yminimo = min([esquina1[1], esquina2[1], esquina3[1], esquina4[1]])
    ymaximo = max([esquina1[1], esquina2[1], esquina3[1], esquina4[1]])
    hola = np.matrix([[0, xminimo], [0, 1, yminimo], [0, 0, 1]])
    homografia_nueva = homografia*np.matrix([[1, 0, xminimo],
                                            [0, 1, yminimo],
                                            [0, 0, 1]])
    nuevas_filas = math.ceil(ymaximo - yminimo)
    nuevas_columas = math.ceil(xmaximo - xminimo)
    nueva_imagen = np.zeros([nuevas_filas, nuevas_columas, 3], dtype=np.uint8)

    for m in range(nuevas_filas):
        for n in range(nuevas_columas):
            coordenadas = homografia_nueva * np.matrix([n, m, 1]).T
            coordenadas = coordenadas / coordenadas[2]
            coordenada = [coor[0] for coor in coordenadas.tolist()]
            evaluar = lambda f: imagen[f['y'], f['x'], :]
            interpolar = lambda p, q1, q2, q3, q4: evaluar(q1) * (q4['x'] - p['x']) * (q4['y'] - p['y']) \
                                                   - evaluar(q2) * (q4['x'] - p['x']) * (q1['y'] - p['y']) - evaluar(
                q3) * (q1['x'] - p['x']) * (q3['y'] - p['y']) \
                                                   + evaluar(q4) * (q1['x'] - p['x']) * (q1['y'] - p['y'])
            verificar = lambda q1, q2: (q1['x'] >= 0) and (q2['x'] <= columnas - 1) and (q1['y'] >= 0) and (
                        q2['y'] <= filas - 1)
            punto['x'] = coordenada[0]
            punto['y'] = coordenada[1]
            vertices[0]['x'] = math.floor(punto['x'])
            vertices[1]['x'] = vertices[0]['x']
            vertices[2]['x'] = math.ceil(punto['x'])
            vertices[3]['x'] = vertices[2]['x']

            vertices[0]['y'] = math.floor(punto['y'])
            vertices[1]['y'] = math.ceil(punto['y'])
            vertices[2]['y'] = vertices[0]['y']
            vertices[3]['y'] = vertices[1]['y']
            if verificar(vertices[0], vertices[3]):
                nueva_imagen[m, n, :] = interpolar(punto, vertices[0], vertices[1], vertices[2], vertices[3])

    plt.imshow(nueva_imagen)
    print(nueva_imagen)
    plt.show()

imagen = "img1.jpg"
homografia = aplicar_homografia(puntos_p)
interpolacion_bilineal(imagen, homografia)
