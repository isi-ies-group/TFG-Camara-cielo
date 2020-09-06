# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:36:32 2020

@author: Martin
"""

##########################################################################################
#######                                                                           ########
#######                                                                           ########
#######            INTERFAZ EN PYTHON PARA EL MANEJO DE LA CÁMARA                 ########
#######                                                                           ########
#######              Y DE LA APLIACICIÓN PARA ESTIMACIÓN DE LA                    ########
#######                                                                           ########
#######                          RADIACIÓN DIFUSA                                 ########
#######                                                                           ########
#######                                                                           ########
##########################################################################################



##########################################################################################
#######                                                                           ########
#######                               IMPORTS                                     ########
#######                                                                           ########
##########################################################################################

import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import pvlib as pv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import sys
sys.path.append('../')
import json

##########################################################################################
#######                                                                           ########
#######                           VARIABLES INTERNAS                              ########
#######                                                                           ########
##########################################################################################

#########  AÑADIR VARIABLES A FICHERO DE CONFIGURACIÓN
with open('config_interfaz.txt', 'r') as json_file:
    config_datos = json.load(json_file)

direccion_camara = config_datos['Direccion camara']

###########
path_fotos_patron =  config_datos['Patron']
img_patron = cv2.imread(path_fotos_patron, cv2.IMREAD_COLOR)

# Posición y clase de la cámara. PVLib.
Latitud = config_datos['Posicion camara']['Latitud']
Longitud = config_datos['Posicion camara']['Longitud']
Altitud = config_datos['Posicion camara']['Altitud']
Cam_Location = pv.location.Location(Latitud, Longitud, tz='Europe/Madrid', altitude=Altitud)

# Offset del ángulo azimutal: 187 - 100.52
offset_azimuth = config_datos['Offset azimuth']

# Variables relativas al tamaño de la imágen
X = config_datos['Imagen']['X']
Y = config_datos['Imagen']['Y']
centro = (int(Y/2), int(X/2))
R = int(Y/2)

# Rutas de los ficheros de datos
ruta_datos_areas = config_datos['Rutas']['Datos Areas']
ruta_datos_errores = config_datos['Rutas']['Datos Errores']
ruta_datos_regresion_nublado = config_datos['Rutas']['Datos Regresion Nublado']
ruta_datos_regresion_despejado = config_datos['Rutas']['Datos Regresion Despejado']


##########################################################################################
#######                                                                           ########
#######                      FUNCIONES INTERNAS DE LA CÁMARA                      ########
#######                                                                           ########
##########################################################################################

def cambio_pos_camara(latitud, longitud, altitud):
    '''
    Función encargada de modificar la posiciñon de la cámara en los ficehro de configuración.
    Importante mantener este dato actualizado para obtener los cáluclos 
    de los ángulos azimut y zenit de forma correcta.
    '''

    global Latitud, Longitud, Altitud, Cam_Location
    
    Latitud = latitud
    Longitud = longitud
    Altitud = altitud
    # Se obtiene de nuevo la clase pvlib.Location que calcula la posición del sol
    Cam_Location = pv.location.Location(Latitud, Longitud, tz='Europe/Madrid', altitude=Altitud)
    
    # Se actualizan los datos en el fichero json de configuración
    config_datos['Posicion camara']['Latitud'] = Latitud
    config_datos['Posicion camara']['Longitud'] = Longitud
    config_datos['Posicion camara']['Altitud'] = Altitud
    
    with open('config_interfaz.txt', 'w') as json_file:
        json.dump(config_datos, json_file)
    
def cambio_caracteriticas_imagen(img):
    '''
    Función encargada de modificar las características de la imágen
    en los fichero de configiración.
    '''

    global X, Y, centro, R
    
    X, Y, canales = img.shape
    centro = (int(Y/2), int(X/2))
    R = int(Y/2)
    
    # Se actualizan los datos en el fichero json de configuración
    config_datos['Imagen']['X'] = X
    config_datos['Imagen']['Y'] = Y 
    
    with open('config_interfaz.txt', 'w') as json_file:
        json.dump(config_datos, json_file)
    
def rescalado_imagen(frame, percent=75):
    '''
    Función encargada de reducir e tamaño de la imágen,
    con el fin de que esta se pueda observar en pantalla.
    '''
    
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

def conecta_camara():
    '''
    Función encargada de establecer la conexión con la cámara.
    '''
    
    global cam
    
    try:
        return cam
    except:
        ret = cv2.VideoCapture(direccion_camara)
        if ret.isOpened():
            cam = ret
            return cam
        else:
            print('Error en la conexión con  la cámara.')
            return -1

def toma_imagen():
    '''
    Devuelve la imagen obtenida de la cámara.

    '''
    conecta_camara()
    ret, frame = cam.read()
    
    if ret == True:
        return frame
    
    else:
        print('Error en la comunicación de la cámara.')
        return False

##########################################################################################
#######                                                                           ########
#######                      FUNCIONES INTERNAS DE LA APLIACIÓN                   ########
#######                                                                           ########
##########################################################################################

def cv2plt(img_cv):
    '''
    Función que convierte una imagen de la librería de OpenCV en
    el formato de la librería Matplotlib
    '''
    
    # Modelo de color OpenCV: BGR
    # Modelo de color de matplotlib: RGB
    
    img_plt = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_plt

def plt2cv(img_plt):
    
    '''
    Función que convierte una imagen de la librería de Matplotlib en
    el formato de la librería OpenCV
    '''
    
    # Modelo de color OpenCV: BGR
    # Modelo de color de matplotlib: RGB
    
    img_cv = cv2.cvtColor(img_plt, cv2.COLOR_RGB2BGR)
    return img_cv
    
def mascara_inicial():
    '''
    Función encargada de obtener la máscara que se ha de aplicar a la imágen
    en bruto para corregir los efectos de la lente.
    '''
    
    global img_patron
    mask = img_patron.copy()
    X, Y, colores = mask.shape
    Y = mask.shape[1]
    centro = (int(Y/2), int(X/2))
    R = int(Y/2)

    cv2.circle(mask, centro, R, color=(255,255,255), thickness=-1)
    
    # Conversión a escala de grises
    mask = np.uint8(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    mask = cv2.inRange(mask, 255, 255)


    # masked_data = cv2.bitwise_and(img_patron, img_patron, mask=mask)
    # plt.imshow(masked_data)
    
    return mask

 
def mascara_cielo():
    '''
    Función encargada de obtener la máscara que se ha de aplicar a la imágen
    en bruto para obtener la zona de cielo potencialmente visible
    '''
    global cielo
    
    try:
        return cielo
    except:
        
        global img_patron
        img_proc = cv2.bitwise_and(img_patron, img_patron, mask=mascara_inicial())
        
        blue = cv2.inRange(img_proc[:,:,0], 0, 130)
        # blue = cv2.medianBlur(blue, 5)
        
        green = cv2.inRange(img_proc[:,:,1], 0, 110)
        # green = cv2.medianBlur(green, 5)
        
        red = cv2.inRange(img_proc[:,:,2], 0, 60)
        # red = cv2.medianBlur(red, 5)
        
        mask_nocielo = cv2.bitwise_or(blue, green)
        mask_nocielo = cv2.bitwise_or(mask_nocielo, red)
        # mask_nocielo = cv2.medianBlur(mask_nocielo, 5)
    
        cielo = cv2.bitwise_not(np.uint8(mask_nocielo))
        # masked_data = cv2.bitwise_and(img_patron, img_patron, mask=mask_nocielo)
        
        # plt.figure(1)
        # plt.imshow(blue)
        # plt.figure(2)      
        # plt.imshow(green)
        # plt.figure(3)      
        # plt.imshow(red)
    
        return cielo

# Se define la máscara de cielo potencialmente visible
cielo = mascara_cielo()

def pre_procesado(img_bgr):
    '''
    Realiza un procesado de la imagen original, para obtener la imagen 
    con la que se trabajará.

    '''
    
    img_proc = cv2.bitwise_and(img_bgr, img_bgr, mask=cielo)
    return img_proc
    
def hora_imagen(ruta_imagen):
    '''
    Funcion que obtiene la hora de la imagen a partir de la ruta de imagen.
    '''
    
    nombre_fichero = ruta_imagen.split('\\')[-1]
    fecha_formato = nombre_fichero.split('_')[-1]
    hora_str = fecha_formato.split('.')[0]
    hora = dt.strptime(hora_str, '%Y-%m-%d-%Hh%Mmin')

    return hora

def f_area_zenith():
    
    '''
    Devuelve la función obtenida de computar los datos de los errores
    de las áreas del sol, en función del ratio (ángulo zenit)/90.
    
    '''
    
    global p_areas
    try:
        return p_areas
    
    except:
        areas = pd.read_csv(ruta_datos_areas)
        
        # Área del disco solar en función del zenith
        x_zenith = areas['Zenith']
        y_areas = areas['Area']
        z_areas = np.polyfit(x_zenith, y_areas, 8)
        p_areas = np.poly1d(z_areas)
        
        return p_areas

def f_error_azimuth():
    '''
    Devuelve la función obtenida de computar los datos de los errores
    del ángulo azimut, en función del ángulo azimutal real.
    
    '''
    
    global f_e_azimuth
    try: 
        return f_e_azimuth
    
    except:
        error = pd.read_csv(ruta_datos_errores)
        
        # Azimuth
        x_e_azimuth = error['Azimuth']
        y_e_azimuth = error['E_Azimuth']
        z_e_azimuth = np.polyfit(x_e_azimuth, y_e_azimuth, 8)
        f_e_azimuth = np.poly1d(z_e_azimuth)
        
        return f_e_azimuth

def f_error_zenith():
    '''
    Devuelve la función obtenida de computar los datos de los errores
    del ángulo zenit, en función del ángulo azimutal real.
    
    '''
    
    global f_e_zenith
    try:
        return f_e_zenith
    
    except:
        error = pd.read_csv(ruta_datos_errores)
        
        # Zenith
        x_e_zenith = error['Azimuth']
        y_e_zenith = error['E_Zenith']
        z_e_zenith = np.polyfit(x_e_zenith, y_e_zenith, 8)
        f_e_zenith = np.poly1d(z_e_zenith)
        
        return f_e_zenith

def f_rad_difusa_nublado_ghi():
    '''
    Devuelve la función obtenida de computar los datos para la estimación
    de la rediación difusa en cielos nublados, a partir de la radiación global estimada.
    '''
    
    global regr_nublado_ghi
    try:
        return regr_nublado_ghi
    
    except:
        # Lectura del dataframe con las variables:
        pd_info_regresion_nublado = pd.read_csv(ruta_datos_regresion_nublado)
       
        grados = 1
        regr_nublado_ghi = Pipeline([('poly', PolynomialFeatures(degree=grados)),
                          ('linear', LinearRegression())])
    
        y_rad_nublado = pd_info_regresion_nublado['Radiacion difusa']
        x_rad_nublado = pd_info_regresion_nublado[['ghi', 'Ratio nubes', 'Intensidad nubes']]
    
        regr_nublado_ghi.fit(x_rad_nublado, y_rad_nublado)
        return regr_nublado_ghi
    
def f_rad_difusa_nublado():
    '''
    Devuelve la función obtenida de computar los datos para la estimación
    de la rediación difusa en cielos nublados, a partir de la radiación global medida.
    '''
    
    global regr_nublado
    try:
        return regr_nublado
    
    except:
        # Lectura del dataframe con las variables:
        pd_info_regresion_nublado = pd.read_csv(ruta_datos_regresion_nublado)
       
        grados = 1
        regr_nublado = Pipeline([('poly', PolynomialFeatures(degree=grados)),
                          ('linear', LinearRegression())])
    
        y_rad_nublado = pd_info_regresion_nublado['Radiacion difusa']
        x_rad_nublado = pd_info_regresion_nublado[['Radiacion', 'Ratio nubes', 'Intensidad nubes']]
    
        regr_nublado.fit(x_rad_nublado, y_rad_nublado)
        return regr_nublado
        
def f_rad_difusa_despejado():
    '''
    Devuelve la función obtenida de computar los datos para la estimación
    de la rediación difusa en cielos despejados.
    '''
    
    global regr_despejado
    try:
        return regr_despejado
    
    except:
        # Lectura del dataframe con las variables:
        pd_info_regresion_despejado = pd.read_csv(ruta_datos_regresion_despejado)
        
        grados = 1
        regr_despejado = Pipeline([('poly', PolynomialFeatures(degree=grados)),
                          ('linear', LinearRegression())])
                          
        y_rad_despejado = pd_info_regresion_despejado['Radiacion difusa']
        x_rad_despejado = pd_info_regresion_despejado[['Radiacion', 'Intensidad cielo']]
    
        regr_despejado.fit(x_rad_despejado, y_rad_despejado)
        return regr_despejado

def rad_gh_teorica(hora):
    '''
    Función encaragda de devolver la radiación global horizontal téórica
    en la posición en la que se encuentra la cámara.
    '''
    
    times = pd.DatetimeIndex([hora], tz='Europe/Madrid')
    pd_rad_teorica = Cam_Location.get_clearsky(times, model='haurwitz')
    ghi = pd_rad_teorica['ghi'].to_list()[0]
    
    return ghi

def pixel_zenith(punto):
    '''
    Devuelve el ángulo cenital en coordenadas de la imagen
    del pixel que se indique
    '''
   
    delta = np.array(punto) - np.array([centro[1], centro[0]])
    radio_imagen = int(np.linalg.norm(delta))

    R = int(Y/2)
    zenith_imagen = radio_imagen / R * 90
    
    return zenith_imagen

def pixel_azimuth(punto):
    '''
    Devuelve el ángulo azimutal en coordenadas de la imagen
    del pixel que se indique
    '''
    
    delta = np.array(punto) - np.array([centro[1], centro[0]])
    delta_x = delta[0]
    delta_y = delta[1]

    # Cáculo del ángulo azimutal
    alpha = np.rad2deg(np.arctan(delta_x / delta_y))
    
    if alpha <= 0: # Segundo o cuatro cuadrante
        if delta_x <= 0: # Cuarto cuadrante
            alpha = alpha
        else: # Segundo cuatrante
            alpha += 180
    else: # Primer o tercer cuadrante
        if delta_x >= 0: # Primer cuadrante
            alpha = alpha
        else: # Tercer cuadrante
            alpha += 180

    azimuth_real_imagen = 270 - alpha
    
    return azimuth_real_imagen
    
def matriz_posiciones():
    '''
    Devuelve una matriz, del tamaño de la imagen, con dos valores por cada elemento,
    siendo el primer valor el zenit y el segundo el azimut, valores correspondientes a la imagen, 
    no se encuentran corregidos
    '''
    
    global pixels_pos
    
    try:
        return pixels_pos
        
    except:
        pixel_pos = lambda pos: (pixel_zenith(pos), pixel_azimuth(pos))
        pixels_pos = np.array([[pixel_pos([x,y]) for y in range(Y)] for x in range(X)])
        
        return pixels_pos

def pixel_dist(punto, pnt_centro):
    '''
    Devuelve la distancia de un píxeles a otro,
    en valor de píxeles.
    '''
    
    delta = np.array(punto) - np.array(pnt_centro)
    dist = int(np.linalg.norm(delta))

    return dist
    
def matriz_distancias(punto):
    '''
    Devuelve una matriz, del tamaño de la imagen, con la distancia de cada píxel
    al punto especificado como parámetro.
    '''

    pixels_dist = np.array([[pixel_dist([y,x], punto) for y in range(Y)] for x in range(X)])
    
    return pixels_dist

##########################################################################################
#######                                                                           ########
#######                FUNCIONES INTERMEDIAS PARA EL USUARIO                      ########
#######                                                                           ########
##########################################################################################

def mascara_nubes(img_bgr, centroide=None):
    '''
    Función que genera la máscara de nubes de una imágen utilizando el método propio propuesto.

    Parámetros
    ----------
    img_bgr :
        Imagen original.
    centroide : list, (Y,X)
        Posición del sol. 
        En el caso de que se indique, se eliminará
        el contorno que encierre al centroide,
        según el algoritmo propuesto.

    Devuelve
    -------
    mask_cloud
    '''

    # Separación de los canales de la imagen:
    Blue = img_bgr[:,:,0]
    Green = img_bgr[:,:,1]
    Red = img_bgr[:,:,2]
        
    # Método propio en base a las máscaras de los canales
    mask_green = cv2.inRange(Green, 0, 140)
    mask_red = cv2.inRange(Red, 0, 70)
    mask_R_B = cv2.inRange(Red - Blue, 90, 255)
    mask_B_G = cv2.inRange(Blue - Green, 30, 255)
    
    # Creación de las máscaras de cielo despejado
    mask_sky = cv2.bitwise_or(mask_R_B, mask_green, mask=mask_B_G)
    mask_sky = cv2.bitwise_or(mask_sky, mask_red, mask=mask_B_G)
    mask_sky = cv2.bitwise_and(mask_sky, mask_sky, mask=cielo)
    
    # Se crea la máscara de las nubes negando la máscara del cielo despejado
    mask_cloud = cv2.bitwise_not(mask_sky)
    mask_cloud = cv2.bitwise_and(mask_cloud, mask_cloud, mask=cielo)
    
    # Si se indica centroide, se elimina la falsa nube que se puede detectar en casos en los que el sol sature la cámara
    if centroide != None:
        # Se obtiene los contornos de las nubes
        mask_cloud = cv2.medianBlur(mask_cloud.astype(np.uint8),5)
        cnt_nubes, hierarchy = cv2.findContours(mask_cloud, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Se obtiene el contorno de nube que encierra al centroide:
        for i in cnt_nubes:
            if cv2.pointPolygonTest(i, centroide, False) >= 0:
                cnt_sol = i
        
        # Se obtiene la máscara en la que se encuentra la "sombra" del sol, con el fin de quitar esta zona de la máscara de nubes
        # f_mask_sol = lambda pos: 255 if (cv2.pointPolygonTest(cnt_sol, pos, False) >= 0) else 0
        # mask_sol = [[f_mask_sol((y,x)) for y in range(Y)] for x in range(X)]
        # mask_sol = np.array(mask_sol, np.uint8)
        f_mask_sol = lambda x,y: 255 if (cv2.pointPolygonTest(cnt_sol, (y,x), False) >= 0) else 0
        mask_sol = np.fromfunction(np.vectorize(f_mask_sol), (X,Y), dtype=int).astype(np.uint8)
        
        mask_no_sol = cv2.bitwise_not(mask_sol)
        mask_cloud = cv2.bitwise_and(mask_cloud, mask_no_sol)
    
    return mask_cloud
    
def mascara_cielo_visible(mask_nubes, mask_cielo=cielo):
    '''
    Esta función devuelve la máscara de cielo visible a partir 
    de una mácara de nubes.
    
    '''
    
    mask_cielo_visible = cv2.bitwise_not(mask_nubes, mask=mask_cielo)
    return mask_cielo_visible

def mascara_contorno_nubes(mask_nubes):
    '''
    Se obtiene los contornos de la máscara de nubes.
    '''

    cnt_nubes, hierarchy = cv2.findContours(mask_nubes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_nubes_cnt = np.zeros((X,Y), np.uint8)
    cv2.drawContours(mask_nubes_cnt, cnt_nubes, -1, color=255, thickness=10)
    
    return mask_nubes_cnt
    

def porcion_nubes_cielo(mask_nubes, mask_cielo=cielo):
    '''
    Función que calcula el tanto por uno de nubes del cielo,
    dada una máscara de nubes.

    Parameters
    ----------
    mask_nubes : máscaras con las nubes del cielo, 
        previamente creada.

    Returns
    -------
    ratio_nubes, en tanto por uno
        float
    '''
    
    # Cálculo del área de la imagen que corresponde a cielo visible por la cámara
    area_cielo = cv2.countNonZero(mask_cielo)

    # Cálculo del área del cielo cubierta por nubes
    area_nubes = cv2.countNonZero(mask_nubes)
    
    ratio_nubes = area_nubes / area_cielo
    return ratio_nubes

def pos_solar(hora, imagen=False):
    '''
    Se obtiene las posición solar, definida por los ángulos zenit y azimut, teórica en un determinado momento.
    Si imagen == True, se realizan las correcciones necesarias para obtener el zenit y el azimut
    en la imagen.
    '''
    
    hora_timestamp = pd.Timestamp(hora, tz = 'Europe/Madrid')
    times = pd.DatetimeIndex([hora_timestamp], tz='Europe/Madrid')
    solar_pos = Cam_Location.get_solarposition(times)
    
    # Se obitne la posición del sol en el momento de la toma de la imágen
    zenith = solar_pos.loc[hora_timestamp]['zenith']
    azimuth = solar_pos.loc[hora_timestamp]['azimuth']
    
    # Correcciones necearias para obtener los ángulos en la imagen
    if imagen == True:
        # Para calibrar azimuth: antena en parte izq de la imagen: ángulo -> 187 en imagen, ángulo -> 100.52 en la realidad,
        # obtenido con Google Maps, mirar notas
        azimuth_imagen = azimuth + offset_azimuth
        
        # Corrección de los errores
        # Zenith: Zenith_imagen = Error_Zenith + Zenith
        zenith += f_error_zenith()(azimuth)
        
        # Azimuth: Azimuth_imagen = Error_Azimuth + Azimuth
        azimuth_imagen += f_error_azimuth()(azimuth)
        
        return zenith, azimuth_imagen
    
    return zenith, azimuth

def centroide_solar(hora, imagen=False):
    '''
    Se obtiene las posición solar, definida por el centroide solar en coordendas de la imagen, 
    teórica en un determinado momento.
    
    Si imagen == True, se realizan las correcciones necesarias para obtener el zenit y el azimut
    en la imagen.
    '''
    
    zenith, azimuth_imagen = pos_solar(hora, imagen)
    
    # Transformación de coordenadas realies a coordenadas de la imágen de la posición del sol
    radio = int(zenith / 90 * R)
    pnt_y = int(centro[0] + radio * np.cos(azimuth_imagen / 180 * np.pi))
    pnt_x = int(centro[1] - radio * np.sin(azimuth_imagen / 180 * np.pi))

    # Centroide:
    centroide = (pnt_y, pnt_x)

    return centroide
    
def salida_puesta_sol(dia_img):
    '''
    Función encargada de obtener la hora de salida y puesta del sol
    en el día indicado.
    '''
    
    fechas = []
    # Se obtiene la hora de la puesta y salida de sol del día de hoy
    dia = dia_img.strftime('%Y-%m-%d %H:%M:%S')
    fechas.append(dia)
    time = pd.DatetimeIndex(fechas, tz='Europe/Madrid')
    info = Cam_Location.get_sun_rise_set_transit(time)
    
    # En formato str...
    sunrise = info['sunrise'].loc[fechas[0]].strftime('%Y-%m-%d %H:%M:%S')
    sunset = info['sunset'].loc[fechas[0]].strftime('%Y-%m-%d %H:%M:%S')
    
    # Se obtiene la hora actual, y la puesta/salida del sol, en formato Datetime
    salida_sol = dt.strptime(sunrise,'%Y-%m-%d %H:%M:%S')# +- timedelta(hours=1)
    puesta_sol = dt.strptime(sunset,'%Y-%m-%d %H:%M:%S')# +- timedelta(hours=1)
    
    # Comprueba si la hora actual se encuentra entre la salida y la puesta del sol
    return salida_sol, puesta_sol
    
def mascara_solar(img_bgr):
    '''
    Devuelve la máscara del sol, obtenida de la imagen, mediante el método basado en 
    el modelo de color HLS.
    '''
    
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    L = img_hls[:,:,1]

    mask_hls = cv2.inRange(L, 240, 255)

    return mask_hls

def sol_cubierto(img_bgr, hora):
    '''
    Se calcula el factor de sol cubierto por nubes.
    
    Supuestos:
        * Circularidad alta y área solar relativamente alta: sol descubierto, factor --> 1
        * Circularidad alta y área solar relativamente baja: sol cubierto por nubes densas, factor --> 0.2
        * Circularidad alta, resto de casos: sol cubierto por nubes muy poco densas, factor --> 0.85
        * Circularidad baja y área solar relativamente alta: sol cubierto por nubes de baja densidad, factor --> 0.75
        * Circularidad baja y área solar relativamente baja: sol cubierto por nubes de alta densidad, factor --> 0.25
        * Circularidad muy baja, área solar muy alta o muy baja: sol totalmente cubierto, lo que se observa son los reflejos de
            las nubes, factor --> 0, se procederá a calcular el centroide por medio de la librería pvlib

    Parameters
    ----------
    img_bgr

    Returns
    -------
    factor_sol_cubierto : float
    '''
    
    # Se obtiene la máscara del sol
    mask_hls = mascara_solar(img_bgr)
    
    # Se obtiene los contornos de lá máscara HLS para obtener la posición del sol
    mask_hls_cnt = cv2.medianBlur(mask_hls.astype(np.uint8),3)
    cnt, hierarchy = cv2.findContours(mask_hls_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Se obtiene el contorno de mayor area y se calcula su circularidad, C
    # Se ha supuesto que el área de mayor tamaño corresponde al sol
    A = 0.0
    if len(cnt) > 0:
        for j in cnt:
            temp = cv2.contourArea(j)
            if temp >= A:
                A = temp
                max_cnt = j
        
        # Cálculo del perímetro del sol, con el fin de calcular su circularidad
        P = cv2.arcLength(max_cnt, True)
        if P > 0:
            C = 4 * np.pi * A / P**2
        else:
            return 0.0
        
    else: # No se ha detectado el sol
        return 0.0
    
    # Se obtiene el ratio zenit/90, para la estimación del área solar
    zenith, azimuth = pos_solar(hora, imagen=True)
    zenith /= 90 
    
    # Circularidad alta
    if (C > 0.65): 
        # Área dentro de rango
        if (abs(A - f_area_zenith()(zenith))) < (0.2 * A): # Sol totalmente descubierto
            factor_sol_cubierto = 1
        # Área muy inferior a la esperada
        elif A < (0.6 * f_area_zenith()(zenith)): # Sol cubierto por nubes densas
            factor_sol_cubierto = 0.2
        # Área algo inferior o superior a la esperada
        else:
            factor_sol_cubierto = 0.85 # Sol cubierto por nubes muy poco densas
    # Circularidad relativamente baja
    elif C > 0.35:
        # Área dentro del rang esperado
        if (abs(A - f_area_zenith()(zenith)) < (0.25 * A)): # Sol cubierto por nubes poco densas
            factor_sol_cubierto = 0.75
        # Área fuera del rango esperado
        else:
            factor_sol_cubierto = 0.25 # Sol cubierto por nubes densas
    # Circularidad muy baja
    else: # Sol no detectado o zonas muy dispersas, luz reflejada por las nubes
        factor_sol_cubierto = 0.0 # Sol totalmente cubierto
    
    if factor_sol_cubierto != 0: # Si el sol se ha detectado, se devuelve el centroide
       # Cálculo del centroide con los momentos del controne de mayor tamaño de la imágen
        moments = cv2.moments(max_cnt)
        # (y,x)
        if (moments['m00']>0) & (moments['m00']>0):
            centroide = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
        else:
            return 0.0
        
        return factor_sol_cubierto, centroide, A, C
    
    else:
        return factor_sol_cubierto

def area_circumsolar(img_bgr):
    '''
    Se obtiene el área del disco solar detectado en la imagen.
    '''
    
    # Se obtiene la máscara del sol
    mask_hls = mascara_solar(img_bgr)
    
    # Se obtiene los contornos de lá máscara HLS para obtener la posición del sol
    mask_hls_cnt = cv2.medianBlur(mask_hls.astype(np.uint8),3)
    cnt, hierarchy = cv2.findContours(mask_hls_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Se obtiene el contorno de mayor area y se calcula su circularidad, C
    # Se ha supuesto que el área de mayor tamaño corresponde al sol
    A = 0.0
    if cnt != None:
        for j in cnt:
            temp = cv2.contourArea(j)
            if temp > A:
                A = temp
    return A

def gamma_corr(img_bgr, gamma=2.2, maxVal=255):
    '''
    La función devuelve una copia de la imagen a la que se le ha aplicado
    la correción gamma, indicando el valor gamma de la imagen y el valor máximo de los píxeles de la imágen.
    '''
    
    gamma_corr = 1 / gamma
    # Se normaliza la matriz con las imágenes
    img_norm = img_bgr / maxVal
    # Se obtiene la imagen en color corregida y desnormalizada
    img_corr = np.power(img_norm, gamma_corr) * maxVal

    return np.array(img_corr, np.uint8)

def intensidad_equiponderada(img_bgr):
    '''
    Cálculo de la luminancia del pixeles, como media de los canales B, G y R
    '''
    
    B = img_bgr[:,:,0]
    G = img_bgr[:,:,1]
    R = img_bgr[:,:,2]
    
    I = np.array(B/3 + G/3 + R/3, np.uint8)
    return I

    
def intensidad_CCIR(img_bgr):
    '''
    Cálculo de la luminancia del pixeles, Basado en el convenio CCIR 601 para imágenes analógicas
    '''
    
    B = img_bgr[:,:,0]
    G = img_bgr[:,:,1]
    R = img_bgr[:,:,2]

    I = np.array(B * 0.114 + G * 0.587 + R * 0.299, np.uint8)
    return I 
    
def intensidad_ITUR(img_bgr):
    '''
    Cálculo de la luminancia del pixeles, Bbsado en el convenio ITU-R BT 709 para imágenes digitales
    '''
    
    B = img_bgr[:,:,0]
    G = img_bgr[:,:,1]
    R = img_bgr[:,:,2]

    I = np.array(B * 0.0722 + G * 0.7152 + R * 0.2126, np.uint8)
    return I 

def intensidad_media(img_bgr, mask=None, modo='ITUR', coseno=False, gamma=False):
    '''
    Cálculo de la intensidad media de los píxeles de una imagen.

    Parámetros
    ----------
    img_bgr : np.array
        Imágen BGR.
    mask : np.array, opcional
        Mácara a la que aplicar el cálaculo de intensidad media.
    modo : string, Equiponderado, CCIR o ITUR
        Indica el método que se utilzará para la calcular la luminancia de los píxeles.
    coseno : bool, opcional
        Indica si se aplica a la matriz de luminania el coseno del ángulo cenital al que corresponden.
    gamma : float, opcional
        Se indica el factor gamma de la imagen, con el objetivo de realizar la corrección gamma.
    '''
    
    modos = {'Equiponderado': intensidad_equiponderada,
            'CCIR': intensidad_CCIR,
            'ITUR': intensidad_ITUR}
            
    I = modos[modo](img_bgr)
    
    # Corrección gamma
    if gamma:
        I = gamma_corr(I, gamma)
    
    # Multiplicación de la matriz de luminancia por la matriz de los ángulos cenitales
    if coseno:
        zenits = matriz_posiciones()[:,:,0]
        I = np.multiply(I, np.cos(np.deg2rad(zenits)))
    
    # Se obtiene el valor medio de la intensidad de la imagen
    media = cv2.mean(I, mask)[0]
    
    return media

def intensidad_acumulada(img_bgr, mask_parcial, mask_total=cielo, modo='ITUR', coseno=False, gamma=False):
    '''
    Cálculo de la intensidad acumulada en una porción de la imágen respecto de
    una máscara que la encierra.

    Parámetros
    ----------
    img_bgr : np.array
        Imágen BGR.
    mask_parcial : np.array, opcional
        Máscara de la que se queire obtener la porción de intensidad que se observa.
    mask_total : np.array, opcional
        Máscara que encierra a la parcial.
    modo : string, Equiponderado, CCIR o ITUR
        Indica el método que se utilzará para la calcular la luminancia de los píxeles.
    coseno : bool, opcional
        Indica si se aplica a la matriz de luminania el coseno del ángulo cenital al que corresponden.
    gamma : float, opcional
        Se indica el factor gamma de la imagen, con el objetivo de realizar la corrección gamma.
    '''
    
    modos = {'Equiponderado': intensidad_equiponderada,
            'CCIR': intensidad_CCIR,
            'ITUR': intensidad_ITUR}
            
    I = modos[modo](img_bgr)
    
    # Corrección gamma
    if gamma:
        I = gamma_corr(I, gamma)
    
    # Multiplicación de la matriz de luminancia por la matriz de los ángulos cenitales
    if coseno:
        zenits = matriz_posiciones()[:,:,0]
        I = np.multiply(I, np.cos(np.deg2rad(zenits)))
    
    # Se obtiene la matriz con las intensidades de los píxeles de cada región
    I_parcial = cv2.bitwise_and(I, I, mask=mask_parcial)
    I_total = cv2.bitwise_and(I, I, mask=mask_total)
    
    intensidad_acumulada_parcial = np.sum(I_parcial)
    intensidad_acumulada_total = np.sum(I_total)
    
    ratio = intensidad_acumulada_parcial / intensidad_acumulada_total
    
    return ratio

def porcion_mascaras(mask_parcial, mask_total):
    '''
    Función que calcula el tanto por uno de relación entre los píxeles
    de una máscara y otra que la contiene
    '''
    
    # Cálculo del área de la imagen que corresponde a cielo visible por la cámara
    area_total = cv2.countNonZero(mask_total)

    # Cálculo del área del cielo cubierta por nubes
    area_parcial = cv2.countNonZero(mask_parcial)
    
    ratio_acumulado = area_parcial / area_total
    return ratio_acumulado

def muestra_imagen(img_cv2):
    '''
    Función encargada de la muestra en pantalla de una imágen.
    '''
    
    img_plt = cv2plt(img_cv2)
    plt.imshow(img_plt)

def dibujo_puntos_cardinales(img):
    '''
    Devuelve una copia de la imágen en la que se han dibujado los puntos cardinales.
    '''
    
    img_bgr = img.copy()
        
    # Lista con los azimuths de los puntos cardinales
    azimuths = [(lambda x: 90*x)(i) for i in range(4)]
    
    # Norte: azul, Sur: rojo, Este:morado, Oeste:Verde
    color = {0:(255, 0, 0), 90:(125,0,125), 180:(0,0,255), 270:(0,125,0)}
    texto = {0:'N', 90:'E', 180:'S', 270:'W'}
    
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 5
    
    for azimuth in azimuths:
        e_azimuth = f_error_azimuth()(azimuth)
        azimuth_imagen = azimuth + e_azimuth
        azimuth_imagen = azimuth + offset_azimuth
        
        # Transformación de coordenadas realies a coordenadas de la imágen de la posición del sol
        pnt_x = int(centro[0] + R * np.cos(azimuth_imagen / 180 * np.pi))
        pnt_y = int(centro[1] - R * np.sin(azimuth_imagen / 180 * np.pi))
    
        # Dibujo del Azimuth
        cv2.line(img_bgr, centro, (pnt_x, pnt_y), color=color[azimuth], thickness=10)
        
        
        # Nombre del punto cardinal
        if azimuth == 0:
            r = int(X*0.45)
            alpha = azimuth_imagen - 2
        elif azimuth == 90:
            r = int(Y*0.47)
            alpha = azimuth_imagen - 2
        elif azimuth == 180:
            r = int(X*0.48)
            alpha = azimuth_imagen + 2
        else:
            r = int(Y*0.45)
            alpha = azimuth_imagen
            
        pnt_x = int(centro[0] + r * np.cos(alpha / 180 * np.pi))
        pnt_y = int(centro[1] - r * np.sin(alpha / 180 * np.pi))
        cv2.putText(img_bgr, texto[azimuth], (pnt_x, pnt_y), fuente, escala, color[azimuth], thickness=20)
        
    return img_bgr
    
def dibujo_camino_sol(img_bgr, hora_img=dt.now(), delta_=timedelta(hours=2), color_=(0,255,0)):
    '''
    Devuelve una copia de la imágen en la que se ha dibujado el camino del sol en el día especificado.
    '''    
  
    # Se preoaran las variables auxiliares
    img_bgr_copy = img_bgr.copy()
    lista_horas = []
    delta = timedelta(minutes=3)
    dia_img = hora_img.date()
    
    # Se obtiene las horas de puesta y salida del sol en el día de la toma de la imágen
    hora_salida_sol, hora_puesta_sol = salida_puesta_sol(dia_img)
    hora = hora_salida_sol + delta_
    
    # Se obtiene una lista con las instantes en los que dibujar los centroides solares
    while hora < hora_puesta_sol - delta_:
        hora += delta
        lista_horas.append(hora)
    
    # Se recorre la lista de horas, dibujando en la imñagen el centroide solar en ese instante
    for hora in lista_horas:
        centroide_img = centroide_solar(hora, imagen=True)
        cv2.circle(img_bgr_copy, centroide_img, 2, color=color_, thickness=20)
        
        
    return img_bgr_copy

def mascara_ang_solido(rango, metodo, centroide=None, hora=None):
    '''
    Función encargada de obtener la máscara de una porción del cielo, de la que se obtendrá
    la radiación solar difusa percibida.

    Parámetros
    ----------
    rango : list
        Se definen el rango en el que se tiene que encontrar los valores que definen el método utilizado.
    metodo : 'angulos' o 'distancia'
        Se escoge el método que se desea utiliza para la otención de la máscara.
        
        'angulos': basado en los ángulos zenit y azimut, en este caso la varible rango debe ser una matriz de 2x2.
        En la primera fila se definirá en rango del zenit y en la segunda en rango del azimut.
        Por ejemplo, se desea obtener la zona de cielo que se encuentra entre el ángulo zenit 20 y 80 y entre el ángulo 
            azimut 190 y 270: 
                rango = [[20, 80], [190, 270]]
                
        'distancia': basado en la distancia al centroide solar, la variable rango corresponde a los valores entre los 
        que se desea obtener la máscara.
        Por ejemplo: se desea observar la zona del cielo que se encuentra entre 400px y 1000px del centroide solar:
            rango = [400, 1000]
        
    centroide y hora:
        Es necesario indicar uno de estos valores en el caso de que se utilize el método 'distancia'.   
    '''
    
    if metodo == 'angulos':
        matriz_posicion = matriz_posiciones()
        pixels_zeniths = matriz_posicion[:,:,0]
        pixels_azimuths = matriz_posicion[:,:,1]
        
        mask_Z = cv2.inRange(pixels_zeniths, rango[0][0], rango[0][1])
        mask_A = cv2.inRange(pixels_azimuths, rango[1][0], rango[1][1])
        mask_ang_sol = cv2.bitwise_and(mask_Z, mask_A)
    
    elif metodo == 'distancia':
        if centroide is None:
            centroide = centroide_solar(hora, imagen=True)
            
        matriz_distan_centroide = matriz_distancias(centroide)
        mask_ang_sol = cv2.inRange(matriz_distan_centroide, rango[0], rango[1])
        
    else:
        mask_ang_sol = None
        
    return mask_ang_sol

def estimar_radiacion_sector(radiacion_total, mask_sector_parcial, mask_sector_total, tipo_cielo='despejado', params=None):
    '''
    Función encargada de estimar la radiación percibida en un sector del cielo.

    Parameters
    ----------
    radiacion_total : float
        Radiación difusa total percibida.
    mask_sector_parcial : np.array
        Máscara del secror en el que se desea obtener la radiación difusa percibida
    mask_sector_total : np.array
        Máscara de nubes en el caso de cielo nublado o de cielo libre de nubes en el caso de cielo despejado
    tipo_cielo : string, 'despejado' o 'nublado'
        Indica el método que se desea utilizar.
    params : list, opcional
        Parámetros adicionales utilizado en el caso de cielos nublados:
            params = [img, modo, coseno, gamma]
            img: imagen del 
            modo: modo en el que se desea calcular la intensidad de los píxeles de la imagen. ITUR, CCIR o Equiponderado
            coseno: True o False. Indicando si se aplica o no la matriz de coseno del ángulo cenital.
            gamma: factor gamma, en caso de que se quiera aplicar la corrección gamma, en caso contrario se debe indicar el valor None
    '''
    
    if tipo_cielo == 'despejado':
        ratio = porcion_mascaras(mask_sector_parcial, mask_sector_total)
        
    elif tipo_cielo == 'nublado':
        img = params[0]
        modo_ = params[1]
        coseno_ = params[2]
        gamma_ = params[3]
        
        ratio = intensidad_acumulada(img, mask_sector_parcial, mask_sector_total, modo=modo_, coseno=coseno_, gamma=gamma_)
        
    radiacion_estimada = ratio * radiacion_total
    
    return radiacion_estimada

def rad_difusa(radiacion, ratio_nubes, factor_solar, intens_media, cielo='despejado', tipo_radiacion='teorica'):
    '''
    Devuelve la radiación difusa estimada, obtenida con los parámetros indicados.

    Parámetros
    ----------
    radiacion : float
        La radiacion global medida o la radiación global teórica, dependiendo del método utilizado.
    ratio_nubes : float
        El ratio de nubes obtenido de la imagen del cielo..
    factor_solar : float
        El factor solar obtenido de la imagen del cielo.
    intens_media : float
        Intenisdad media de los píxeles de la imagen, aplicando los distintos métodos en función del tipo de cielo.
    cielo : string. Despejado o nublado.
        Indica el tipo de cielo para el que se quiere obtener la radiación.
    tipo_radiacion: string. Estimada o real.
        Indica si el valor de la variable "radiacion" corresponde con la ghi medida por los piranómetros o 
        con la ghi estimada por la librería PVLib.
        * Se puede observar que en el caso de cielos despejados esta variable afecta al modelo utilizado.   
    '''

    f_rad_difusa = {'nublado': {'real':f_rad_difusa_nublado, 'teorica': f_rad_difusa_nublado_ghi},
                    'despejado': {'real':f_rad_difusa_despejado, 'teorica': f_rad_difusa_despejado}}
                    
    regr_difusa = f_rad_difusa[cielo][tipo_radiacion]()
    
    if cielo == 'despejado':
        # intens_media *= (1 - ratio_nubes)
        rad = regr_difusa.predict([[radiacion, intens_media]])[0]
    else:
        # intens_media *= ratio_nubes
        rad = regr_difusa.predict([[radiacion, ratio_nubes, intens_media]])[0]
    
    return rad


##########################################################################################
#######                                                                           ########
#######                FUNCIONES DISPONIBLES PARA EL USUARIO                      ########
#######                                                                           ########
##########################################################################################

def muestra_imagen_cielo():
    '''
    Muestra la imagen de la cámara en tiempo real,
    para salir presionar la tecla
    '''
    
    while True:
        
        imagen = toma_imagen()
        imagen = pre_procesado(imagen)
        
        img_proc = rescalado_imagen(imagen, 40)
        cv2.imshow('Imagen del cielo en tiempo real', img_proc)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
def muestra_camino_sol(img_bgr=None, hora=dt.now()):
    '''
    Muestra la imagen de la cámara en tiempo real,
    sobre ella se dibuja el camino teórico del sol.
    '''
    
    while True:
        if img_bgr is None:
            imagen = toma_imagen()
        img_proc = dibujo_camino_sol(imagen, hora)
        img_proc = pre_procesado(img_proc)
        
        img_proc = rescalado_imagen(img_proc, 40)
        cv2.imshow('Camino solar', img_proc)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows() 

def muestra_puntos_cardinales(img_bgr=None):
    '''
    Muestra la imagen de la cámara en tiempo real,
    sobre ella se dibujan los cuatro puntos cardinales.
    '''
    
    while True:
        if img_bgr is None:
            imagen = toma_imagen()
        imagen = dibujo_puntos_cardinales(imagen)
        imagen = pre_procesado(imagen)
        
        img_proc = rescalado_imagen(imagen, 40)
        cv2.imshow('Puntos cardinales', img_proc)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()  
    
def muestra_cielo_visible(img_bgr=None):
    '''
    Muestra la imagen de la cámara en tiempo real,
    sobre ella se aplica la máscara que obtiene la parte de cielo visible.

    '''
    while True:
        if img_bgr is None:
            imagen = toma_imagen()
        imagen = pre_procesado(imagen)
        mask_nubes = mascara_nubes(imagen)
        
        img_proc = cv2.bitwise_and(imagen, imagen, mask=cv2.bitwise_not(mask_nubes, mask=cielo))
  
        img_proc = rescalado_imagen(img_proc, 40)
        cv2.imshow('Cielo visible', img_proc)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()  
    
def muestra_nubes(img_bgr=None):
    '''
    Muestra la imagen de la cámara en tiempo real,
    sobre ella se aplica la máscra de nubes obtenida.

    '''
    
    while True:
        if img_bgr is None:
            imagen = toma_imagen()
        imagen = pre_procesado(imagen)
        mask_nubes = mascara_nubes(imagen)
        
        img_proc = cv2.bitwise_and(imagen, imagen, mask=mask_nubes)
        
        img_proc = rescalado_imagen(img_proc, 40)
        cv2.imshow('Nubes', img_proc)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
def muestra_contorno_nubes(img_bgr=None):
    '''
    Muestra la imagen de la cámara en tiempo real,
    sobre ella se aplica la máscra de nubes obtenida.

    '''
    
    while True:
        if img_bgr is None:
            imagen = toma_imagen()
        imagen = pre_procesado(imagen)
        mask_nubes = mascara_nubes(imagen)
        
        # img_proc = cv2.bitwise_and(imagen, imagen, mask=mask_nubes)
  
        cnt_nubes, hierarchy = cv2.findContours(mask_nubes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imagen, cnt_nubes, -1, color=255, thickness=5)
        
        img_proc = rescalado_imagen(imagen, 40)
        cv2.imshow('Nubes', img_proc)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows() 

def muestra_centroide_solar():
    '''
    Muestra la imagen de la cámara en tiempo real,
    sobre ella se dibuja el centroide teórico del sol 
    en el momento de la imagen
    '''
    
    while True:
        imagen = toma_imagen()
        imagen = pre_procesado(imagen)
        
        centroide_img = centroide_solar(dt.now(), imagen=True)
        cv2.circle(imagen, centroide_img, 2, color=(0,0,255), thickness=20)
        
        img_proc = rescalado_imagen(imagen, 40)
        cv2.imshow('Centroide solar', img_proc)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
def estimar_radiacion_difusa():
    '''
    Devuelve la radiación difusa estimada en el momento actual,
    mediante el ánalisis de la imagen del cielo
    '''

    img_bgr = toma_imagen()
    img_bgr = pre_procesado(img_bgr)
    
    mask_nubes = mascara_nubes(img_bgr)
    ratio_nubes = porcion_nubes_cielo(mask_nubes)
    
    res_factor_solar = sol_cubierto(img_bgr, dt.now())
    if res_factor_solar != 0.0:
        factor_solar = res_factor_solar[0]
        centroide_img = res_factor_solar[1]
        area_circumsolar = res_factor_solar[2]
        circulridad_circumsolar = res_factor_solar[3]
    
    if factor_solar >= 0.85:
        mask_nubes = mascara_nubes(img_bgr, centroide_img)
        ratio_nubes = porcion_nubes_cielo(mask_nubes)
    
    if (ratio_nubes < 0.1) & (factor_solar > 0.75):
        tipo_cielo = 'despejado'
        intens_media = intensidad_media(img_bgr, coseno=True, gamma=2.2)/255
    else:
        tipo_cielo = 'nublado'
        intens_media = intensidad_media(img_bgr, gamma=2.2)/255

    try:
        import pygeonica as geo
        datos_estacion = geo.estacion.lee_canales(316)
        rad = datos_estacion[1]['PIRAN.1'][0]
        tipo_rad = 'real'
    except:
        rad = rad_gh_teorica(dt.now())
        tipo_rad = 'teorica'  
    finally:
        print(rad, ratio_nubes, factor_solar, intens_media, tipo_cielo)
        radiacion_difusa_estimada = rad_difusa(rad, ratio_nubes, factor_solar, intens_media, cielo=tipo_cielo, tipo_radiacion=tipo_rad)   
    
        return radiacion_difusa_estimada