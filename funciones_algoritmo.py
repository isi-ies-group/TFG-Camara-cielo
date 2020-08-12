# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:49:01 2020

@author: Martin
"""
import cv2
from mascaras import mascara_cielo
import numpy as np
from sklearn import linear_model
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import pvlib as pv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import sys
sys.path.append('../')

# Posición y clase de la cámara. PVLib
Latitud = 40.453472
Longitud = -3.727028
Altitud = 650
Cam_Location = pv.location.Location(Latitud, Longitud, tz='Europe/Madrid', altitude=Altitud)

offset_azimuth = 187 - 100.52

cielo = mascara_cielo()

# Variables relativas al tamaño de la imágen
X = 1944
Y = 2592
centro = (int(Y/2), int(X/2))
R = int(Y/2)

# Rutas globales
ruta_datos_areas = 'Datos/datos_areas.csv'
ruta_datos_errores = 'Datos/datos_errores.csv'
ruta_datos_regresion_nublado = 'Datos/info_regresion_nublado_completo.csv'
ruta_datos_regresion_despejado = 'Datos/info_regresion_despejado_completo.csv'

#%% FUNCIONES DEL ALGORITMO

def f_area_zenith():
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

def f_rad_difusa_nublado():
    global regr_nublado
    try:
        return regr_nublado
    
    except:
        # Lectura del dataframe con las variables:
        pd_info_regresion_nublado = pd.read_csv(ruta_datos_regresion_nublado)
       
        grados = 2
        regr_nublado = Pipeline([('poly', PolynomialFeatures(degree=grados)),
                          ('linear', LinearRegression())])
    
        y_rad_nublado = pd_info_regresion_nublado['Radiacion difusa']
        x_rad_nublado = pd_info_regresion_nublado[['ghi', 'Ratio nubes', 'Factor solar', 'Intensidad nubes ITUR Coseno:False Gamma_corr:True']]
    
        regr_nublado.fit(x_rad_nublado, y_rad_nublado)
        return regr_nublado
        
def f_rad_difusa_despejado():
    global regr_despejado
    try:
        return regr_despejado
    
    except:
        # Lectura del dataframe con las variables:
        pd_info_regresion_despejado = pd.read_csv(ruta_datos_regresion_despejado)
        
        grados = 2
        regr_despejado = Pipeline([('poly', PolynomialFeatures(degree=grados)),
                          ('linear', LinearRegression())])
                          
        y_rad_despejado = pd_info_regresion_despejado['Radiacion difusa']
        x_rad_despejado = pd_info_regresion_despejado[['Radiacion', 'Intensidad cielo ITUR Coseno:True Gamma_corr:True']]
    
        regr_despejado.fit(x_rad_despejado, y_rad_despejado)
        return regr_despejado
    
def rad_gh_teorica(hora):
    times = pd.DatetimeIndex([hora], tz='Europe/Madrid')
    pd_rad_teorica = Cam_Location.get_clearsky(times, model='haurwitz')
    ghi = pd_rad_teorica['ghi'].to_list()[0]
    
    return ghi

def cambio_pos_camara(latitud, longitud, altitud):
    # Se modifica la posición de la cámara
    # Importante para el cálculo de los ángulos azimutal y zenital
    global Latitud, Longitud, Altitud, Cam_Location
    
    Latitud = latitud
    Longitud = longitud
    Altitud = altitud
    Cam_Location = pv.location.Location(Latitud, Longitud, tz='Europe/Madrid', altitude=Altitud)
    
    
def cambio_caracteriticas_imagen(img):
    # Se modifican las variables globales que caracterizan a las imgágenes
    global X, Y, centro, R
    
    X, Y, canales = img.shape
    centro = (int(Y/2), int(X/2))
    R = int(Y/2)

def pre_procesado(img_bgr):
    '''
    Realiza una procesado de la imagen original, para obtener la imagen 
    con la que se trabajará.

    Parameters
    ----------
    img_bgr 

    Returns
    -------
    img_proc
    '''
    
    img_proc = cv2.bitwise_and(img_bgr, img_bgr, mask=cielo)
    return img_proc
    
def hora_imagen(ruta_imagen):
    # Funcion que convierte la ruta de imagen, en la hora de la imagen
    
    nombre_fichero = ruta_imagen.split('\\')[-1]
    fecha_formato = nombre_fichero.split('_')[-1]
    hora_str = fecha_formato.split('.')[0]
    hora = dt.strptime(hora_str, '%Y-%m-%d-%Hh%Mmin')

    return hora

def puntos_cardinales(img):
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
    
def dibujo_camino_sol(img_bgr, hora_img, delta_=timedelta(hours=2), color_=(0,255,0)):
    # Se preoaran las variables auxiliares
    img_bgr_copy = img_bgr.copy()
    lista_horas = []
    delta = timedelta(minutes=5)
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

def pixel_zenith(punto):
    # Se obtiene el ángulo zenital en coordenadas de la imagen
    # del pixel que se indique
    delta = np.array(punto) - np.array([centro[1], centro[0]])
    radio_imagen = int(np.linalg.norm(delta))

    R = int(Y/2)
    zenith_imagen = radio_imagen / R * 90
    
    return zenith_imagen

def pixel_azimuth(punto):
    # Se obtiene el ángulo azimutal en coordenadas de la imagen
    # del pixel que se indique
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
    # Devuelve una matriz, del tamaño de la imagen, con dos valores por cada elemento,
    # siendo el primer valor el zenit y el segundo el azimut, valores correspondientes a la imagen, no se encuentran corregidos
    global pixels_pos
    
    try:
        return pixels_pos
        
    except:
        pixel_pos = lambda pos: (pixel_zenith(pos), pixel_azimuth(pos))
        pixels_pos = np.array([[pixel_pos([x,y]) for y in range(Y)] for x in range(X)])
        
        return pixels_pos

def pixel_dist(punto, pnt_centro):
    delta = np.array(punto) - np.array(pnt_centro)
    dist = int(np.linalg.norm(delta))

    return dist
    
def matriz_distancias(punto):
    # Devuelve una mattriz, del tamaño de la imagen, con la dstancia de cada pixel
    # al punto especificado como parámetro

    pixels_dist = np.array([[pixel_dist([y,x], punto) for y in range(Y)] for x in range(X)])
    
    return pixels_dist

def mascara_nubes(img_bgr, centroide=None):
    '''
    Función que genera la máscara de nubes de una imágen según el método propio.

    Parameters
    ----------
    img_bgr :
        Imagen original.
    centroide : list, (Y,X)
        Posición del sol. 
        En el caso de que se indique, se eliminará
        el contorno que encierre al centroide,
        según el algoritmo propuesto.

    Returns
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

def mascara_contorno_nubes(mask_nubes):
    '''
    Se obtiene los contornos de las nubes, con el fin de ser una variable más a la hora de caluclar la radiación difusa

    Parameters
    ----------
    mask_nubes 

    Returns
    -------
    mask_nubes_cnt
    '''

    cnt_nubes, hierarchy = cv2.findContours(mask_nubes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_nubes_cnt = np.zeros((X,Y), np.uint8)
    cv2.drawContours(mask_nubes_cnt, cnt_nubes, -1, color=255, thickness=10)
    
    return mask_nubes_cnt
    

def porcion_nubes_cielo(mask_nubes):
    '''
    Función que calcula el porcentaje de nubes respecto de una máscara, 
    por defecto esta máscara es la de nubes.

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
    area_cielo = cv2.countNonZero(cielo)

    # Cálculo del área del cielo cubierta por nubes
    area_nubes = cv2.countNonZero(mask_nubes)
    
    ratio_nubes = area_nubes / area_cielo
    return ratio_nubes

def pos_solar(hora, imagen=False):
    '''
    Se obtiene las posición solar teórica en un determinado momento.
    Si imagen == True, se realizan las correcciones necesarias para obtener el zenith y el azimuth
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

    Returns
    -------
    salida_sol
    puesta_sol
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
    # Circumsolar Detection HLS
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    L = img_hls[:,:,1]

    mask_hls = cv2.inRange(L, 240, 255)

    return mask_hls

def sol_cubierto(img_bgr, hora):
    '''
    Se calcula el factor de sol cubierto po nubes.
    
    Supuestos:
        * Circularidad alta y área solar relativamente alta: sol descubierto, factor --> 1
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
    
    # Circumsolar Detection HLS
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    L = img_hls[:,:,1]

    mask_hls = cv2.inRange(L, 240, 255)
    
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
        
        # Cáculo del perímetro del sol, con el fin de calcular su circularidad
        P = cv2.arcLength(max_cnt, True)
        if P > 0:
            C = 4 * np.pi * A / P**2
        else:
            return 0.0
        
    else: # No se ha detectado el sol
        return 0.0
    '''
    # Cálculo del centroide con los momentos del controne de mayor tamaño de la imágen
    moments = cv2.moments(max_cnt)
    # (y,x)
    if (moments['m00']>0) & (moments['m00']>0):
        centroide = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
    else:
        return 0.0
    
    if C > 0.8:
        return 1.0, centroide
    elif C > 0.6:
        return 0.5, centroide
    else:
        return 0.0
    '''
    
    zenith, azimuth = pos_solar(hora, imagen=True)
    zenith /= 90 
    
    # Circularidad alta
    if (C > 0.7): 
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
    # Circumsolar Detection HLS
    img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    L = img_hls[:,:,1]

    mask_hls = cv2.inRange(L, 240, 255)
    
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
    gamma_corr = 1 / gamma
    # Se normaliza la matriz con las imágenes
    img_norm = img_bgr / maxVal
    # Se obtiene la imagen en color corregida y desnormalizada
    img_corr = np.power(img_norm, gamma_corr) * maxVal

    return np.array(img_corr, np.uint8)

def intensidad_equiponderada(img_bgr):
    # Cálculo de la luminancia del pixeles,
    # como media de los canales B, G y R
    
    B = img_bgr[:,:,0]
    G = img_bgr[:,:,1]
    R = img_bgr[:,:,2]
    
    I = np.array(B/3 + G/3 + R/3, np.uint8)
    return I

    
def intensidad_CCIR(img_bgr):
    # Basado en el convenio CCIr 601 para el cáluclo de 
    # luminancia en imágenes analógicas
    
    B = img_bgr[:,:,0]
    G = img_bgr[:,:,1]
    R = img_bgr[:,:,2]

    I = np.array(B * 0.114 + G * 0.587 + R * 0.299, np.uint8)
    return I 
    
def intensidad_ITUR(img_bgr):
    # Basado en el convenio ITU-R BT 709 para el cáluclo de 
    # luminancia en imágenes digitales
    
    B = img_bgr[:,:,0]
    G = img_bgr[:,:,1]
    R = img_bgr[:,:,2]

    I = np.array(B * 0.0722 + G * 0.7152 + R * 0.2126, np.uint8)
    return I 

def intensidad_media(img_bgr, mask=None, modo='ITUR', coseno=False, gamma=False):
    modos = {'No ponderado': intensidad_equiponderada,
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

def rad_difusa(radiacion, ratio_nubes, factor_solar, intens_media, cielo='despejado'):

    f_rad_difusa = {'nublado': f_rad_difusa_nublado,
                    'despejado': f_rad_difusa_despejado}
                    
    regr_difusa = f_rad_difusa[cielo]()
    
    if cielo == 'despejado':
        intens_media *= (1 - ratio_nubes)
        rad = regr_difusa.predict([[radiacion, intens_media]])[0]
    else:
        intens_media *= ratio_nubes
        rad = regr_difusa.predict([[radiacion, ratio_nubes, factor_solar, intens_media]])[0]
    
    return rad