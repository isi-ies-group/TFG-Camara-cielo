# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:22:31 2020

@author: Martin
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

def cv2plt(img_cv):
    '''
    Función que convierte una imagen de la librería de OpenCV para su 
    muestra por medio de la librería Matplotlib
    '''
    
    # Modelo de color OpenCV: BGR
    # Modelo de color de matplotlib: RGB
    
    img_plt = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return img_plt

def plt2cv(img_plt):
    
    # Modelo de color OpenCV: BGR
    # Modelo de color de matplotlib: RGB
    
    img_cv = cv2.cvtColor(img_plt, cv2.COLOR_RGB2BGR)
    return img_cv

def muestra_imagen(img_cv2):
    img_plt = cv2plt(img_cv2)
    plt.imshow(img_plt)
    
    
