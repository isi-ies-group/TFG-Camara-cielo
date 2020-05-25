#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import cv2
from matplotlib import pyplot as plt


# In[13]:


path_fotos_patron = 'C:/Users/nitra/Documents/GitHub/TFG_Camara/Fotos Cielo/patron/'
img_patron = cv2.imread(path_fotos_patron + 'imagen_patron.jpg', cv2.IMREAD_COLOR)


# In[14]:


# Máscara circular, para corregir los efectos de la lente
def mascara_inicial():
    mask = img_patron.copy()
    X, Y, colores = mask.shape
    Y = mask.shape[1]
    centro = (int(Y/2), int(X/2))
    R = int(Y/2)

    cv2.circle(mask, centro, R, color=(255,255,255), thickness=-1)
    # Conversión a escala de grises
    mask = np.uint8(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    mask = cv2.inRange(mask, 255, 255)


    #masked_data = cv2.bitwise_and(img_patron, img_patron, mask=mask)
    #plt.imshow(masked_data)
    return mask


# Máscara para determinar zone de cuielo visibe
def mascara_cielo():
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

    mask_nocielo = cv2.bitwise_not(np.uint8(mask_nocielo))
    # masked_data = cv2.bitwise_and(img_patron, img_patron, mask=mask_nocielo)
    
    # plt.figure(1)
    # plt.imshow(blue)
    # plt.figure(2)      
    # plt.imshow(green)
    # plt.figure(3)      
    # plt.imshow(red)

    return mask_nocielo

