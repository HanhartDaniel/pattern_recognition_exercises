# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:15:25 2020

@author: trill
"""

import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import copy
from shapely.geometry import Polygon, Point
import re
import numpy as np



#read in words
path = os.getcwd()+"/for_image_0"
Words = list()
for i in os.listdir(path):
    paths = os.getcwd()+"/for_image_0/"+i
    Words.append(np.asarray(Image.open(paths)))   
   
    
def window_trimmer(window):
    i = 0

    while i < len(window):
        if window[i] != 0:
            window = window[i:]
            i = len(window)
        i += 1

    j = len(window) - 1

    while j > 0:
        if window[j] != 0:
            window = window[0: j + 1]
            j = 0
        j = j - 1
    return(window)



  


def feature_upper(word):
    time = list()
    feature = list()
    for i in range(0, len(word)-1):       
        vect_1 = window_trimmer(word[i])
        vect_2 = window_trimmer(word[i+1])
         
        vect_1_upper = vect_1[:int((len(vect_1)+1)/2)]
        vect_2_upper = vect_2[:int((len(vect_2)+1)/2)]

        feature_count_1 = len(vect_1_upper[vect_1_upper > 128])
        feature_count_2 = len(vect_2_upper[vect_2_upper > 128])
        
        diff = feature_count_2 -feature_count_1
        
        time.append(i)
        feature.append(diff)
  
    return time,feature
 

def feature_lower(word):
    time = list()
    feature = list()
    for i in range(0, len(word)-1):       
        vect_1 = window_trimmer(word[i])
        vect_2 = window_trimmer(word[i+1])
         
        vect_1_upper = vect_1[int((len(vect_1)+1)/2):]
        vect_2_upper = vect_2[int((len(vect_2)+1)/2):]

        feature_count_1 = len(vect_1_upper[vect_1_upper > 128])
        feature_count_2 = len(vect_2_upper[vect_2_upper > 128])
        
        diff = feature_count_2 -feature_count_1
        
        time.append(i)
        feature.append(diff)
  
    return time,feature
 



def distance(vector_1, vector_2):
    n,m = len(vector_1), len(vector_2)
    DWT_m = np.zeros((n+1,m+1)) 
    for i in range(n+1):
        for j in range(m+1):
            DWT_m[i,j] = np.inf
    DWT_m[0,0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(vector_1[i-1]-vector_2[j-1])
            last_min = np.min([DWT_m[i-1,j], DWT_m[i,j-1], DWT_m[i-1,j-1]])
            
            DWT_m[i,j] = cost + last_min
    return DWT_m[n,m]

  


for word in Words:
    

    x1,y1 = feature_upper(Words[0])        
    x2,y2 = feature_upper(word)
    
    #plt.plot(x,y)  
    #x1,y1 = feature_lower(Words[0])        
    #x,y = feature_lower(Words[1]) 
    #plt.plot(x2,y2)
    

    #Image.fromarray(Words[0]).show()
    
    d = distance(y1,y2)
    if d < 70:
        print(d)
        Image.fromarray(word).show()  




      