# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:12:08 2017

@author: Juilee Rege
"""

import cv2
import glob
import pandas as pd

df = pd.read_csv('./ISIC_ground_truth.csv', index_col=['image_id'], names = ['image_id','melanoma', 'seb_ker'], skiprows=1)
indices = df.index.values

##dividing into folders based on label
for name in indices:
    for filename in glob.glob('./ISIC_cleaned_data/*.jpg'):
        fn = filename[20:]
        if(len(name)==(len(fn)-4)):    
            if(fn.startswith(name)):
                img = cv2.imread(filename)
                if(df.ix[name]['melanoma'] == 0):
                    cv2.imwrite('./preprocessed_data/benign/'+ fn, img)
                else:
                    cv2.imwrite('./preprocessed_data/malignant/'+ fn, img)
