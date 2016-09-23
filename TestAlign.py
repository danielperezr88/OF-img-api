# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:05:55 2016

@author: Dani
"""

from requests import get

#import numpy as np

import json
import pickle
#import cv2

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('image', type=str,
                        help="Path to dlib's face predictor.")
    
    args = parser.parse_args()

    data = {
        'image' : args.image,
        'size' : 96,
        'dlibFacePredictor' : 'shape_predictor_68_face_landmarks.dat',
        'landmarks' : 'outerEyesAndNose',
        'version' : 1,
        'skipMulti' : True
    }

    r = get('http://localhost:8889/api/align', data = json.dumps(data), headers = {'Content-Type':'application/json'})

    if(r.status_code==200):
        img, bb, H = (pickle.loads(a) for a in r.json()['data'])
        
        #cv2.namedWindow('Total',cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow('Recortada',cv2.WINDOW_AUTOSIZE)
        
        #cv2.imshow('Total',img_orig)
        #cv2.imshow('Recortada',img)
        print bb
        print H
        
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()