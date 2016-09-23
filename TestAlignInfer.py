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

import os

import argparse

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
openfaceModelDir = os.path.join(modelDir, 'openface')

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
        'skipMulti' : True,
        'cuda' : False,
        'networkModel' : os.path.join(openfaceModelDir,'nn4.small2.v1.t7'),
        'classifierModel' : 'age_classifier.pkl'
    }
    
    r = get('http://192.168.99.100:8889/api/aligninfer', data = json.dumps(data), headers = {'Content-Type':'application/json'})

    if(r.status_code==200):
        label, confidence, bb, H = (pickle.loads(a) for a in r.json()['data'])
        
        #cv2.namedWindow('Total',cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow('Recortada',cv2.WINDOW_AUTOSIZE)
        
        #cv2.imshow('Total',img_orig)
        #cv2.imshow('Recortada',img)
        print label, confidence
        print bb
        print H
        
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
