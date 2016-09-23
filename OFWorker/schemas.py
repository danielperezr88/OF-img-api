# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:48:04 2016

@author: Dani
"""

schemas = {

    'AlignHandler':{
        "type": "object",
        "properties": {
            "image": {
                "description": "Images to process",
                "type": "string"
            },
            "size": {
                "description": "Aligned image size",
                "type": "integer",
                "default": 96
            },
            "dlibFacePredictor": {
                "description": "Path to dlib's face predictor.",
                "type": "string",
                "default": "shape_predictor_68_face_landmarks.dat"
            },
            "landmarks": {
                "description": "The landmarks to align to.",
                "type": "string",
                "choices": ['outerEyesAndNose', 'innerEyesAndBottomLip'],
                "default": "outerEyesAndNose"
            },
            "version": {
                "description": "The alignment version to use.",
                "type": "integer",
                "choices": [1,2],
                "default": 1
            },
            "skipMulti": {
                "description": "Skip images with more than one face.",
                "type": "boolean",
                "default": False
            }
        },
        "required": ["image"]
    },

    'InferHandler':{
        "type":"object",
        "properties": {
            "image": {
                "description": "Images to process",
                "type": "string"
            },
            "classifierModel": {
                "description": "The Python pickle representing the classifie"+\
                    "r. This is NOT the Torch network model, which can be se"+\
                    "t with --networkModel.",
                "type": "string"
            },
            "size": {
                "description": "Aligned image size",
                "type": "integer",
                "default": 96
            },
            "networkModel": {
                "description": "Path to Torch network model.",
                "type": "string",
                "default": "nn4.small2.v1.t7"
            },
            "cuda": {
                "description": "Use cuda implementation.",
                "type": "boolean",
                "default": False
            }
        },
        "required": ["image","classifierModel"]
    },

    'AligninferHandler':{
        "type":"object",
        "properties": {
            "image": {
                "description": "Images to process",
                "type": "string"
            },
            "classifierModel": {
                "description": "The Python pickle representing the classifie"+\
                    "r. This is NOT the Torch network model, which can be se"+\
                    "t with --networkModel.",
                "type": "string"
            },
            "size": {
                "description": "Aligned image size",
                "type": "integer",
                "default": 96
            },
            "networkModel": {
                "description": "Path to Torch network model.",
                "type": "string",
                "default": "nn4.small2.v1.t7"
            },
            "cuda": {
                "description": "Use cuda implementation.",
                "type": "boolean",
                "default": False
            },
            "dlibFacePredictor": {
                "description": "Path to dlib's face predictor.",
                "type": "string",
                "default": "shape_predictor_68_face_landmarks.dat"
            },
            "landmarks": {
                "description": "The landmarks to align to.",
                "type": "string",
                "choices": ['outerEyesAndNose', 'innerEyesAndBottomLip'],
                "default": "outerEyesAndNose"
            },
            "version": {
                "description": "The alignment version to use.",
                "type": "integer",
                "choices": [1,2],
                "default": 1
            },
            "skipMulti": {
                "description": "Skip images with more than one face.",
                "type": "boolean",
                "default": False
            }
        },
        "required": ["image","classifierModel"]
    }
}