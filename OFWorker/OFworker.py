# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:40:37 2016

@author: Dani
"""

import argparse
import os
import cv2
import numpy as np
import pickle

import urllib

import openface
import openface.helper

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

def align_v2 (rgbImg, H, imgDim):
    imgWidth = np.shape(rgbImg)[1]
    imgHeight = np.shape(rgbImg)[0]
    thumbnail = np.zeros((imgDim, imgDim, 3), np.uint8)

    # interpolation from input image to output pixels using transformation mat H to compute
    # which input coordinates map to output
    for y in range(imgDim):
        for x in range(imgDim):
            xprime = x * H[0][1] + y * H[0][0] + H[0][2]
            yprime = x * H[1][1] + y * H[1][0] + H[1][2]
            tx = int(xprime)
            ty = int(yprime)
            horzOffset = 1
            vertOffset = 1
            if(tx < 0 or tx >= imgWidth or ty < 0 or ty >= imgHeight):
                continue
            if(tx == imgWidth - 1):
                horzOffset = 0
            if(ty == imgHeight - 1):
                vertOffset = 0
            f1 = xprime - float(tx)
            f2 = yprime - float(ty)
            upperLeft = rgbImg[ty][tx]
            upperRight = rgbImg[ty][tx + horzOffset]
            bottomLeft = rgbImg[ty + vertOffset][tx]
            bottomRight = rgbImg[ty + vertOffset][tx + horzOffset]

            thumbnail[x][y][0] = upperLeft[0] * (1.0 - f1) * (1.0 - f2) + upperRight[0] * f1 * (
                1.0 - f2) + bottomLeft[0] * (1.0 - f1) * f2 + bottomRight[0] * f1 * f2
            thumbnail[x][y][1] = upperLeft[1] * (1.0 - f1) * (1.0 - f2) + upperRight[1] * f1 * (
                1.0 - f2) + bottomLeft[1] * (1.0 - f1) * f2 + bottomRight[1] * f1 * f2
            thumbnail[x][y][2] = upperLeft[2] * (1.0 - f1) * (1.0 - f2) + upperRight[2] * f1 * (
                1.0 - f2) + bottomLeft[2] * (1.0 - f1) * f2 + bottomRight[2] * f1 * f2

    return thumbnail

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

def AligninferMain(args, pkl=True):
    
    try:
        with open(args.classifierModel, 'r') as f:
            le, clf = pickle.load(f)
    except IOError as e:
        print('Classifier load failed on folder %s: %s' % (os.getcwd(), e))
    
    args.image, bb, H = AlignMain(args,pkl=False)
    
    net = openface.TorchNeuralNet(
        args.networkModel, imgDim=args.size, cuda=args.cuda)
    
    predictions = clf.predict_proba(net.forward(args.image)).ravel()
    #maxI = np.argmax(predictions)
    #label = le.inverse_transform(maxI)
    #confidence = predictions[maxI]
    
    if pkl:
        return pickle.dumps(le.classes_), pickle.dumps(predictions), pickle.dumps(bb), pickle.dumps(H)
    else:
        return le.classes_, predictions, bb, H

def InferMain(args, pkl=True):
    
    try:
        with open(args.classifierModel, 'r') as f:
            le, clf = pickle.load(f)
    except IOError as e:
        print('Classifier load failed on folder %s: %s' % (os.getcwd(), e))
    
    url_response = urllib.urlopen(args.image)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    args.image = cv2.imdecode(img_array, -1)
    
    net = openface.TorchNeuralNet(
        args.networkModel, imgDim=args.size, cuda=args.cuda)
    
    predictions = clf.predict_proba(net.forward(args.image)).ravel()
    #maxI = np.argmax(predictions)
    #label = le.inverse_transform(maxI)
    #confidence = predictions[maxI]
    
    if pkl:
        return pickle.dumps(le.classes_), pickle.dumps(predictions)
    else:
        return le.classes_, predictions
    
def AlignMain(args, pkl=True):
    
    url_response = urllib.urlopen(args.image)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    args.image = cv2.imdecode(img_array, -1)
    
    if np.prod(args.image.shape) == 0:
        raise Exception("None/Emtpy image given...")
    
    if args.landmarks == 'outerEyesAndNose':
        landmarkIndices = openface.AlignDlib.OUTER_EYES_AND_NOSE
    elif args.landmarks == 'innerEyesAndBottomLip':
        landmarkIndices = openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    else:
        raise Exception("Landmarks unrecognized: {}".format(args.landmarks))    

    #raise Exception(os.path.join(dlibModelDir,args.dlibFacePredictor))
    align = openface.AlignDlib(os.path.join(dlibModelDir,args.dlibFacePredictor).encode())
    
    img = cv2.cvtColor(args.image, cv2.COLOR_BGR2RGB)
    
    if img is not None:

        bb = align.getLargestFaceBoundingBox(img, args.skipMulti)
        
        landmarks = align.findLandmarks(img, bb)
        npLandmarks = np.float32(landmarks)
        npLandmarkIndices = np.array(landmarkIndices)
        fidPoints = npLandmarks[npLandmarkIndices]
        templateLandmarks = args.size * MINMAX_TEMPLATE[npLandmarkIndices]
        
        if args.version == 1:
            
            H = cv2.getAffineTransform(fidPoints, templateLandmarks)
            
            img = cv2.warpAffine(img, H, (args.size, args.size))
            
        elif args.version == 2:

            templateMat = np.ones((3, 3), dtype=np.float32)
            for i in range(3):
                for j in range(2):
                    templateMat[i][j] = templateLandmarks[i][j]
    
            templateMat = np.transpose(np.linalg.inv(templateMat))
    
            # create transformation matrix from output pixel coordinates to input
            # pixel coordinates
            H = np.zeros((2, 3), dtype=np.float32)
            for i in range(3):
                H[0][i] = fidPoints[0][0] * templateMat[0][i] + \
                    fidPoints[1][0] * templateMat[1][i] + \
                    fidPoints[2][0] * templateMat[2][i]
                    
                H[1][i] = fidPoints[0][1] * templateMat[0][i] + \
                    fidPoints[1][1] * templateMat[1][i] + \
                    fidPoints[2][1] * templateMat[2][i]
            
            img = align_v2(img, H, args.size)
    
        if img is not None:
            
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
    if pkl:
        return pickle.dumps(img), pickle.dumps(bb), pickle.dumps(H)
    else:
        return img, bb, H

if __name__ == '__main__':

    print 'New request found'

    commonParser = argparse.ArgumentParser()
    
    ##############     Common Parser     ##############
    
    subparsers = commonParser.add_subparsers(dest='mode', help="Mode")
    
    commonParser.add_argument(
        'image',
        type = str,
        help = "Image to process (URL)"
    )
    
    commonParser.add_argument(
        '--size',
        type = int,
        default = 96,
        help = "Default image size."
    )
    
    ##############     Classification Function Parser     ##############
    
    inferParser = argparse.ArgumentParser(add_help=False)
    
    inferParser.add_argument(
        'classifierModel',
        type = str,
        help = 'The Python pickle representing the classifier. This is NOT' + \
        ' the Torch network model, which can be set with --networkModel.'
    )
    
    inferParser.add_argument(
        '--networkModel',
        type = str,
        help="Path to Torch network model.",
        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
    )
    
    inferParser.add_argument(
        '--cuda',
        action = 'store_true'
    )
    
    subparsers.add_parser(
        'infer',
        parents = [inferParser],
        help = 'Apply a classification model with a given image'
    )
    
    
    ##############     Alignment Function Parser     ##############
    
    alignmentParser = argparse.ArgumentParser(add_help=False)
    
    alignmentParser.add_argument(
        '--dlibFacePredictor',
        type = str,
        default = os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"
        ),
        help = "Path to dlib's face predictor."
    )
    
    alignmentParser.add_argument(
        '--landmarks',
        type = str,
        choices = ['outerEyesAndNose', 'innerEyesAndBottomLip'],
        default = 'outerEyesAndNose',
        help='The landmarks to align to.'
    )
    
    alignmentParser.add_argument(
        '--version',
        type = int,
        choices = [1, 2],
        default = 1,
        help = 'The alignment version to use.'
    )
    
    alignmentParser.add_argument(
        '--skipMulti',
        action = 'store_true',
        help="Skip images with more than one face."
    )
    
    subparsers.add_parser(
        'align',
        parents = [alignmentParser],
        help = 'Align a set of images.'
    )
    
    
    ##############     Align & Classify Function Parser     ##############
    
    subparsers.add_parser(
        'aligninfer',
        parents = [inferParser, alignmentParser],
        help = 'Apply a classification model with a given image'
    )
    
    
###############################################################################
    
    args = commonParser.parse_args()
    
    if args.mode == 'infer' and args.classifierModel.endswith(".t7"):
        raise Exception("""
Torch network model passed as the classification model,
which should be a Python pickle (.pkl)

See the documentation for the distinction between the Torch
network and classification models:

        http://cmusatyalab.github.io/openface/demo-3-classifier/
        http://cmusatyalab.github.io/openface/training-new-models/

Use `--networkModel` to set a non-standard Torch network model.""")
            
    if args.mode == 'align':
        AlignMain(args, pkl=False)
    elif args.mode == 'infer':
        InferMain(args, pkl=False)
    elif args.mode == 'aligninfer':
        AligninferMain(args, pkl=False)
    else:
        raise Exception("""
Unknown Mode Value...""")
            
