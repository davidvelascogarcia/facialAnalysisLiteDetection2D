'''
  * ************************************************************
  *      Program: Facial Analysis Lite Detection 2D Module
  *      Type: Python
  *      Author: David Velasco Garcia @davidvelascogarcia
  * ************************************************************
  *
  * | INPUT PORT                             | CONTENT                                                 |
  * |----------------------------------------|---------------------------------------------------------|
  * | /facialAnalysisLiteDetection2D/img:i   | Input image                                             |
  *
  *
  * | OUTPUT PORT                            | CONTENT                                                 |
  * |----------------------------------------|---------------------------------------------------------|
  * | /facialAnalysisLiteDetection2D/img:o   | Output image with facial detection analysis             |
  * | /facialAnalysisLiteDetection2D/data:o  | Output result with facial analysis data                 |
  *
'''

# Libraries
import cv2
import cvlib as cv
import datetime
import imutils
import json
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import yarp

print("")
print("**************************************************************************")
print("**************************************************************************")
print("               Program: Facial Analysis Lite Detection 2D                 ")
print("                     Author: David Velasco Garcia                         ")
print("                             @davidvelascogarcia                          ")
print("**************************************************************************")
print("**************************************************************************")

print("")
print("Starting system ...")
print("")

print("")
print("Loading facialAnalysisLiteDetection2D module ...")
print("")

print("")
print("**************************************************************************")
print("YARP configuration:")
print("**************************************************************************")
print("")
print("Initializing YARP network ...")
print("")

# Init YARP Network
yarp.Network.init()

print("")
print("[INFO] Opening image input port with name /facialAnalysisLiteDetection2D/img:i ...")
print("")

# Open input image port
facialAnalysisLiteDetection2D_portIn = yarp.BufferedPortImageRgb()
facialAnalysisLiteDetection2D_portNameIn = '/facialAnalysisLiteDetection2D/img:i'
facialAnalysisLiteDetection2D_portIn.open(facialAnalysisLiteDetection2D_portNameIn)

print("")
print("[INFO] Opening image output port with name /facialAnalysisLiteDetection2D/img:o ...")
print("")

# Open output image port
facialAnalysisLiteDetection2D_portOut = yarp.Port()
facialAnalysisLiteDetection2D_portNameOut = '/facialAnalysisLiteDetection2D/img:o'
facialAnalysisLiteDetection2D_portOut.open(facialAnalysisLiteDetection2D_portNameOut)

print("")
print("[INFO] Opening data output port with name /facialAnalysisLiteDetection2D/data:o ...")
print("")

# Open output data port
facialAnalysisLiteDetection2D_portOutDet = yarp.Port()
facialAnalysisLiteDetection2D_portNameOutDet = '/facialAnalysisLiteDetection2D/data:o'
facialAnalysisLiteDetection2D_portOutDet.open(facialAnalysisLiteDetection2D_portNameOutDet)

# Create data bootle
outputBottleFacialAnalysisLiteDetection2D = yarp.Bottle()

# Image size
image_w = 640
image_h = 480

# Prepare input image buffer
in_buf_array = np.ones((image_h, image_w, 3), np.uint8)
in_buf_image = yarp.ImageRgb()
in_buf_image.resize(image_w, image_h)
in_buf_image.setExternal(in_buf_array.data, in_buf_array.shape[1], in_buf_array.shape[0])

# Prepare output image buffer
out_buf_image = yarp.ImageRgb()
out_buf_image.resize(image_w, image_h)
out_buf_array = np.zeros((image_h, image_w, 3), np.uint8)
out_buf_image.setExternal(out_buf_array.data, out_buf_array.shape[1], out_buf_array.shape[0])

print("")
print("[INFO] YARP network configured correctly.")
print("")

print("")
print("**************************************************************************")
print("Loading models:")
print("**************************************************************************")
print("")
print("[INFO] Loading models at " + str(datetime.datetime.now()) + " ...")
print("")

# Load models
# Age Models
print("")
print("[INFO] Loading age models at " + str(datetime.datetime.now()) + " ...")
print("")

ageModel = cv2.dnn.readNetFromCaffe('./../models/ageDeployModel.prototxt','./../models/ageNetModel.caffemodel')

print("")
print("Configuring age model parameters ...")
print("")

# Age model dictionary
ageList = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

print("")
print("[INFO] Age model loaded correctly.")
print("")

# Emotion Models
print("")
print("[INFO] Loading emotion models at " + str(datetime.datetime.now()) + " ...")
print("")

# Initializing emotionModel
emotionModel = Sequential()

print("")
print("Configuring emotion model parameters ...")
print("")

# Emotion model dictionary
emotionDictionary = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotionModel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotionModel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotionModel.add(MaxPooling2D(pool_size=(2, 2)))
emotionModel.add(Dropout(0.25))
emotionModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotionModel.add(MaxPooling2D(pool_size=(2, 2)))
emotionModel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotionModel.add(MaxPooling2D(pool_size=(2, 2)))
emotionModel.add(Dropout(0.25))
emotionModel.add(Flatten())
emotionModel.add(Dense(1024, activation='relu'))
emotionModel.add(Dropout(0.5))
emotionModel.add(Dense(7, activation='softmax'))

emotionModel.load_weights('./../models/emotionModel.h5')

print("")
print("[INFO] Emotion model loaded correctly.")
print("")

print("")
print("[INFO] Models loaded correctly.")
print("")

# Control loop
loopControlReadImage = 0

while int(loopControlReadImage) == 0:

    try:

        print("")
        print("**************************************************************************")
        print("Waiting for input image source:")
        print("**************************************************************************")
        print("")
        print("[INFO] Waiting input image source at " + str(datetime.datetime.now()) + " ...")
        print("")

        # Receive image source
        frame = facialAnalysisLiteDetection2D_portIn.read()

        print("")
        print("**************************************************************************")
        print("Processing input image data:")
        print("**************************************************************************")
        print("")
        print("[INFO] Processing input image data at " + str(datetime.datetime.now()) + " ...")
        print("")

        # Buffer processed image
        in_buf_image.copy(frame)
        assert in_buf_array.__array_interface__['data'][0] == in_buf_image.getRawImage().__int__()

        # YARP -> OpenCV
        rgbFrame = in_buf_array[:, :, ::-1]

        # rgbFrame to gray
        grayFrame = cv2.cvtColor(rgbFrame, cv2.COLOR_BGR2GRAY)

        print("")
        print("**************************************************************************")
        print("Analyzing image source:")
        print("**************************************************************************")
        print("")
        print("[INFO] Analyzing image source at " + str(datetime.datetime.now()) + " ...")
        print("")

        # cvlib detect and extract faces
        detectedFaces, detectedFacesConfidence = cv.detect_face(rgbFrame)

        # Configure padding
        padding = 20

        # Pre-configure values with "None"
        genderDetection = "None"
        ageDetection = "None"
        emotionDetection = "None"

        if str(detectedFaces) != "[]":

            # Analyzing detected faces
            for idx, f in enumerate(detectedFaces):

                # Get rectangle coordinates init XY coordinates and last XY coordinates
                (initX,initY) = max(0, f[0] - padding), max(0, f[1] - padding)
                (lastX,lastY) = min(rgbFrame.shape[1] - 1, f[2] + padding), min(rgbFrame.shape[0]-1, f[3] + padding)

                # Print red rectangle in detected faces
                cv2.rectangle(in_buf_array, (initX,initY), (lastX,lastY), (255,0,0), 2)


                # Gender analysis
                # Extract detected faces
                extractedGenderFace = np.copy(rgbFrame[initY:lastY, initX:lastX])

                # Detect gender
                (genderDetectionResult, genderDetectionConfidence) = cv.detect_gender(extractedGenderFace)

                # Prepare label and confidence value
                idx = np.argmax(genderDetectionConfidence)
                genderDetectionLabel = genderDetectionResult[idx]
                genderDetectionLabel = str(genderDetectionLabel) + " " + str(int(genderDetectionConfidence[idx] * 100)) + "%"

                # Get detected gender value to send
                genderDetection = genderDetectionLabel

                # Update full detection label
                detectionFrameLabel = "G: " + str(genderDetection)

                # Age detection
                # Extract detected faces
                extractedAgeFace = rgbFrame[initY:lastY, initX:lastX].copy()

                # Configure model MODEL_MEAN_VALUES
                MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

                # Detect age
                blobFromImageObject = cv2.dnn.blobFromImage(extractedAgeFace, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                ageModel.setInput(blobFromImageObject)
                ageDetected = ageModel.forward()

                # Get ageDetection compare with ageList
                ageDetection = ageList[ageDetected[0].argmax()]
                ageDetection = str(ageDetection)

                # Update full detection label
                detectionFrameLabel = detectionFrameLabel + " A: " + ageDetection

                # Emotion detection
                # Extract detected faces from grayFrame
                extractedEmotionFace = grayFrame[initY:lastY, initX:lastX]

                # Resize grayFrame
                croppedGrayFrame = np.expand_dims(np.expand_dims(cv2.resize(extractedEmotionFace, (48, 48)), -1), 0)

                # Detect emotion
                emotionPredictionDetection = emotionModel.predict(croppedGrayFrame)

                # Get emotion detection compare with emotionDictionary
                emotionPredictionDetectionIndex = int(np.argmax(emotionPredictionDetection))
                emotionDetection = emotionDictionary[emotionPredictionDetectionIndex]

                # Update full detection label
                detectionFrameLabel = detectionFrameLabel + " E: " + str(emotionDictionary[emotionPredictionDetectionIndex])


                # Print detection parameters in face detected in red color
                cv2.putText(in_buf_array, detectionFrameLabel, (initX, initY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

                # Print processed data
                print("")
                print("**************************************************************************")
                print("Resume results:")
                print("**************************************************************************")
                print("")
                print("[RESULTS] Facial analysis results:")
                print("")
                print("[GENDER] Gender: " + str(genderDetection))
                print("[AGE] Age: " + str(ageDetection))
                print("[EMOTION] Emotion: "  + str(emotionDetection))
                print("[DATE] Detection time: " + str(datetime.datetime.now()))
                print("")

                # Sending processed detection
                outputBottleFacialAnalysisLiteDetection2D.clear()
                outputBottleFacialAnalysisLiteDetection2D.addString("GENDER:")
                outputBottleFacialAnalysisLiteDetection2D.addString(str(genderDetection))
                outputBottleFacialAnalysisLiteDetection2D.addString("AGE:")
                outputBottleFacialAnalysisLiteDetection2D.addString(str(ageDetection))
                outputBottleFacialAnalysisLiteDetection2D.addString("EMOTION:")
                outputBottleFacialAnalysisLiteDetection2D.addString(str(emotionDetection))
                outputBottleFacialAnalysisLiteDetection2D.addString("DATE:")
                outputBottleFacialAnalysisLiteDetection2D.addString(str(datetime.datetime.now()))
                facialAnalysisLiteDetection2D_portOutDet.write(outputBottleFacialAnalysisLiteDetection2D)

        # If no faces detected publish "None results"
        else:

            print("")
            print("[INFO] No faces detected.")
            print("")

            # Sending processed detection
            outputBottleFacialAnalysisLiteDetection2D.clear()
            outputBottleFacialAnalysisLiteDetection2D.addString("GENDER:")
            outputBottleFacialAnalysisLiteDetection2D.addString(str(genderDetection))
            outputBottleFacialAnalysisLiteDetection2D.addString("AGE:")
            outputBottleFacialAnalysisLiteDetection2D.addString(str(ageDetection))
            outputBottleFacialAnalysisLiteDetection2D.addString("EMOTION:")
            outputBottleFacialAnalysisLiteDetection2D.addString(str(emotionDetection))
            outputBottleFacialAnalysisLiteDetection2D.addString("DATE:")
            outputBottleFacialAnalysisLiteDetection2D.addString(str(datetime.datetime.now()))
            facialAnalysisLiteDetection2D_portOutDet.write(outputBottleFacialAnalysisLiteDetection2D)


        print("")
        print("[INFO] Image source analysis done correctly.")
        print("")

        # Sending processed image
        print("")
        print("[INFO] Sending processed image at " + str(datetime.datetime.now()) + " ...")
        print("")

        out_buf_array[:,:] = in_buf_array
        facialAnalysisLiteDetection2D_portOut.write(out_buf_image)

    except:
        print("")
        print("[ERROR] Empty frame.")
        print("")

# Close ports
print("[INFO] Closing ports ...")
facialAnalysisLiteDetection2D_portIn.close()
facialAnalysisLiteDetection2D_portOut.close()
facialAnalysisLiteDetection2D_portOutDet.close()

print("")
print("")
print("**************************************************************************")
print("Program finished")
print("**************************************************************************")
print("")
print("facialAnalysisLiteDetection2D program closed correctly.")
print("")
