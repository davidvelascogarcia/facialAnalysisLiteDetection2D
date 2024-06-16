'''
  * ************************************************************
  *      Program: Facial Analysis Lite Detection 2D
  *      Type: Python
  *      Author: David Velasco Garcia @davidvelascogarcia
  * ************************************************************
  *
  * | INPUT PORT                           | CONTENT                                                 |
  * |--------------------------------------|---------------------------------------------------------|
  * | /facialAnalysisLiteDetection2D/img:i | Input image                                             |
  *
  *
  * | OUTPUT PORT                          | CONTENT                                                 |
  * |--------------------------------------|---------------------------------------------------------|
  * | /facialAnalysisLiteDetection2D/img:o | Output image with facial analysis                       |
  * | /facialAnalysisLiteDetection2D/data:o| Output result, facial analysis data                     |
'''

# Libraries
import configparser
import cv2
import cvlib as cv
import datetime
from halo import Halo
import numpy as np
import platform
import queue
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import threading
import time
import yarp


class FacialAnalysisLiteDetection2D:

    # Function: Constructor
    def __init__(self):

        # Build Halo spinner
        self.systemResponse = Halo(spinner='dots')

    # Function: getSystemPlatform
    def getSystemPlatform(self):

        # Get system configuration
        print("\nDetecting system and release version ...\n")
        systemPlatform = platform.system()
        systemRelease = platform.release()

        print("**************************************************************************")
        print("Configuration detected:")
        print("**************************************************************************")
        print("\nPlatform:")
        print(systemPlatform)
        print("Release:")
        print(systemRelease)

        return systemPlatform, systemRelease

    # Function: getAuthenticationData
    def getAuthenticationData(self):

        print("\n**************************************************************************")
        print("Authentication:")
        print("**************************************************************************\n")

        loopControlFileExists = 0

        while int(loopControlFileExists) == 0:
            try:
                # Get authentication data
                print("\nGetting authentication data ...\n")

                authenticationData = configparser.ConfigParser()
                authenticationData.read('../config/config.ini')
                authenticationData.sections()

                imageWidth = authenticationData['Configuration']['image-width']
                imageHeight = authenticationData['Configuration']['image-height']

                ageModel = "./../models/" + authenticationData['Models']['age-model']
                ageLabels = "./../models/" + authenticationData['Models']['age-labels']
                emotionModel = "./../models/" + authenticationData['Models']['emotion-model']

                print("Image width: " + str(imageWidth))
                print("Image height: " + str(imageHeight))

                print("Age model: " + str(ageModel))
                print("Age labels: " + str(ageLabels))
                print("Emotion model: " + str(emotionModel))

                # Exit loop
                loopControlFileExists = 1

            except:

                systemResponseMessage = "\n[ERROR] Sorry, config.ini not founded, waiting 4 seconds to the next check ...\n"
                self.systemResponse.text_color = "red"
                self.systemResponse.fail(systemResponseMessage)
                time.sleep(4)

        systemResponseMessage = "\n[INFO] Data obtained correctly.\n"
        self.systemResponse.text_color = "green"
        self.systemResponse.succeed(systemResponseMessage)

        return imageWidth, imageHeight, ageModel, ageLabels, emotionModel

    # Function: loadAgeModel
    def loadAgeModel(self, ageModel, ageLabels):

        # Load trained age model with their labels
        ageModel = cv2.dnn.readNetFromCaffe(str(ageLabels), str(ageModel))

        # Prepare age list dictionary
        ageList = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']


        systemResponseMessage = "\n[INFO] Age model loaded correctly correctly.\n"
        self.systemResponse.text_color = "green"
        self.systemResponse.succeed(systemResponseMessage)

        return ageModel, ageList

    # Function: loadEmotionModel
    def loadEmotionModel(self, emotionModelPath):

        # Build neural network
        emotionModel = Sequential()

        # Configure neural network
        emotionModel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
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

        # Load trained network and set the trained weights in configured neural network
        emotionModel.load_weights(str(emotionModelPath))

        # Prepare emotion list dictionary
        emotionList = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        systemResponseMessage = "\n[INFO] Emotion model loaded correctly correctly.\n"
        self.systemResponse.text_color = "green"
        self.systemResponse.succeed(systemResponseMessage)

        return emotionModel, emotionList

    # Function: genderAnalysis
    def genderAnalysis(self, userFace, xMin, yMin, xMax, yMax, genderQueueBuffer):

        # Analyze user gender
        (userGender, genderConfidence) = cv.detect_gender(userFace)

        # Get index confidence
        genderConfidenceIndex = np.argmax(genderConfidence)

        # Get prediction based on confidence of gender
        dataSolved = str(userGender[genderConfidenceIndex])

        # Prepare data solved with prediction and de confidence
        dataSolved = str(dataSolved) + " " + str(int(genderConfidence[genderConfidenceIndex] * 100)) + "%"

        genderQueueBuffer.put(dataSolved)

    # Function: ageAnalysis
    def ageAnalysis(self, userFace, xMin, yMin, xMax, yMax, ageModel, ageList, ageQueueBuffer):

        # Configure model values
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

        # Extract blob from image
        blobImage = cv2.dnn.blobFromImage(userFace, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Analyze user age
        ageModel.setInput(blobImage)

        # Get user age
        userAge = ageModel.forward()

        # Get ageDetection compare with ageList and index
        userAge = ageList[userAge[0].argmax()]

        # Prepare data solved with prediction
        dataSolved = str(userAge)

        ageQueueBuffer.put(dataSolved)

    # Function: emotionAnalysis
    def emotionAnalysis(self, userFace, xMin, yMin, xMax, yMax, emotionModel, emotionList, emotionQueueBuffer):

        # BGR to grey scale conversion
        userFace = cv2.cvtColor(userFace, cv2.COLOR_BGR2GRAY)

        # Resize user face to emotion model size
        userFace = np.expand_dims(np.expand_dims(cv2.resize(userFace, (48, 48)), -1), 0)

        # Analyze user emotion
        userEmotion = emotionModel.predict(userFace)

        # Get emotion detection compare with emotionDictionary
        userEmotion = int(np.argmax(userEmotion))
        userEmotion = emotionList[userEmotion]

        # Prepare data solved with prediction
        dataSolved = str(userEmotion)

        emotionQueueBuffer.put(dataSolved)

    # Function: analyzeImage
    def analyzeImage(self, dataToSolve, ageModel, ageList, emotionModel, emotionList, imageWidth, imageHeight, inputImagePort, outputImagePort, outputDataPort):

        # Detect users in data to solve
        detectedUsers, detectionConfidence = cv.detect_face(dataToSolve)

        # Prepare default padding value
        padding = 20

        # If a face is detected
        if str(detectedUsers) != "[]":

            # For each detected user
            for idx, f in enumerate(detectedUsers):

                # Get user rectangle coordinates
                (xMin,yMin) = max(0, f[0] - padding), max(0, f[1] - padding)
                (xMax,yMax) = min(dataToSolve.shape[1] - 1, f[2] + padding), min(dataToSolve.shape[0]-1, f[3] + padding)

                # Draw red rectangle in user detected
                cv2.rectangle(dataToSolve, (xMin, yMin), (xMax, yMax), (255, 0, 0), 2)

                # Extract only user detected face
                userFace = np.copy(dataToSolve[yMin:yMax, xMin:xMax])

                # Build Queue buffers
                genderQueueBuffer = queue.Queue()
                ageQueueBuffer = queue.Queue()
                emotionQueueBuffer = queue.Queue()

                # Build three threads to improve analysis speed
                genderAnalysisThread = threading.Thread(target=self.genderAnalysis, args=(userFace, xMin, yMin, xMax, yMax, genderQueueBuffer))
                ageAnalysisThread = threading.Thread(target=self.ageAnalysis, args=(userFace, xMin, yMin, xMax, yMax, ageModel, ageList, ageQueueBuffer))
                emotionAnalysisThread = threading.Thread(target=self.emotionAnalysis, args=(userFace, xMin, yMin, xMax, yMax, emotionModel, emotionList, emotionQueueBuffer))

                # Start three threads
                genderAnalysisThread.start()
                ageAnalysisThread.start()
                emotionAnalysisThread.start()

                # Wait until three threads ends
                genderAnalysisThread.join()
                ageAnalysisThread.join()
                emotionAnalysisThread.join()

                # Get threads results
                userGender = genderQueueBuffer.get()
                userAge = ageQueueBuffer.get()
                userEmotion = emotionQueueBuffer.get()

                # Prepare data solved results
                dataSolvedResults = "Gender: " + str(userGender) + ", Age: " + str(userAge) + ", Emotion: " + str(userEmotion) + ", Date: " + str(datetime.datetime.now())
                dataSolvedResultsLabel = "G: " + str(userGender) + ", A: " + str(userAge) + ", E: " + str(userEmotion) + ", D: " + str(datetime.datetime.now())

                # Draw data solved results on user detected
                cv2.putText(dataToSolve, dataSolvedResultsLabel, (xMin, yMin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

                # Send detected results
                outputDataPort.send(dataSolvedResults)

        else:
            systemResponseMessage = "\n[INFO] No faces detected.\n"
            self.systemResponse.text_color = "blue"
            self.systemResponse.info(systemResponseMessage)

            # Prepare output results
            dataSolvedResults = "Gender: None, Age: None, Emotion: None, Date: " + str(datetime.datetime.now())

            # Send detected results
            outputDataPort.send(dataSolvedResults)

        systemResponseMessage = "\n" + str(dataSolvedResults) + "\n"
        self.systemResponse.text_color = "green"
        self.systemResponse.succeed(systemResponseMessage)

        dataSolvedImage = dataToSolve

        return dataSolvedImage

    # Function: processRequest
    def processRequests(self, ageModel, ageList, emotionModel, emotionList, imageWidth, imageHeight, inputImagePort, outputImagePort, outputDataPort):

        # Variable to control loopProcessRequests
        loopProcessRequests = 0

        while int(loopProcessRequests) == 0:

            # Waiting to input data request
            print("**************************************************************************")
            print("Waiting for input data request:")
            print("**************************************************************************")

            systemResponseMessage = "\n[INFO] Waiting for input data request at " + str(datetime.datetime.now()) + " ...\n"
            self.systemResponse.text_color = "yellow"
            self.systemResponse.warn(systemResponseMessage)

            # Receive input request
            dataToSolve = inputImagePort.receive()

            print("\n**************************************************************************")
            print("Processing:")
            print("**************************************************************************\n")

            try:
                dataSolvedImage = self.analyzeImage(dataToSolve, ageModel, ageList, emotionModel, emotionList, imageWidth, imageHeight, inputImagePort, outputImagePort, outputDataPort)

                # Send output results
                outputImagePort.send(dataSolvedImage)

            except:
                systemResponseMessage = "\n[ERROR] Sorry, i couldn´t resolve your request.\n"
                self.systemResponse.text_color = "red"
                self.systemResponse.fail(systemResponseMessage)


class YarpDataPort:

    # Function: Constructor
    def __init__(self, portName):

        # Build Halo spinner
        self.systemResponse = Halo(spinner='dots')

        # Build port and bottle
        self.yarpPort = yarp.Port()
        self.yarpBottle = yarp.Bottle()

        systemResponseMessage = "\n[INFO] Opening Yarp data port " + str(portName) + " ...\n"
        self.systemResponse.text_color = "yellow"
        self.systemResponse.warn(systemResponseMessage)

        # Open Yarp port
        self.portName = portName
        self.yarpPort.open(self.portName)

    # Function: receive
    def receive(self):

        self.yarpPort.read(self.yarpBottle)
        dataReceived = self.yarpBottle.toString()
        dataReceived = dataReceived.replace('"', '')

        systemResponseMessage = "\n[RECEIVED] Data received: " + str(dataReceived) + " at " + str(datetime.datetime.now()) + ".\n"
        self.systemResponse.text_color = "blue"
        self.systemResponse.info(systemResponseMessage)

        return dataReceived

    # Function: send
    def send(self, dataToSend):

        self.yarpBottle.clear()
        self.yarpBottle.addString(str(dataToSend))
        self.yarpPort.write(self.yarpBottle)

    # Function: close
    def close(self):

        systemResponseMessage = "\n[INFO] " + str(self.portName) + " port closed correctly.\n"
        self.systemResponse.text_color = "yellow"
        self.systemResponse.warn(systemResponseMessage)

        self.yarpPort.close()


class YarpImagePort:

    # Function: Constructor
    def __init__(self, portName, imageWidth, imageHeight):

        # Build Halo spinner
        self.systemResponse = Halo(spinner='dots')

        # If input image port required
        if "/img:i" in str(portName):
            self.yarpPort = yarp.BufferedPortImageRgb()

        # If output image port required
        else:
            self.yarpPort = yarp.Port()

        systemResponseMessage = "\n[INFO] Opening Yarp image port " + str(portName) + " ...\n"
        self.systemResponse.text_color = "yellow"
        self.systemResponse.warn(systemResponseMessage)

        # Open Yarp port
        self.portName = portName
        self.yarpPort.open(self.portName)

        # Build image buffer
        self.imageWidth = int(imageWidth)
        self.imageHeight = int(imageHeight)
        self.bufferImage = yarp.ImageRgb()
        self.bufferImage.resize(self.imageWidth, self.imageHeight)
        self.bufferArray = np.ones((self.imageHeight, self.imageWidth, 3), np.uint8)
        self.bufferImage.setExternal(self.bufferArray.data, self.bufferArray.shape[1], self.bufferArray.shape[0])

    # Function: receive
    def receive(self):

        image = self.yarpPort.read()
        self.bufferImage.copy(image)
        assert self.bufferArray.__array_interface__['data'][0] == self.bufferImage.getRawImage().__int__()
        image = self.bufferArray[:, :, ::-1]

        return self.bufferArray

    # Function: send
    def send(self, dataToSend):

        self.bufferArray[:,:] = dataToSend
        self.yarpPort.write(self.bufferImage)

    # Function: close
    def close(self):

        systemResponseMessage = "\n[INFO] " + str(self.portName) + " port closed correctly.\n"
        self.systemResponse.text_color = "yellow"
        self.systemResponse.warn(systemResponseMessage)

        self.yarpPort.close()


# Function: main
def main():

    print("**************************************************************************")
    print("**************************************************************************")
    print("             Program: Facial Analysis Lite Detection 2D                   ")
    print("                     Author: David Velasco Garcia                         ")
    print("                             @davidvelascogarcia                          ")
    print("**************************************************************************")
    print("**************************************************************************")

    print("\nLoading Facial Analysis Lite Detection 2D engine ...\n")

    # Build facialAnalysisLiteDetection2D object
    facialAnalysisLiteDetection2D = FacialAnalysisLiteDetection2D()

    # Get system platform
    systemPlatform, systemRelease = facialAnalysisLiteDetection2D.getSystemPlatform()

    # Get authentication data
    imageWidth, imageHeight, ageModel, ageLabels, emotionModel = facialAnalysisLiteDetection2D.getAuthenticationData()

    # Load age model
    ageModel, ageList = facialAnalysisLiteDetection2D.loadAgeModel(ageModel, ageLabels)

    # Load emotion model
    emotionModel, emotionList = facialAnalysisLiteDetection2D.loadEmotionModel(emotionModel)

    # Init Yarp network
    yarp.Network.init()

    # Create Yarp ports
    inputImagePort = YarpImagePort("/facialAnalysisLiteDetection2D/img:i", imageWidth, imageHeight)
    outputImagePort = YarpImagePort("/facialAnalysisLiteDetection2D/img:o", imageWidth, imageHeight)
    outputDataPort = YarpDataPort("/facialAnalysisLiteDetection2D/data:o")

    # Process input requests
    facialAnalysisLiteDetection2D.processRequests(ageModel, ageList, emotionModel, emotionList, imageWidth, imageHeight, inputImagePort, outputImagePort, outputDataPort)

    # Close Yarp ports
    inputImagePort.close()
    outputImagePort.close()
    outputDataPort.close()

    print("**************************************************************************")
    print("Program finished")
    print("**************************************************************************")
    print("\nfacialAnalysisLiteDetection2D program finished correctly.\n")


if __name__ == "__main__":

    # Call main function
    main()