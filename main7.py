#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 10:50:20 2016

@author: johndevera
"""

print("main7 GOOD AND RUNNING")

from PIL import Image
import csv as cs
import os

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import Sequential, model_from_json # import the type of model
from keras.layers.core import Dense, Dropout, Activation, Flatten 
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam, adadelta
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



#import utils2 as u
#from numpy import *
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



def saveBW(defaultPath, batchType):
    listing = os.listdir(defaultPath + batchType) 
    num_samples = len(listing)
    
    for file in listing:
        if file.endswith('.jpg'):
            print("FILE: " + file)
            im = Image.open(defaultPath + batchType +'/' + file)   
            img = im.resize((128,128))
            gray = img.convert('L')
            gray.save(defaultPath +'trainb' + '/' +  file)
    print("NUM: " + str(num_samples))  


def splitImages(defaultPath, batchType):
    labelS = openCSV(defaultPath, batchType)
    imageS, imageList = openImage(defaultPath, batchType)
    savePath = defaultPath + batchType + '1/'
    print("LEN", len(imageList))

    i = 0
    for file in imageList:
        if file.endswith(".jpg"):
            if labelS[i] == '1':
                imageS = Image.open(defaultPath + batchType + '/' + file)
                imageS.save(savePath + '1-structure' + '/' + file)  
            elif labelS[i] == '2':
                imageS = Image.open(defaultPath + batchType + '/' + file)
                imageS.save(savePath + '2-indoor' + '/' + file)  
            elif labelS[i] == '3':
                imageS = Image.open(defaultPath + batchType + '/' + file)
                imageS.save(savePath + '3-people' + '/' + file)  
            elif labelS[i] == '4':
                imageS = Image.open(defaultPath + batchType + '/' + file)
                imageS.save(savePath + '4-animals' + '/' + file)  
            elif labelS[i] == '5':        
                imageS = Image.open(defaultPath + batchType + '/' + file)
                imageS.save(savePath + '5-plantlife' + '/' + file)  
            elif labelS[i] == '6':
                imageS = Image.open(defaultPath + batchType + '/' + file)
                imageS.save(savePath + '6-food' + '/' + file)  
            elif labelS[i] == '7':
                    imageS = Image.open(defaultPath + batchType + '/' + file)
                    imageS.save(savePath + '7-cars' + '/' + file)  
            else:
                imageS = Image.open(defaultPath + batchType + '/' + file)
                imageS.save(savePath + '8-sea' + '/' + file)  
            i=i+1


def openCSV(defaultPath, batchType):
    fileName = defaultPath + batchType
    print("fileName: ", fileName)
    newFile = open(str(fileName)+'.csv')
    reader = cs.reader(newFile)
    data = list(reader)
    data = np.asarray(data)
    len = data.shape[0]
    data = data[1:len] 
    label = data[:,1]
    print("Labels found: " + str(label.shape[0]))
    return label
           

    
def openImage(defaultPath, batchType):
    defaultImage = '00001.jpg' 
    defaultPath = defaultPath + batchType + '/'
    path = defaultPath + defaultImage

    #count number of files
    folder = os.listdir(defaultPath)
    numFiles = 0
    for file in folder:
        if file.endswith(".jpg"):
            numFiles = numFiles+1
    print("Images found: " + str(numFiles))
       
    #make image data matrix
    image = Image.open(path).convert() #for 3 layers
    image = np.array(image)
    data = np.zeros((numFiles, image.shape[0], image.shape[1], image.shape[2]))
    

    #open images and assign to data matrix
    i = 0
    for file in folder:
        path = defaultPath + file
        if file.endswith(".jpg"):

            image = Image.open(path).convert() #for BW
            
            image = np.array(image)
            #print("WOW", image.shape)
            #image = np.ravel(image)
            data[i] = image
            i = i+1        
    return data, folder    
        
            
def openImageBW(defaultPath, batchType):
    defaultImage = '00001.jpg' 
    defaultPath = defaultPath + batchType + '/'
    path = defaultPath + defaultImage

    #count number of files
    folder = os.listdir(defaultPath)
    numFiles = 0
    for file in folder:
        if file.endswith(".jpg"):
            numFiles = numFiles+1
    print("Images found: " + str(numFiles))
       
    #make image data matrix
    image = Image.open(path).convert("L") #for 3 layers
    image = np.array(image)
    data = np.zeros((numFiles, image.shape[0], image.shape[1]))
    

    #open images and assign to data matrix
    i = 0
    for file in folder:
        path = defaultPath + file
        if file.endswith(".jpg"):

            image = Image.open(path).convert("L") #for 3 layers
            image = np.array(image)
            data[i] = image
            i = i+1
    print("Image Shape: " + str(data.shape))        
    return data, folder    
    
    
def processData(images, labels, batchType, test, classes):    
    
    images = images.reshape(images.shape[0],
                            images.shape[3],
                            images.shape[1],
                            images.shape[2])
    images = images.astype('float32')
    images = images / 255
    #labels = np_utils.to_categorical(labels, classes+1)
    
    labelMatrix = np.zeros((labels.shape[0], classes))
    for i in xrange(0, labelMatrix.shape[0], 1):
        y = int(labels[i]) - 1
        labelMatrix[i, y] = 1
    labels = labelMatrix
 
    
    if batchType != 'val':    
        trainImages = images[:-test,:]
        trainLabels = labels[:-test]
        testImages = images[images.shape[0] - test:,:] #save last amount for test
        testLabels = labels[labels.shape[0] - test:] #save last amount for test

    else:
        trainImages = images
        trainLabels = labels
        testImages = images
        testLabels = labels

    return images, labels, trainImages, trainLabels, testImages, testLabels


def processDataBW(images, labels, batchType, test, classes):    
    
    images = images.reshape(images.shape[0],
                            1,
                            images.shape[1],
                            images.shape[2])
    images = images.astype('float32')
    images = images / 255
    #labels = np_utils.to_categorical(labels, classes+1)
    
    labelMatrix = np.zeros((labels.shape[0], classes))
    for i in xrange(0, labelMatrix.shape[0], 1):
        y = int(labels[i]) - 1
        labelMatrix[i, y] = 1
    labels = labelMatrix
 
    
    if batchType != 'val':    
        trainImages = images[:-test,:]
        trainLabels = labels[:-test]
        testImages = images[images.shape[0] - test:,:] #save last amount for test
        testLabels = labels[labels.shape[0] - test:] #save last amount for test

    else:
        trainImages = images
        trainLabels = labels
        testImages = images
        testLabels = labels

    return images, labels, trainImages, trainLabels, testImages, testLabels     
        

def loadW(fileName):
    fileH5 = fileName + '.H5'
    fileJSON = fileName + '.JSON'
    if os.path.exists(fileH5) & os.path.exists(fileJSON) == True :
            json_file = open(fileJSON, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(fileH5)
            loaded_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
            model = loaded_model
            print("Loaded trained model")
    else:
        print("Tried loading weights, but failed")         
    return model

def saveW(model, fileName):
    fileH5 = fileName + '.H5'
    fileJSON = fileName + '.JSON'
    if os.path.exists(fileH5) & os.path.exists(fileJSON) == True :
        model_json = model.to_json()
        with open(fileJSON, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(fileH5, overwrite=True)
        #model.save_weights(fileWeights, overwrite=True)
        print("Saved Weights")
    else:
        print("Tried saving weights, but failed")
    
def cnnArchitecture3A(model, filters, epochs, convSize, imageAmount, imageDepth, imageRows, imageColumns, poolSize, neurons, classes):
    #Save Bottleneck Features
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, imageRows, imageColumns)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
        # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    
    
    return model
    
def cnnArchitecture3B(model, filters, epochs, convSize, imageAmount, imageDepth, imageRows, imageColumns, poolSize, neurons, classes):
    #Train top feautures
    
    
    
    return model    

    
def cnnArchitecture2(model, filters, epochs, convSize, imageAmount, imageDepth, imageRows, imageColumns, poolSize, neurons, classes, opt):
    print("CNN Archticture2")
    print("VGG Based ConvNet")

    model = Sequential()
    # input: 128x128 images with 3 channels -> (3, 128, 128) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(imageDepth, imageRows, imageColumns)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    
    return model
    
        
def cnnArchitecture1(model, filters, convSize, imageAmount, imageDepth, imageRows, imageColumns, poolSize, neurons, classes, opt):
    print("CNN Archticture1")
    
    
    model.add(Convolution2D(filters, convSize, convSize, input_shape=(3, imageRows, imageColumns)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    
    model.add(Convolution2D(filters, convSize, convSize))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    
    model.add(Convolution2D(2*filters, convSize, convSize))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(poolSize, poolSize))) 
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))   
    model.add(Dense(classes))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) 
    

    return model    
    
    
def cnnArchitecture0(model, filters, convSize, imageAmount, imageDepth, imageRows, imageColumns, poolSize, neurons, classes, opt):    
    print("CNN Archticture0")
    model.add(Convolution2D(filters, convSize, convSize,
                                border_mode='valid', input_shape=(3, imageRows, imageColumns)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Convolution2D(filters, convSize, convSize))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))
    model.add(Dropout(0.5))   
    model.add(Flatten())
    model.add(Dense(neurons))  
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    #model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) 
    

    return model   
    
def imageAugmentation(model, trainDir, validationDir, reshapeSize, batchSize, classMode, trainingSize, validationSize):
    
    trainDataGen = ImageDataGenerator(
                                       rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    testDataGen = ImageDataGenerator(rescale=1./255)
    """
    img = load_img('data/train1/1-structure/00016.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    #    and saves the results to the `preview/` directory
    i = 0
    for batch in trainDataGen.flow(x, batch_size=1,
                          save_to_dir='data', save_prefix='st16', save_format='jpeg'):
        i += 1
        if i > 20:
            break
    """
    
    
    trainGenerator = trainDataGen.flow_from_directory(
                        trainDir,  # this is the target directory
                        target_size=(reshapeSize, reshapeSize),  # all images will be resized to 150x150
                        batch_size= batchSize,
                        class_mode=classMode)  # since we use categorical_crossentropy loss, we need binary labels
    validationGenerator = testDataGen.flow_from_directory(
                                        validationDir,
                                        target_size=(reshapeSize, reshapeSize),
                                        batch_size= batchSize,
                                        class_mode=classMode)
    
    
    
    
    return trainGenerator, validationGenerator

    
def savePlot(hist, xAxis, yAxis1, yAxis2, fileName, fig=0, location=0):      
    
    title = str(yAxis1) + ' ' + str(fileName)
    x = range(xAxis)
    y1 = hist.history[yAxis1]
    y2 = hist.history[yAxis2]
    
    plt.figure(fig, figsize=(7,5)) 
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.xlabel('epochs')
    plt.ylabel(yAxis1)
    plt.legend([yAxis1, yAxis2], loc=location )
    plt.title(title)
    plt.savefig('graphs/' + title + '.png')
    
    plt.draw()
    plt.pause(0.0001)
  
def testResults(model, trainImages, testImages, testLabels):
    prob = model.predict(testImages)
    start = trainImages.shape[0]
    
    wrong = 0
    for i in xrange(0, testImages.shape[0], 1):
        
        prediction = np.argsort(prob[i])[-1] + 1
        #print("PRB", prob[i], prediction)      
        actual = np.argmax(testLabels[i]) + 1
        
        if actual != prediction:
            wrong = wrong + 1
            print("Image: " + str(start+i+1) + " Prediction: " + str(prediction)
                                    + " Actual: " + str(actual))
            """
            #i=9
            im = testImages[i].reshape(128, 128, 3)
            lb = testLabels[i]
            print("pred: ", prob[i])            
            print("label: ", lb)
            
            plt.figure(i, figsize=(7,5))
            plt.imshow(im)
            plt.show()
            plt.savefig('data/act' + str(start+i+1) + '.png')
            """
    frac = 100.00*wrong/testImages.shape[0]
    print("TEST SET: Number Wrong: " + str(wrong))
    print("TEST SET: Total Number: " + str(testImages.shape[0]))
    print("TEST SET: Fraction Wrong: " + str(frac))
    print("TEST SET: Fraction Right: " + str(100-frac))
    
def saveCSVfile(model, images, labels, csvFileName, csvFileName2, test):
        csvFileName = csvFileName + '.csv'
        csvFileName2 = csvFileName2 + '.csv'
        prob = model.predict(images)
        classes = model.predict_classes(trainImages, batch_size=32)

        wrong = 0
        with open(csvFileName, 'w') as csvFile:
            csvFile.write("Id" + ',' + 'Prediction' + '\n')
            j = 0
            for i in xrange(0, images.shape[0], 1):
                prediction = np.argsort(prob[i])[-1] + 1
                actual = np.argmax(labels[i]) + 1
                csvFile.write(str(i+1) + ',' + str(prediction) + '\n')
                j = i
                if actual != prediction:
                    wrong = wrong + 1
            j = j + 1
            print("J: " + str(j))
            for i in xrange(j, j+2000, 1):
                csvFile.write(str(i + 1) + ',' + str(0) + '\n')
            csvFile.close()
        frac = 100.00*wrong/images.shape[0]
        print("WHOLE SET: Number Wrong: " + str(wrong))
        print("WHOLE SET: Total Number: " + str(images.shape[0]))        
        print("WHOLE SET: Fraction Wrong: " + str(frac))
        print("WHOLE SET: Fraction Right: " + str(100-frac))
        print("Saved: " + csvFileName)
        
        wrong = 0
        with open(csvFileName2, 'w') as csvFile:
            csvFile.write("Id" + ',' + 'Prediction' + '\n')
            j = 0
            for i in xrange(0, images.shape[0] - test, 1):
                prediction = classes[i] + 1
                actual = np.argmax(labels[i]) + 1
                csvFile.write(str(i+1) + ',' + str(prediction) + '\n')
                j = i
                if actual != prediction:
                    wrong = wrong + 1
            j = j + 1
            print("J: " + str(j))
            for i in xrange(j, j+2000, 1):
                csvFile.write(str(i + 1) + ',' + str(0) + '\n')
            csvFile.close()
        frac = 100.00*wrong/images.shape[0]
        print("WHOLE SET: Number Wrong: " + str(wrong))
        print("WHOLE SET: Total Number: " + str(images.shape[0]))        
        print("WHOLE SET: Fraction Wrong: " + str(frac))
        print("WHOLE SET: Fraction Right: " + str(100-frac))
        print("Saved: " + csvFileName2)
        
        
        
        

def report(model, images, labels, targetNames):
    

    yPredict = model.predict(images)
    yPredict = np.argmax(yPredict, axis=1)    
                
    print(classification_report(np.argmax(labels, axis=1), yPredict, target_names = targetNames))
    score = model.evaluate(images, labels, show_accuracy=True, verbose=2)
    print('Test score:', '{0:0.3g}'.format(score[0]))
    print('Test accuracy:', '{0:0.4g}'.format(score[1]))        

if __name__ == '__main__':        
        
    targetNames = ['1-Structures', '2-Indoor', '3-People', '4-Animals',
                        '5-Plant Life', '6-Food', '7-Car', '8-Sea']        
    
    trainingLoss = 'loss'
    validationLoss = 'val_loss'
    trainingAccuracy = 'acc'
    validationAccuracy = 'val_acc'
                        
    val = "val"
    train = "train"
    dummy10 = "dummy10"
    dummy100 = "dummy100"
    dummy970 = "dummy970"
    
    defaultPath = 'data/'
    file0 = 'cnn0'
    file1 = 'cnn1'
    file2 = 'cnn2'
    file3 = 'cnn3'
    file4 = 'cnn4'
    
    csv1 = 'cnn1'
    csv2 = 'cnn2'
    csv3 = 'cnn3'
    csv4 = 'cnn4'
    
    
    classMode = 'categorical'
    imageRows = 128
    imageColumns = 128
    imageChannels = 3
    filters = 32
    poolSize = 2
    convSize = 3
    classes = 8
    neurons = 100
    
    
    #NETBL-----------------------------------------------------------
    sgd = SGD(lr=0.1, decay=1e-2, momentum=0.9, nesterov=True)
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    adadelta = adadelta(lr=0.01, rho=0.95, epsilon=1e-08, decay=0.0)
    
    batchType = train 
    epochs = 40
    test = 300
    batchSize = 500
    loadWeights = False
    saveWeights = True
    saveCSV = True
    reshapeSize = 128
    trainingSize = 14078 #14078 #6900 #7178
    validationSize = 101 #101 #100 #101
    trainDir= 'data/train3'
    validationDir= 'data/validation2'
    cnnType = 0
    fileName = trainCSVfileName = file0
    trainCSVdummy = 'sample_submission0'
    valCSVfileName = 'sample_submission'
    valCSVfileName2 = 'sample_submission2'
    
    plotName = str(fileName + ' Ep' + str(epochs)+ ' T'+ str(test)+ 
                ' B'+ str(batchSize)+ ' L'+ str(loadWeights))
    
    
    #---------------START PROGRAM
    #saveBW(defaultPath, batchType)
    #splitImages(defaultPath, batchType)
    
    labels = openCSV(defaultPath, batchType)
    images, _ = openImage(defaultPath, batchType)
    
    #images, _ = openImageBW(defaultPath, batchType)
    #images, labels, trainImages, trainLabels, testImages, testLabels = processDataBW(images, labels, batchType, test, classes)
    
    
    imageAmount = images.shape[0]
    imageRows = images.shape[1]
    imageColumns = images.shape[2]
    imageDepth = images.shape[3]
    
    """
    i=9
    im = images[i].reshape(128, 128, 3)
    lb = labels[i]
    print("label: ", lb)
    print("label.shape", labels.shape)
    plt.imshow(im)
    """
      
    
    images, labels, trainImages, trainLabels, testImages, testLabels = processData(images, labels, batchType, test, classes)    
        
        
    
    model = Sequential()
    
    i=2
    imgNum = trainImages.shape[0]+i+1
    im = testImages[i].reshape(128, 128, 3)
    lb = testLabels[i]
    #print("pred: ", prob[i])            
    print("image: ", imgNum, " label: ",lb)
        
    plt.figure(i, figsize=(7,5))
    plt.imshow(im)
    plt.show()
    plt.savefig('data/act' + str(imgNum) + '.png')
    
    imgNum = 0+i+1
    im = trainImages[i].reshape(128, 128, 3)
    lb = trainLabels[i]
    #print("pred: ", prob[i])            
    print("image: ", imgNum, " label: ",lb)
        
    plt.figure(i, figsize=(7,5))
    plt.imshow(im)
    plt.show()
    plt.savefig('data/act' + str(imgNum) + '.png')
    
    
    
    if loadWeights == True: #ONLY LOADING NOT GOING TO SAVE
            mod = loadW(fileName)
            #hist = mod.fit(trainImages, trainLabels, batch_size = batchSize, nb_epoch=epochs,
            #                show_accuracy=True, verbose=1, validation_data=(testImages, testLabels))
            
    else: #NOT LOADED. MAY SAVE NEW WEIGHTS
        print("Did not load weights")
        if cnnType == 0:
            mod = cnnArchitecture0(model, filters, convSize, imageAmount, imageDepth, imageRows,
                         imageColumns, poolSize, neurons, classes, sgd)
            #hist = mod.fit(trainImages, trainLabels, batch_size = batchSize, nb_epoch=epochs,
            #            show_accuracy=True, verbose=1, validation_data=(testImages, testLabels))
            #hist = mod.fit(trainImages, trainLabels, batch_size = batchSize, nb_epoch=epochs,
            #            show_accuracy=True, verbose=1, validation_split=0.2)
            trainGenerator, validationGenerator = imageAugmentation(model,
                        trainDir, validationDir, reshapeSize, batchSize, classMode, 
                        trainingSize, validationSize)
    
            hist = mod.fit_generator(trainGenerator,
                                   samples_per_epoch=trainingSize,
                                   nb_epoch=epochs,
                                   validation_data=validationGenerator,
                                   nb_val_samples=validationSize)
            mod.summary()
            a = mod.get_weights()
            print(len(a))
            print("1", a[0].shape)
            print("2", a[1].shape)
            print("3", a[2].shape)
            print("4", a[3].shape)
            print("5", a[4].shape)
            print("6", a[5].shape)
            print("7", a[6].shape)
            print("8", a[7].shape)
            w1, b1, w2, b2, w3, b3, w4, b4 = mod.get_weights()
            plt.pcolormesh(w4)
       
        if cnnType == 1:
             mod = cnnArchitecture1(model, filters, convSize, imageAmount, imageDepth, imageRows,
                     imageColumns, poolSize, neurons, classes, rmsprop)

             trainGenerator, validationGenerator = imageAugmentation(model,
                        trainDir, validationDir, reshapeSize, batchSize, classMode, 
                        trainingSize, validationSize)
    
             hist = mod.fit_generator(trainGenerator,
                                   samples_per_epoch=trainingSize,
                                   nb_epoch=epochs,
                                   validation_data=validationGenerator,
                                   nb_val_samples=validationSize)
             mod.summary()
             a = mod.get_weights()
             print(len(a))
             print("1", a[0].shape)
             print("2", a[1].shape)
             print("3", a[2].shape)
             print("4", a[3].shape)
             print("5", a[4].shape)
             print("6", a[5].shape)
             print("7", a[6].shape)
             print("8", a[7].shape)
             print("9", a[8].shape)
             print("10", a[9].shape)
             w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = mod.get_weights()
             plt.pcolormesh(w5, figure=1)
    
        if cnnType == 2:     
             mod = cnnArchitecture2(model, filters, epochs, convSize, imageAmount, imageDepth, imageRows,
                     imageColumns, poolSize, neurons, classes, sgd)
             hist = mod.fit(trainImages, trainLabels, batch_size=32, nb_epoch=epochs,
                            show_accuracy=True, verbose=1, validation_data=(testImages, testLabels))
             mod.summary()
             a = mod.get_weights()
             print(len(a))
             print("1", a[0].shape)
             print("2", a[1].shape)
             print("3", a[2].shape)
             print("4", a[3].shape)
             print("5", a[4].shape)
             print("6", a[5].shape)
             print("7", a[6].shape)
             print("8", a[7].shape)
             print("9", a[8].shape)
             print("10", a[9].shape)
             print("11", a[10].shape)
             print("12", a[11].shape)
             w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6 = mod.get_weights()
             plt.pcolormesh(w6)
             

        if saveWeights == True:
            saveW(mod, fileName)
        
    
        savePlot(hist, epochs, trainingLoss, validationLoss, plotName, 2)
        savePlot(hist, epochs, trainingAccuracy, validationAccuracy, plotName, 3, 4)
    
    
    #print("Classes, " + str(classes[0]))
    
    testResults(mod, trainImages, testImages, testLabels)
    if saveCSV == True:
        saveCSVfile(mod, images, labels, trainCSVfileName, trainCSVdummy, test)
    report(mod, images, labels, targetNames)   
       
    
    #-----------Doing Validation test--------------
    
    
    labels = openCSV(defaultPath, val)
    images, _ = openImage(defaultPath, val)
    imageAmount = images.shape[0]
    imageRows = images.shape[1]
    imageColumns = images.shape[2]
    imageDepth = images.shape[3]
    images, labels, trainImages, trainLabels, testImages, testLabels = processData(images, labels, batchType, test, classes)     
    testResults(mod, trainImages, testImages, testLabels)
    if saveCSV == True:
        saveCSVfile(mod, images, labels, valCSVfileName, valCSVfileName2, test)
    report(mod, images, labels, targetNames)      

   
