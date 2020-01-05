#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys

if(len(sys.argv)<2):
    print("You have to specify the path as argument ")
    exit()
path = sys.argv[1]

class MG:

    def __init__(
        self,
        numOfGauss=1,
        height=100,
        width=100,
        ):

        self.width = width
        self.height = height
        self.numOfGauss = numOfGauss
        self.mus = np.zeros((height, width), np.double)
        self.sigmas = np.zeros((height, width), np.double)
        self.alpha = 0.1
        self.first = True
        self.beta=0.1

    def calculate(self, frame):

        if self.first:
            self.prevFrame = self.mus = frame
            self.first = False
            return
        for i in range(self.height):
            for j in range(0, self.width):
                self.sigmas[i][j] = self.beta * (self.prevFrame[i][j]
                        - self.mus[i][j]) ** 2 + (1 - self.beta) \
                    * self.sigmas[i][j]

        for i in range(0, self.height):
            for j in range(0, self.width):

                self.mus[i][j] = self.alpha * self.prevFrame[i][j] + (1
                        - self.alpha) * self.mus[i][j]
        self.prevFrame = frame

    def substract(self, frame):

        diff = np.zeros((self.height, self.width), np.double)
        for i in range(self.height):
            for j in range(self.width):
                T = 6.25* self.sigmas[i][j]**0.5
                if abs(self.mus[i][j] - frame[i][j]) > T:
                    diff[i][j] = 255
        return diff


(h, w) = np.shape(cv2.imread(path + '1.png', 0))
subtractor = MG(height=h, width=w)

for i in range(1, 1000 - 2):
    print ('frame ' + str(i))
    image0 = cv2.imread(path + str(i) + '.png')
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    #image0 = cv2.GaussianBlur(image0, (5, 5), 0)

    # image0 = cv2.resize(image0, (100, 100),                       interpolation=cv2.INTER_AREA)

    image0 = image0.astype(float)
    subtractor.beta=subtractor.alpha=1/i
    if(i<80 or i%5==0):
    		subtractor.calculate(image0)


        # image1 = cv2.resize(image1, (w,h),interpolation=cv2.INTER_AREA)

    image1 = subtractor.substract(image0)

    cv2.imshow('', image1)
    cv2.waitKey(1)


			