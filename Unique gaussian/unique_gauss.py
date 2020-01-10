#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys

if(len(sys.argv)<3):
    print("You have to specify the path as argument and  the dataset for train size")
    exit()
path = sys.argv[1]
size= int(sys.argv[2])

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
        self.mean=0.0
        self.diff = np.zeros((height, width), np.double)


    def calculate(self, frame):

        if self.first:
            self.prevFrame = self.mus = frame
            for i in range(self.height):
                for j in range(0, self.width):
                    self.sigmas[i][j]=127
            self.first = False
            return
        self.diff=cv2.absdiff(frame,self.mus)
        self.diff/=(self.sigmas)**0.5
        sigmas=self.alpha*abs(self.mus - frame)**2+(1-self.alpha)*self.sigmas
        mus=self.alpha*frame+(1-self.alpha)*self.mus
        
        for i in range(self.height):
                for j in range(0, self.width):
                    if(self.diff[i][j]<6.25):
                        self.mus[i][j]=mus[i][j]
                        self.sigmas[i][j]=sigmas[i][j]

        self.prevFrame = frame

    def substract(self, frame):
        diff = np.zeros((self.height, self.width), np.double)
        for i in range(self.height):
                for j in range(0, self.width):
                    if(self.diff[i][j]>6.25):
                        diff[i][j]=255
        return diff
        


(h, w) = np.shape(cv2.imread(path + '1.png', 0))
subtractor = MG(height=h, width=w)
framecnt=0
for i in range(1, 1000 - 2):
    print ('frame ' + str(i))
    image0 = cv2.imread(path + str(i) + '.png')
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    #image0 = cv2.GaussianBlur(image0, (5, 5), 0)

    # image0 = cv2.resize(image0, (100, 100),                       interpolation=cv2.INTER_AREA)

    image0 = image0.astype(float)
    if(i<size or i % 5 ==0 ):
            framecnt+=1
            subtractor.beta=subtractor.alpha=1/framecnt
            subtractor.calculate(image0)


        # image1 = cv2.resize(image1, (w,h),interpolation=cv2.INTER_AREA)

    image1 = subtractor.substract(image0)

    cv2.imshow('', image1)
    cv2.waitKey(1)


			
