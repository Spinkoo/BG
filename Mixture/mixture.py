import sys
import os
import numpy as np
import cv2
from scipy.stats import multivariate_normal


if(len(sys.argv)<2):
    print("You have to specify the path as argument ")
    exit()
path = sys.argv[1]
def soustaction(image0, image1):

    seuil=5
    h, w = image0.shape
    img_diff = np.zeros((h, w), dtype=np.uint8)

    for i in range(0, h):
        for j in range(0, w):
            diff = int(abs(int(image1[i, j]) - int(image0[i, j])))

            # diff = int(int(image1[i, j]) - int(image0[i, j]))
            if (diff >= seuil):
                img_diff[i, j] = 255
                #img_diff[i, j] = int(image1[i, j])

    #cv2.imshow("img_diff", img_diff)
    return img_diff


class MOG():
    def __init__(self,nGausses=4,
        bg_prob=0.6, 
        learnr=0.01,
        height=1,
        width=1):

        self.height=height
        self.width=width
        self.nGausses=nGausses
        self.n_mu=np.zeros((self.height,self.width, self.nGausses,3)) 
        self.sigmas=np.zeros((self.height,self.width,self.nGausses)) 
        self.weights=np.zeros((self.height,self.width,self.nGausses))
        self.bg_prob=bg_prob
        self.lr=learnr
        
 
        for i in range(self.height):
            for j in range(self.width):
                self.n_mu[i,j]=np.array([[125, 125, 125]]*self.nGausses)
                self.sigmas[i,j]=[10.0]*self.nGausses
                self.weights[i,j]=[1.0/self.nGausses]*self.nGausses
   


    def updateMOG(self, frame, BG):
        labels=np.zeros((self.height,self.width))
        for i in range(self.height):
            for j in range(self.width):
                val=frame[i,j]
                match=-1
                for k in range(self.nGausses):
                    covInv=np.linalg.inv(self.sigmas[i,j,k]*np.eye(3))
                    X_mu=val-self.n_mu[i,j,k]
                    diff=np.dot(X_mu.T, np.dot(covInv, X_mu))
                    if diff<6.25*self.sigmas[i,j,k]:
                        match=k
                        break
                if match!=-1:  
                    self.weights[i,j]=(1.0-self.lr)*self.weights[i,j]
                    self.weights[i,j,match]+=self.lr
                    alpha=self.lr * multivariate_normal.pdf(val,self.n_mu[i,j,match],np.linalg.inv(covInv))
                    self.sigmas[match]=(1.0-alpha)*self.sigmas[i,j,match]+alpha*np.dot((val-self.n_mu[i,j,match]).T, (val-self.n_mu[i,j,match]))
                    self.n_mu[i,j,match]=(1.0-alpha)*self.n_mu[i,j,match]+alpha*val
                    if match>BG[i,j]:
                        labels[i,j]=255
                else:
                    #no match so bg
                    self.n_mu[i,j,-1]=val
                    labels[i,j]=255
        return labels


    def updateParams(self):
        BG=np.zeros((self.height,self.width),dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                BG[i,j]=-1
                ratios=[]
                for k in range(self.nGausses):
                    ratios.append(self.weights[i,j,k]/np.sqrt(self.sigmas[i,j,k]))
                indices=np.array(np.argsort(ratios)[::-1])
                self.n_mu[i,j]=self.n_mu[i,j][indices]
                self.sigmas[i,j]=self.sigmas[i,j][indices]
                self.weights[i,j]=self.weights[i,j][indices]
                Probcummul=0
                for l in range(self.nGausses):
                    Probcummul+=self.weights[i,j,l]
                    if Probcummul>=self.bg_prob and l<self.nGausses-1:
                        BG[i,j]=l
                        break
                if BG[i,j]==-1:
                    BG[i,j]=self.nGausses-2
        return BG
  
            
    def work(self):
      
        frameCnt=0
        
        
        while(True):
            frame = cv2.imread(path + str(frameCnt+1) + ".png",1)
            print(frameCnt)
            h,w,_=np.shape(frame)

            frame = cv2.resize(frame, (100,100),interpolation=cv2.INTER_AREA)
            BGs=self.updateParams()
            labels=self.updateMOG(frame,BGs)
            labels = cv2.resize(labels, (200,200),interpolation=cv2.INTER_AREA)
            cv2.imshow("",labels)
         
            cv2.waitKey(1)
            frameCnt+=1

        
    
        cv2.destroyAllWindows()
        
        
        

h,w=np.shape(cv2.imread(path+"1.png",0))
subtractor=MOG(width=100,height=100)
subtractor.work()
    
