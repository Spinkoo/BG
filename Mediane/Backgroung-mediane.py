import cv2
import numpy as np
import sys
kernel = np.ones((5,5),np.float32)/25

taille_dataset = 1000

buff = []
cpt = 3
seuil = 5
if(len(sys.argv)<2):
    print("You have to specify the path as argument ")
    exit()
path = sys.argv[1]
def soustaction(image0, image1):

    h, w,_ = image0.shape
    img_diff = np.zeros((h, w,3), dtype=np.uint8)

    for i in range(0, h):
        for j in range(0, w):
            diff = int(abs(int(image1[i, j,0]) - int(image0[i, j,0]))), int(abs(int(image1[i, j,1]) - int(image0[i, j,1]))), int(abs(int(image1[i, j,2]) - int(image0[i, j,2])))

            # diff = int(int(image1[i, j]) - int(image0[i, j]))
            if (diff[0] >= seuil and diff[1] >= seuil and diff[2] >= seuil ):
                img_diff[i, j] = 255
                #img_diff[i, j] = int(image1[i, j])

    #cv2.imshow("img_diff", img_diff)
    return img_diff

def get_mediane_buff(buff):
   
    median = np.median(buff, axis=0).astype(dtype=np.uint8)    


    return median



buff=[]

for i in range(1, taille_dataset-2):
    image0 = cv2.imread(path + str(i) + ".png",1)
    #image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    image0 = cv2.GaussianBlur(image0, (5, 5), 0)

    buff.append(image0)
   

    h, w,_ = image0.shape
    img_diff = np.zeros((h, w,3), dtype=np.uint8)
    print("i = " + str(i))
    if(i==cpt):
        img_mask = get_mediane_buff(buff)

    if (i % cpt != 0) :
        if i >= cpt:
            img_diff = soustaction(img_mask, image0)  #si on veut afficher chaque frame sinon on supp cette ligne
            cv2.imshow("img_mask", img_mask)
            continue

    elif i >= cpt: #else if(i!=1)   (i-1) % 3 == 0

        img_mask = get_mediane_buff(buff)
        buff.clear()
        img_diff = soustaction(img_mask, image0)
        cv2.imshow("img_mask", img_mask)


    cv2.waitKey(1)


    cv2.imshow("img_diff", img_diff)

cv2.waitKey(0)
cv2.destroyAllWindows()