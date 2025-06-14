import cv2 
import numpy as np
img = cv2.imread('/home/radwa/Documents/assets/steel_beam_03.jpg')
gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

# SHI-Tomasi 

corners = cv2.goodFeaturesToTrack(gray_img , maxCorners= 100 , qualityLevel= 0.1 , minDistance= 50)
corners = np.intp(corners)
for c in corners:
    x , y = c.ravel()
    img = cv2.circle( img , center = (x , y ) , radius = 20 , color= ( 0,0,255) , thickness= -1 )


# Harris 
corners = cv2.goodFeaturesToTrack(gray_img , maxCorners= 100 , qualityLevel= 0.01 , minDistance= 50 ,
                                    useHarrisDetector=True , k=0.1)
corners = np.intp(corners)

for c in corners:
    x , y = c.ravel()
    img = cv2.circle( img , center = (x , y ) , radius = 10 , color= ( 0,254,0) , thickness= -1 )


cv2.imshow('Beam' , img )
cv2.waitKey(0)
cv2.destroyAllWindows