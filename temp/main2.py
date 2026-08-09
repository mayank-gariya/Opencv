import cv2 as cv
import matplotlib.pyplot as plt
import os 
import numpy as np

def grayimg():
    imgPath = input('entre your image path .....')
    outputpath = input('entre your output image path .....')
    
    image = cv.imread(imgPath)
    
    user_input = int(input('entre what you wanna do 1. show image , 2.covert to gray , 3. save or not  or 4 all '))
    if user_input == 4:
        cv.imshow('some gray dog',gray)
        cv.waitKey(0)
        cv.destroyAllWindows()
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        cv.imwrite(outputpath,gray)
        
    if user_input == 1:
        cv.imshow('some gray dog',gray)
        cv.waitKey(0)
        cv.destroyAllWindows()
    elif user_input == 2:
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    else:
        cv.imwrite(outputpath,gray)


grayimg()
    
    