import easyocr
import cv2
import os
import skimage.morphology
import numpy as np
import math
import random

def easyocr_get_centroids(path_or_img):
    easyocr_reader = easyocr.Reader(['en']) # need to run only once to load model into memory
    result = easyocr_reader.readtext(path_or_img)
    if isinstance(path_or_img,str):
        img = cv2.imread(path_or_img)
    else:
        img = path_or_img
    extracted = []
    for i in result:
        pts = i[0]
        print(i[1])
        cX,cY = int((pts[0][0]+pts[2][0])/2), int((pts[0][1]+pts[2][1])/2)
        cv2.rectangle(img,tuple(pts[0]),tuple(pts[2]),(123,0,12),2)
        cv2.circle(img, (cX,cY),5, (123,0,12))
        data = {
            "text":i[1],
            "center":(cX,cY),
            "data":i
        }
        extracted.append(data)
    cv2.imwrite("Lol.jpg", img)
    return extracted


def img_find_lines(img):
    lines = []
    height,width = img.shape[0], img.shape[1]
    #canny = cv2.Canny(img, 100,200)
    rho = 3
    theta = np.pi/180
    hline_thresh = 20
    hline_min_line_length = min(height, width)*0.1
    hline_max_line_gap = 10 #min(height, width)*0.12
    all_lines = cv2.HoughLinesP(img, rho, theta, 
                           hline_thresh, None, 
                           hline_min_line_length, hline_max_line_gap) #TODO
    for i,line in enumerate(all_lines):
      for x1,y1,x2,y2 in line:
        length= math.sqrt((x1-x2)**2 + (y1-y2)**2)
        lines.append({"label":f"line_{i}", "coordinates":(x1,y1,x2,y2), "length":length})
    return lines, draw_lines(cv2.fastNlMeansDenoising(img,None,15,15,25), lines)

def draw_lines(img, coords, thickness=10):
    for i in coords:
        x1,y1,x2,y2 = i["coordinates"]
        color = 0#random.randint(250,255)
        cv2.line(img,(x1,y1),(x2,y2),color,thickness)
    return img



threshed = img_load("data/hand.png")[2]
lns = img_find_lines(threshed)[1]
#easyocr_get_centroids(lns)
cv2.imwrite("lol.jpg", lns)