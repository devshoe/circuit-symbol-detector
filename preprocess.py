import cv2
import os
import skimage.morphology
import numpy as np
import math
import random
import sys

#np.set_printoptions(threshold=sys.maxsize)

#img_load takes in a image and returns 4 images
#[0] the original RGB or otherwise
#[1] grayscaled, essentially 2D, with values ranging from 0 to 255
#[2] thresholded, 2D but binarized, values are either 0 or 255
#[3] skeletonized, morphologized to get single pixel widths
def img_load(image_path):
    start_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) #don't touch this.
    if start_image.shape[-1] == 4: #if it has alpha channel
      trans_mask = start_image[:,:,3] == 0 #mask wherever alpha is 0
      start_image[trans_mask] = [255, 255, 255, 255]
      start_image = cv2.cvtColor(start_image, cv2.COLOR_BGRA2BGR)
    start_image = cv2.resize(start_image,(500,500))
    img = start_image
    if len(img.shape)==3:
        grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayscaled = img
    grayscaled = cv2.fastNlMeansDenoising(grayscaled,None,15,15,20)
    thresholded = img_threshold(grayscaled)
    #thresholded = cv2.dilate(thresholded, np.ones((3,3)),iterations=2)
    skeleton = skimage.morphology.skeletonize(thresholded/255).astype("uint8")*255
    return start_image, grayscaled, thresholded, skeleton

#img_threshold is slightly problematic
#presently counts grayscale pixel occurences and takes decision based on sums
def img_threshold(img, sum_ratio=0.033):
    occurence = [0 for i in range(256)]
    for row in img:
        for elem in row:
            occurence[elem]+=1
    threshold_value = 90 #85 and 150ish
    ratios = []
    for idx,val in enumerate(occurence):
        try:
            ratios.append(sum(occurence[:idx])/sum(occurence[idx:]))
        except:
            ratios.append(0)
        if ratios[-1]>sum_ratio:
            threshold_value = idx
            break

    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    print(threshold_value)
    _, thresholded = cv2.threshold(img, thresh=threshold_value, maxval=255, type=method)
    return thresholded

def get_contour_mask(img, reject_area_below=25):
    cv2.imwrite("dil.jpg",img)
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    
    img_right_y,img_top_x = img.shape[0]-1, img.shape[1]-1
    component_cnt = []
    text_cnts = []
    for i,cnt in enumerate(contours):
        appended = False
        for pt in cnt:
            if pt[0][1] in [0,img_right_y] or pt[0][0] in [0,img_top_x]:
                component_cnt.append(cnt)
                appended=True
                break
        if appended:
            appended=False
        else:
            text_cnts.append(cnt)
    
    #new_contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    

    return component_cnt,text_cnts

def mask():
    comp,text = get_contour_mask(i[2])
    img = np.zeros((500,500),"uint8")
    cv2.drawContours(img, text, -1, 255,cv2.FILLED)
    print(len(text))
    
def extract_circuit(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img,connectivity=4)
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    img2 = np.zeros(output.shape)
    img3 = np.zeros(output.shape)
    img2[output == max_label] = 255
    print(output)
    for i in range(1,nb_components):
        if i != max_label:
            img3[output==i] = 255
    return img2,img3

if __name__ == "__main__":
    i=img_load('data/ckt.png')  
    ckt = extract_circuit(i[2])
    #skeleton = skimage.morphology.skeletonize(ckt/255).astype("uint8")*255
    # skeleton=draw_lines(skeleton, img_find_lines(skeleton)[0])
    cv2.imwrite("circuit.jpg", ckt[0])
    cv2.imwrite("text.jpg", ckt[1])