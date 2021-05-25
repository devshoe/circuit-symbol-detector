import cv2
import os
import skimage.morphology
import numpy as np
import math
import random
import sys
import easyocr
import keras
import shutil
from tabulate import tabulate
import argparse
# from prettyprint import pp

#np.set_printoptions(threshold=sys.maxsize)
CLASS_MAP = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
MODEL_PATH = "data/full_model.h5"
loaded_model = keras.models.load_model(MODEL_PATH) #this is our model
BASE_SIZE = 1000
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
    w,h = start_image.shape[0],start_image.shape[1]
    print(start_image.shape)

    start_image = cv2.resize(start_image,(BASE_SIZE,int(BASE_SIZE*w/h)))
    print(start_image.shape)
    img = start_image
    if len(img.shape)==3:
        grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayscaled = img
    grayscaled = cv2.fastNlMeansDenoising(grayscaled,None,15,15,20)
    thresholded = img_threshold(grayscaled)
    thresholded = cv2.dilate(thresholded, np.ones((3,3)),iterations=1)
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
    for i in range(1,nb_components):
        if i != max_label:
            img3[output==i] = 255
    return img2.astype("uint8"),img3.astype("uint8") #2 is ckt, 3 is text

def easyocr_get_centroids(path_or_img):
    easyocr_reader = easyocr.Reader(['en']) # this is easy_ocr
    result = easyocr_reader.readtext(path_or_img)
    if isinstance(path_or_img,str):
        img = cv2.imread(path_or_img)
    else:
        img = path_or_img
    extracted = []
    for i in result:
        pts = i[0]
        cX,cY = int((pts[0][0]+pts[2][0])/2), int((pts[0][1]+pts[2][1])/2)
        try: cv2.rectangle(img,tuple(pts[0]),tuple(pts[2]),(123,0,12),2)
        except:pass
        # cv2.circle(img, (cX,cY),5, (123,0,12))
        cv2.putText(img, i[1], (cX,cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),5)
        data = {
            "text":i[1],
            "center":(cX,cY),
            "data":i
        }
        extracted.append(data)

    return extracted, img

def predict_single_char_model(img):
    img = cv2.resize(img, (28,28))
    cv2.imshow("a",img)
    cv2.waitKey()
    
    img = img.reshape(1,28,28,-1)

    prediction = list(loaded_model.predict(img)[0]) #find index of 1
    print(max(prediction))
    return CLASS_MAP[prediction.index(max(prediction))]


def separated_chars_and_predict(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)    

    for i,cnt in enumerate(contours):
        img = np.full((500,500),255,"uint8")
        cv2.drawContours(img, contours, i, 0,cv2.FILLED)
        cv2.imshow("a",img)
        cv2.waitKey()
        print(predict_single_char_model(img))

def length(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def find_lines(img, vert_group_size=BASE_SIZE/8, hor_group_size=BASE_SIZE/8):
    canny = cv2.Canny(img, 100,200)
    height,width = img.shape[0], img.shape[1]
    hline_thresh = 90
    hline_min_line_length = min(height, width)*0.2
    hline_max_line_gap = min(height, width)*0.2
    all_lines = cv2.HoughLinesP(canny, 1, np.pi/180, 
                           hline_thresh, None, 
                           hline_min_line_length, hline_max_line_gap) #TODO
    lines = []
    for i,line in enumerate(all_lines):
      for x1,y1,x2,y2 in line:
        length= math.sqrt((x1-x2)**2 + (y1-y2)**2)
        slope = (y2-y1)/(x2-x1)
        center = (int((x1+x2)/2),int((y1+y2)/2))
        orientation = ""
        grouping = 0
        if abs(slope) != math.inf: 
            slope = round(slope)
            if slope ==0: 
                orientation = "horizontal"
                grouping = center[1] - (center[1]%hor_group_size)
            else:
                orientation = "vertical"
                grouping = center[0] - (center[0]%vert_group_size)
        else:
            orientation = "vertical"
            grouping = center[0] - (center[0]%vert_group_size)
        
        line_info = {
            "label":f"B{i}", 
            "coordinates":(x1,y1,x2,y2), 
            "length":length, 
            "center":center,
            "slope": slope,
            "orientation":orientation,
            "grouping":grouping,
            "components":[], #like ["4uH", "10R"]
            "intersections": [], #like ["b1","b3"]
            "merge": [],
            }
        lines.append(line_info)
    return line_cleanup(lines)

def draw_lines(img, lines, thickness=5):
    for i, line in enumerate(lines):
        x1,y1,x2,y2 = line["coordinates"]
        cv2.line(img,(x1,y1),(x2,y2),(random.randint(0,255),random.randint(0,255),random.randint(0,255)),thickness)
        cv2.putText(img, f"B{i}", line["center"], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),5)
    return img
'''
okay to merge lines, they must have same orientation, ie angle < 10 degrees

'''
def line_cleanup(lines):
    cleaned = []
    grouped = {}
    if len(lines)<2:
        return lines
    vertical_lines = [l for l in lines if l["orientation"] in ["angled","vertical"]]
    vertical_groups = {}
    horizontal_lines = [l for l in lines if l["orientation"]=="horizontal"]
    horizontal_groups = {}
    for k in vertical_lines:
        if k["grouping"] not in vertical_groups.keys():
            vertical_groups[k["grouping"]] = [k]
        else:
            vertical_groups[k["grouping"]].append(k)
    for k in horizontal_lines:
        if k["grouping"] not in horizontal_groups.keys():
            horizontal_groups[k["grouping"]] = [k]
        else:
            horizontal_groups[k["grouping"]].append(k)

    for group,lines in vertical_groups.items():
        x,y = [],[]
        for l in lines:
            x.append(l["coordinates"][0])
            x.append(l["coordinates"][2])
            y.append(l["coordinates"][1])
            y.append(l["coordinates"][3])
        avg = int(sum(x)/len(x))
        line_info = {
            "label":f"B{len(cleaned)}", 
            "coordinates":(avg,min(y),avg,max(y)), 
            "center":(avg, int((min(y)+max(y))/2)),
            "components":[], #like ["4uH", "10R"]
            "intersections": [], #like ["b1","b3"]
            "merge": [],
            }
        cleaned.append(line_info)
    print(len(vertical_groups), vertical_groups.keys())
    for group,lines in horizontal_groups.items():
        x,y = [], []
        for l in lines:
            x.append(l["coordinates"][0])
            x.append(l["coordinates"][2])
            y.append(l["coordinates"][1])
            y.append(l["coordinates"][3])
        avg = int(sum(y)/len(y))
        line_info = {
            "label":f"B{len(cleaned)}", 
            "coordinates":(min(x),avg,max(x),avg), 
            "center":(int((min(x)+max(x))/2), avg),
            "components":[], #like ["4uH", "10R"]
            "intersections": [], #like ["b1","b3"]
            "merge": [],
            }
        cleaned.append(line_info)
    return cleaned

#comps is all the text, branches are lines.
def map_comps_to_branches(comps, branches):
    for comp in comps:
        closest,amt=0,1000000
        for j,b in enumerate(branches):
            l = length(comp["center"][0],comp["center"][1],b["center"][0], b["center"][1])
            if l<amt:
                amt = l
                closest = j
        branches[closest]["components"].append(comp)
        branches[closest]["comp_loc"] = comp["center"]
    return branches

def intersections(branches):
    for i,b1 in enumerate(branches):
        x1,y1,x2,y2 = b1["coordinates"]
        for j,b2 in enumerate(branches):
            if j==i: continue
            x3,y3,x4,y4 = b2["coordinates"]
            l11 = point_is_on_line(x1,y1,x2,y2,x3,y3)
            l12 = point_is_on_line(x1,y1,x2,y2, x4,y4)
            l21 = point_is_on_line(x3,y3,x4,y4,x1,y1)
            l22 = point_is_on_line(x3,y3,x4,y4, x2,y2)
            if l11 or l12 or l21 or l22:
                branches[i]["intersections"].append(b2["label"])
    return branches

def point_is_on_line(lx1,ly1,lx2,ly2,x,y,tolerance=BASE_SIZE/10):
  return (x<max(lx1,lx2)+tolerance and x>min(lx1,lx2)-tolerance) and (y<max(ly1,ly2)+tolerance and y>min(ly1,ly2)-tolerance)

def connect_matrix(branches):
    branch_dict = {b["label"]:b for b in branches}
    final_list = {}
    for branch_name,branch in branch_dict.items():
        for component_on_branch in branch["components"]:
            final_list[component_on_branch["text"]] = []
            for i in branch["intersections"]:
                components_on_connected_branch = [j["text"] for j in branch_dict[i]["components"]]
                locs_of_comps = [j["center"] for j in branch_dict[i]["components"]]
                distances_from_branch_center = [length(branch["center"][0],branch["center"][0], i[0], i[1] )for i in locs_of_comps]
                comps_to_append = []
                if len(components_on_connected_branch)>1:#handle connection cases here
                    if abs(distances_from_branch_center[0] - distances_from_branch_center[1]) < BASE_SIZE * 0.1:
                       comps_to_append = components_on_connected_branch
                    else:
                        comps_to_append = [components_on_connected_branch[distances_from_branch_center.index(min(distances_from_branch_center))]]
                else:
                    comps_to_append = components_on_connected_branch
                final_list[component_on_branch["text"]] += comps_to_append
    return final_list


def full(imgpath):
    try: 
        shutil.rmtree("results")
    except:
        pass
    os.mkdir("results")
    i=img_load(imgpath)  
    cpy = i[0].copy()
    ckt = extract_circuit(i[2])

    cv2.imwrite("results/0original.jpg", i[0])
    cv2.imwrite("results/1grayscaled.jpg",i[1])
    cv2.imwrite("results/2thresholded.jpg",i[2])
    cv2.imwrite("results/3text.jpg", ckt[1])
    cv2.imwrite("results/3circuit.jpg", ckt[0])
    lines = find_lines(i[0])
    cv2.imwrite("results/4lines.jpg",draw_lines(i[0],lines))
    pts, img = easyocr_get_centroids(cpy)
    b = intersections(map_comps_to_branches(pts, lines))

    print("branch connections:")
    connections = {x["label"]:x["intersections"] for x in b}
    print(connections)

    print("components mapped to branches:")
    comps = {x["label"]:[l["text"] for l in x["components"]] for x in b}
    print(comps)
    order = []
    for val in comps.values():
        order+=val
    print("connection matrix:")
    conns = connect_matrix(b)
    rows = []
    print(order)
    for i in order:
        connections = conns[i]
        idxes = []
        out = [i]
        for x in connections: idxes.append(order.index(x))
        for index,j in enumerate(order):
            if j == i: out.append("x")
            elif index in idxes: out.append("1")
            else: out.append("0")
        rows.append(out)
    print(tabulate(rows, headers=order))
    cv2.imwrite("results/3ocr.jpg", img)

if __name__ == "__main__":
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument("-p", "-path", required=True)
    # args = argparser.parse_args()
    full("sample3.jpeg")