import keras
import cv2
import numpy as np
import random
import easyocr

CLASS_MAP = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
MODEL_PATH = "data/full_model.h5"

loaded_model = keras.models.load_model(MODEL_PATH)
loaded_digits = np.array(cv2.imread("data/digits.png"))
digit_cells = np.array([np.hsplit(row,100) for row in np.vsplit(loaded_digits,50)])

def predict_single_char_model(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28,28))
    img = img.reshape(1,28,28,-1)
    prediction = list(loaded_model.predict(img)[0]) #find index of 1
    return CLASS_MAP[prediction.index(max(prediction))]


    
def predict_component_knn(img):
    return

def imshow(img):
    cv2.imshow(img)
    cv2.waitKey()

def imwrite(img,path):
    cv2.imwrite(path,img)

def find_random_digit_pic(num):
    num = num%10
    selected_row = (num*5) + random.randint(0,4)
    selected_elem = random.randint(0,99)
    cv2.imwrite("dig.jpg",digit_cells[selected_row][selected_elem])
    return digit_cells[selected_row][selected_elem]

# for i in range(10):
#     print(predict_single_char_model(find_random_digit_pic(3)))

using_easyocr("data/hand.png")