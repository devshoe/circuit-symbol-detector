import numpy as np, math
import cv2
import matplotlib.pyplot as plt
import keras
import skimage.morphology
import argparse
import matplotlib.pyplot as plt
import random
from os.path import expanduser
home = expanduser("~")


MODEL_PATH = home+"/Desktop/circuit-symbol-detector/colab_images/model3.h5"
model = keras.models.load_model(MODEL_PATH)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model_labels = {0:"diode", 1:"resistor", 2:"inductor", 3:"ground", 4:"voltage", 5:"cap"}
 
class Circuit():
  def __init__(self, image_path, threshold=50, min_comp_size_pct=0.000):
 
    self.start_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) #don't touch this.
 
    if self.start_image.shape[-1] == 4: #if it has alpha channel
      trans_mask = self.start_image[:,:,3] == 0 #mask wherever alpha is 0
      self.start_image[trans_mask] = [255, 255, 255, 255]
      self.start_image = cv2.cvtColor(self.start_image, cv2.COLOR_BGRA2BGR)
 
    self.width = self.start_image.shape[0] 
    self.height = self.start_image.shape[1]
    self.area = self.height * self.width

    self.grayscaled = cv2.cvtColor(self.start_image.copy(), cv2.COLOR_BGR2GRAY)
    self.grayscaled = cv2.resize(self.grayscaled, (self.height,self.width))
    _, self.thresholded = cv2.threshold(self.grayscaled, threshold, 255, cv2.THRESH_BINARY)
    self.skeletonized = (skimage.morphology.skeletonize(self.thresholded == 0) * 255).astype('uint8')
    self.pipelined = self.skeletonized.copy()
 
    self.lines = []
    self.components = []
    self.extracted_components = []
    self.blank = self.pipelined * 0

    self.remove_lines()
    self.find_components(reject_below_factor=min_comp_size_pct).find_components(True,False,0.005, True)
    for img in self.components: self.extracted_components.append(img["img"])
    self.predict()

  def get_components(self): return self.extracted_components

  def remove_lines(self, thickness=10):
    self.canny = cv2.Canny(self.pipelined, 100,200)
    self.hline_thresh = 75
    self.hline_min_line_length = min(self.height, self.width)*0.1
    self.hline_max_line_gap = min(self.height, self.width)*0.1
    self.all_lines = cv2.HoughLinesP(self.canny, 1, np.pi/180, 
                           self.hline_thresh, None, 
                           self.hline_min_line_length, self.hline_max_line_gap) #TODO
    for i,line in enumerate(self.all_lines):
      for x1,y1,x2,y2 in line:
        length= math.sqrt((x1-x2)**2 + (y1-y2)**2)
        self.lines.append({"label":f"line_{i}", "coordinates":(x1,y1,x2,y2), "length":length})
        cv2.line(self.pipelined,(x1,y1),(x2,y2),0,thickness)
    return self
 
  def find_components(self, draw_on_blank=False, draw_on_pipeline=True, reject_below_factor=0.005, store=False):
    """Find contours. Blob it up."""
    self.blank = self.pipelined * 0
    acceptable_area = (0, self.area * 0.8)
    thickness = 25
    all_contours, h1 = cv2.findContours(self.pipelined,cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
    self.contours = [cnt for cnt in all_contours if acceptable_area[0] <= cv2.contourArea(cnt) < acceptable_area[1]]
    for cnt in self.contours:
      x,y,w,h = cv2.boundingRect(cnt)
      if w*h > self.area * reject_below_factor:
        if store: self.components.append({"loc":(x,y),"dim":(w,h), "img":self.start_image[y:int(y+h*1.1), x:int(x+w*1.1)]})
        if draw_on_blank:cv2.rectangle(self.blank,(x,y),(x+w,y+h),(123,0,12),10)
        if draw_on_pipeline:cv2.drawContours(self.pipelined, cnt, -1, 255, thickness)
    return self

  def predict(self):
    for component in self.components:
      img = cv2.resize(component["img"],(100,100), 3)
      img = np.reshape(img, (1,100,100,3))
      classes = list(model.predict(img)[0])
      component["label"]=model_labels[classes.index(max(classes))]

  def show_with_labels(self):
    for component in self.components:
      cv2.imshow(component["label"],component["img"])

  def write_out(self):
    print("hello")
    for component in self.components:
      name = "dump/"+component["label"]+ str(random.randint(0,1000))+".jpg"
      cv2.imwrite(name,component["img"])
      print(name)
# ckt0 = Circuit("C:/Users/devshoe/Google Drive/colab_images/res.png")
# ckt1 = Circuit("C:/Users/devshoe/Google Drive/colab_images/hand.png", 125)
# ckt2 = Circuit("C:/Users/devshoe/Google Drive/colab_images/nolabel.png")
# ckt3 = Circuit("C:/Users/devshoe/Google Drive/colab_images/new.jpg", 125, 0)
# test_list = [ckt0, ckt1, ckt2, ckt3]

# ckt3.show_with_labels()

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument("-p", "-path", required=True)
  args = argparser.parse_args()
  Circuit(args.p).write_out()