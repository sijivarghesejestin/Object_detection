import cv2
import matplotlib.pyplot as plt                                                   #Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

#importing and using necessary files

config_file = '/content/drive/MyDrive/Object detection/ssd_mobilenet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = '/content/drive/MyDrive/Object detection/ssd_mobilenet/frozen_inference_graph (2).pb'

#Tenserflow object detection model
model = cv2.dnn_DetectionModel(frozen_model, config_file) 

#Reading Coco dataset
classLabels = []
file_name = '/content/drive/MyDrive/Object detection/ssd_mobilenet/labels.txt'  
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)


#Model training
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5)) 
model.setInputSwapRB(True)

#reading image
img = cv2.imread('/content/drive/MyDrive/Object detection/image1.jpg')
plt.imshow(img)

#object detection :class index (object), confidence (accuracy level) and bbox (location co-ordinates).
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

print(ClassIndex)


#plotting boxes
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
  cv2.rectangle(img, boxes, (255,0,0), 2)
  cv2.putText(img, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0,255,0), thickness=3)

#converting image from BGR to RGB
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))