import cv2
import matplotlib.pyplot as plt
config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"
model = cv2.dnn_DetectionModel(frozen_model,config_file)
classLabels = []
file_name = "Labels.txt"
with open(file_name,'rt') as ftp:
    classLabels =ftp.read().rstrip('\n').split('\n')
img = cv2.imread("carNperson.jpg")
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
plt.imshow(img)
ClassIndex, confidence , bbox = model.detect(img,confThreshold=0.5)
#print(ClassIndex)
font_scale = 4
font = cv2.FONT_HERSHEY_SIMPLEX
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten() , bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255,0,0),thickness=4)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))