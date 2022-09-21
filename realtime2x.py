import numpy as np
import cv2 
import tkinter as tk
from tkinter import simpledialog
from datetime import datetime

wht=320
conflevel=0.5
classesfile = 'coco.names'
nmsthreshold=0.3
classNames =[]
with open(classesfile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
print(classNames)  
print(len(classNames))

ROOT = tk.Tk()

ROOT.withdraw()
# the input dialog
choice = simpledialog.askstring(title="Test",
                                  prompt="What' do you wnat to search in ?:")

# check it out
search=simpledialog.askstring(title="Test",
                              prompt="What' do you wnat to search?:")



cnt=0
totalcnt=0

modelconfigration ='yolov3.cfg'
modelweight='yolov3.weights'
net=cv2.dnn.readNetFromDarknet(modelconfigration,modelweight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

f = open('history1.txt', 'r+')
f.truncate(0)

def findObjects(outputs,img):
    global cnt,totalcnt
    Ht,Wt,Ct=img.shape
    bbox=[]

    classIds=[]
    confs=[]

    for output in outputs:
        for det in output:
            scores=det[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]
            if confidence > conflevel:                         
                w,h=int(det[2]*Wt), int(det[3]*Ht)
                x,y=int((det[0]*Wt)-w/2),int((det[1]*Ht)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
                
    indicies = cv2.dnn.NMSBoxes(bbox,confs,conflevel,nmsthreshold)
    for i in indicies:
        i=i[0]
        box=bbox[i]
        x,y,w,h=box[0],box[1],box[2],box[3]
        if search.lower()==classNames[classIds[i]]:
            print("detected")
            cnt+=1
            totalcnt+=1
            print(cnt)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.putText(img,f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',
            (x,y-8),cv2.FONT_HERSHEY_COMPLEX,0.64,(255,0,255),1)
        else:
            print("Not detected")
            totalcnt+=1
           
    file = open("history1.txt", "a")
    file.write("\n")
    file.write(str(datetime.now()))
    file.write("\n")
    file.write("Total object detected= " + str(totalcnt) + "\n" +"Tareted object detected = "+ str(cnt) + "\n"+ "******************************" )        
    file.write("\n")
    file.write("\n")      

    totalcnt=0
    cnt=0            

"""real time"""
if choice.lower()=='real':
    cap=cv2.VideoCapture(0)
    while True:
     success, img =cap.read()    

     blob=cv2.dnn.blobFromImage(img,1/255,(wht,wht),[0,0,0],1,crop=False)
     net.setInput(blob)    
     layerNames=net.getLayerNames()    
     outputNames= [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
        
     outputs=net.forward(outputNames)
        
     findObjects(outputs,img)   

     cv2.imshow('imageleft',img)
     cv2.waitKey(1)
    

if choice=='input':
    cap = cv2.imread('bus1.jpg')
    imgin=cap
    blob=cv2.dnn.blobFromImage(imgin,1/255,(wht,wht),[0,0,0],1,crop=False)
    net.setInput(blob)    
    layerNames=net.getLayerNames()    
    outputNames= [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]       
    outputs=net.forward(outputNames)
        
    findObjects(outputs,imgin)   

    cv2.imshow('final',imgin)
    cv2.waitKey(0)

 