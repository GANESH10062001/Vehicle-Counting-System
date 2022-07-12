
import cv2
from cv2 import blur
import numpy as np

#webcam
cap=cv2.VideoCapture('highway.webm')

min_width_r=80
min_hight_r=80

count_line_position=500

#initialize substructor
algo=cv2.bgsegm.createBackgroundSubtractorMOG()


def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect = []
offeset=6#allowable error betweenn pixel
counter=0
    
while True:
    ret,frame1=cap.read()
    grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub= algo.apply(blur)
    dilate=cv2.dilate(img_sub,np.ones((5,5)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilate1=cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    dilate1=cv2.morphologyEx(dilate1,cv2.MORPH_CLOSE,kernel)
    counterShape,h =cv2.findContours(dilate1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(20,count_line_position),(1550,count_line_position),(225,127,0),3)
    

    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter=(w>=min_width_r) and (h>=min_hight_r)
        if not validate_counter:
            continue
        
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"VEHICLE"+str(counter),(x,y-20),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,244,0),2)



        center=center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)

         
          
        for(x,y) in detect:
            if y<(count_line_position+offeset) and  y>(count_line_position-offeset):
                counter+=1
        cv2.line(frame1,(20,count_line_position),(1550,count_line_position),(0,127,255),3)
        detect.remove((x,y)) 
        print("Vehicle Counter:"+str(counter))



    cv2.putText(frame1,"VEHICLE COUNTER :"+str(counter),(450,70),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,2,(0,0,225),5)





    #cv2.imshow('Detecter',dilate1)
    cv2.imshow('video original',frame1)
    
    if cv2.waitKey(1)==13:
        break

cv2.destroyAllWindows()
cap.release()

