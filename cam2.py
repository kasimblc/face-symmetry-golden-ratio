import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

url = "http://192.168.137.36:8080" 
vs = cv2.VideoCapture(url+"/video")

print("Altın Oran Hesaplama ") 
print("Yaklaşık Değer") 
while True:
    _, frame = vs.read()
    if not _:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)
        golden=float(1.618)
        totalgolden=4*golden
         #n48=landmarks.part(48).x   
         #n54=landmarks.part(54).x
         #n31=landmarks.part(31).x
         #☼n35=landmarks.part(35).x
        agbg=((landmarks.part(54).x)-(landmarks.part(48).x))/((landmarks.part(35).x)-(landmarks.part(31).x))
        goz=(((landmarks.part(39).x)-(landmarks.part(36).x))/2)-(((landmarks.part(45).x)-(landmarks.part(42).x))/2)
        ggkg=goz/((landmarks.part(26).x)-(landmarks.part(17).x))
        yuyg=((landmarks.part(71).x)-(landmarks.part(8).x))/((landmarks.part(15).x)-(landmarks.part(1).x))
        dudak=((landmarks.part(66).x)-(landmarks.part(57).x))/((landmarks.part(62).x)-(landmarks.part(51).x))
        
        if agbg <= golden:
            agbg=(agbg/golden)*10
        else: 
            agbg=(golden/agbg)*10
            
        if ggkg <= golden:
            ggkg=(ggkg/golden)*10
        else: 
            ggkg=(golden/ggkg)*10
            
        if yuyg <= golden:
           yuyg=(yuyg/golden)*10
        else: 
           yuyg=(golden/yuyg)*10
            
        if dudak <= golden:
           dudak=(dudak/golden)*10
        else: 
           dudak=(golden/dudak)*10
        g=(agbg*ggkg*yuyg*dudak)/10000    
        print("Altın Oran Puanınız",g)
        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y  
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            
    #putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
    cam=cv2.resize(frame,(640, 480))
    cv2.imshow('Frame', cam)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        vs.release()
        cv2.destroyAllWindows()
        break
