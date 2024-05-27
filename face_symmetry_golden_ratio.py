import cv2
import numpy as np
import dlib

# Yüz ve landmark dedektörlerini yükle
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

# IP kamera veya video kaynağının URL'si
url = "http://192.168.137.36:8080" 
vs = cv2.VideoCapture(url + "/video")

print("Altın Oran Hesaplama")
print("Yaklaşık Değer")

golden_ratio = 1.618

def score_ratio(ratio, golden):
    """Oranı altın oranla karşılaştır ve puanla"""
    return (ratio / golden) * 10 if ratio <= golden else (golden / ratio) * 10

while True:
    ret, frame = vs.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Oranları hesapla
        agbg = (landmarks.part(54).x - landmarks.part(48).x) / (landmarks.part(35).x - landmarks.part(31).x)
        göz_oranı = (((landmarks.part(39).x) - (landmarks.part(36).x)) / 2) - (((landmarks.part(45).x) - (landmarks.part(42).x)) / 2)
        ggkg = göz_oranı / (landmarks.part(26).x - landmarks.part(17).x)
        yuyg = (landmarks.part(71).x - landmarks.part(8).x) / (landmarks.part(15).x - landmarks.part(1).x)
        dudak = (landmarks.part(66).x - landmarks.part(57).x) / (landmarks.part(62).x - landmarks.part(51).x)

        # Puanlamalar
        agbg_score = score_ratio(agbg, golden_ratio)
        ggkg_score = score_ratio(ggkg, golden_ratio)
        yuyg_score = score_ratio(yuyg, golden_ratio)
        dudak_score = score_ratio(dudak, golden_ratio)

        # Genel altın oran puanı
        golden_score = (agbg_score * ggkg_score * yuyg_score * dudak_score) / 10000
        print("Altın Oran Puanınız", golden_score)

        # Landmark noktalarını çiz
        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y  
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
    
    # Çerçeveyi yeniden boyutlandır ve göster
    cam = cv2.resize(frame, (640, 480))
    cv2.imshow('Frame', cam)
    
    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()
