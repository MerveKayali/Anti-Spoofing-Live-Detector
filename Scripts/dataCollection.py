import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from time import time

######################################
offsetsPercentageW=10
offsetsPercentageH=20
camWidth,camHeight=640,480
floatingPoint=6

confidence=0.8
outputFolderPath='Dataset/DataCollect'
classId=1 #0 fake 1 gerçek
save=True
blurTreshold=35 #Değer büyüdükçe daha çok odaklı demek
debug=False
#####################################

cap = cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4,camHeight)
detector = FaceDetector()
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgOut=img.copy()
    img, bboxs = detector.findFaces(img,draw=False)

    listBlur=[] #yüzün blurlu mu değil mi oolduğunu gösteren liste
    listInfo=[] #Normalize değerler ve etiket txt dosyası için sınıf adı

    if bboxs:
        # bboxInfo - "id","bbox","score","center"

        for bbox in bboxs:
            x,y,w,h=bbox["bbox"]
            score=bbox["score"][0]
            #print(x,y,w,h)

            #------- Skor Kontrolü---------------
            if score>confidence:
                # burada bbox istediğimiz boyutta değil onu büyütmek için bazı ayarlamalar yapıyoruz.
                offsetW = (offsetsPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)

                offsetH = (offsetsPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                # -------- 0'dan küçük değer geldiği zaman hata oluşmaması için--------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # -------- Bulanıklıkları Bulma --------------------
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValues = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                #Burada yüz blurlu değil ise o yüz için listeye True yazıyor blurlu ise False
                if blurValues>blurTreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                #---------Normalizasyon----------------
                ih,iw,_=img.shape
                xc,yc=x+w/2,y+h/2#center points

                xcn,ycn=round(xc/iw,floatingPoint),round(yc/ih,floatingPoint)
                wn,hn=round(w/iw,floatingPoint),round(h/ih,floatingPoint)
                #print(xcn, ycn,wn,hn)

                # -------- 1'den büyük değer geldiği zaman hata oluşmaması için--------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 0

                listInfo.append(f'{classId} {xcn} {ycn} {wn} {hn}\n')


                # ------- Box Çizim -------------------
                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score*100)}% Blur: {blurValues}', (x, y - 20),
                                   scale=2,thickness=3)

                #------------Debug---------
                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValues}', (x, y - 20),
                                       scale=2, thickness=3)

        #Burada tüm resimler net ise (blur değeri düşük ise) kaydetmek istiyoruz
        #çünkü temiz bir data oluşturmak istiyoruz.
        #örneğin iki yüz var ve biri blurlu bu resmi kaydetme diyoruz eğer ikiside net ise kaydediyoruz.
        if save:
            if all(listBlur) and listBlur!=[]:
                timeNow=time()
                timeNow=str(timeNow).split('.')
                timeNow=timeNow[0]+timeNow[1]
                print(timeNow)
                cv2.imwrite(f'{outputFolderPath}/{timeNow}.jpg',img)
                #-------------- Label Text Dosyası Kaydetme------------
                for info in listInfo:
                    f = open(f'{outputFolderPath}/{timeNow}.txt', 'a')
                    f.write(info)
                    f.close()


    cv2.imshow("Image", imgOut)
    cv2.waitKey(1)