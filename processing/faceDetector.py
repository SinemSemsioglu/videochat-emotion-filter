from cvzone.FaceDetectionModule import FaceDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import cv2
import processing.emotionRec
from math import floor

processing.emotionRec.init()

cap = cv2.VideoCapture(0)
detector = FaceDetector()
segmentor = SelfiSegmentation()

circleSize = 200
circleRed = 100


def cutSquare(img):
    (h, w, c) = img.shape
    s = min(h,w)
    return img[floor((h-s)/2):floor((h+s)/2), floor((w-s)/2):floor((w+s)/2), :]

def drawCircle(img, center, circleSize, circleRed):
    overlay = img.copy()

    # A filled circle
    cv2.circle(overlay, center, circleSize, (100, 100, circleRed), cv2.FILLED)
    
    alpha = 0.6  # Transparency factor.
    
    # Following line overlays transparent rectangle
    # over the image
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def processCameraFeed():
    c = 0
    while True:
        c += 1
        success, img = cap.read()

        imgOut = processFrame(img, c/30) # todo 30 stands for fps get the real fps

        cv2.imshow("Image", imgOut)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

threshold = 0.1 # a small number in this case roughly corresponds to the time_base value of the stream

def processFrame(img, counter):
    global circleRed
    global circleSize

    imgBackground = img.copy()
    imgForeground = img.copy()
    
    img, bboxs = detector.findFaces(img)

    imgEmotion = cv2.resize(cutSquare(imgBackground), (256, 256))
     
    #print(f'counter {counter} and {counter % 5}')
    if counter % 3 < threshold:
    #    print('inside')
        emotionScores = processing.emotionRec.score_frame(cv2.cvtColor(imgEmotion, cv2.COLOR_BGR2RGB))
        circleSize = floor(150 + 100 * emotionScores['dimensional'][1]) # assuming arousal value is in [0,1]
        circleRed = 125 + 125 * emotionScores['dimensional'][0] # assuming valence is in [-1,1]

    if bboxs:
        # bboxInfo - "id","bbox","score","center"
        center = bboxs[0]["center"]
        imgBackground = drawCircle(imgBackground, center, circleSize, circleRed)

    return segmentor.removeBG(imgForeground, imgBackground, threshold=0.5)
