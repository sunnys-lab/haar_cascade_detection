import os
import cv2

face_Cascade = cv2.CascadeClassifier("haarcascades\haarcascade_frontalface_alt.xml")
eye_Cascade = cv2.CascadeClassifier("haarcascades\haarcascade_eye.xml")
cat_Cascade = cv2.CascadeClassifier("haarcascades\haarcascade_frontalcatface.xml")
profile_Cascade = cv2.CascadeClassifier("haarcascades\haarcascade_profileface.xml")
banana_Cascade = cv2.CascadeClassifier("haarcascades\_banana_classifier.xml")

eye_detect = False
profile_detect = False

font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread(os.getcwd() + '/files/cats.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_Cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
profile = profile_Cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
banana = banana_Cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=7, minSize=(80, 80))
cats = cat_Cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

for (x, y, w, h) in cats:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    if eye_detect:
        eyes = eye_Cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

if profile_detect:
    for (x, y, w, h) in profile:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

for (x, y, w, h) in banana:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
