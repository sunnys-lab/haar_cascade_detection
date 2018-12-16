import cv2

face_Cascade = cv2.CascadeClassifier("haarcascades\haarcascade_frontalface_alt.xml")
eye_Cascade = cv2.CascadeClassifier("haarcascades\haarcascade_eye.xml")

eye_detect = True
info = ''
font = cv2.FONT_HERSHEY_SIMPLEX

# Capture frame-by-frame
try:
    cap = cv2.VideoCapture("files/sample.avi")
except:
    print('video loading failed')


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if eye_detect:
        info = 'Eye Detection On'
    else:
        info = 'Eye Detection Off'

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_Cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60))

    cv2.putText(frame, info, (5,15), font, 0.5, (255, 0, 255), 1)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Detected Face', (x-5, y-5), font, 0.5, (255, 255, 0), 1)
        if eye_detect:
            h2 = int(round(h*3/5))
            rol_gray  = gray[y:y+h2, x:x+w]
            rol_color = frame[y:y+h2, x:x+w]
            eyes = eye_Cascade.detectMultiScale(rol_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(rol_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
                #cv2.putText(rol_color, 'Detected Eye', (ex - 5, ey - 5), font, 0.5, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.namedWindow('Face', cv2.WINDOW_NORMAL)
    cv2.imshow('Face', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
