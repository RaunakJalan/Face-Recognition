#In LBPHFaceRecognizer_create in Python 3.3 and above load is replaced with read and save is replaced with write


import cv2
import os


def name_of_user(path, id):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]

    for imagePath in imagePaths:

        id_path = int(((imagePath.split("\\")[1]).split(' ')[1]))

        if id_path == id:

            nou = (imagePath.split("\\")[1]).split(' ')[0]
            break

    return nou

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\TrainingData.yml")

path = 'dataset'
name = ''

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    try:


            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cropped_face = gray[y:y+h, x:x+w]
                result = rec.predict(cropped_face)

                name = name_of_user(path, result[0])

                if result[1] < 500:
                    confidence = int( 100 * (1 - (result[1])/400) )

                    if confidence > 75:
                        display_string = str(result[0])+ ". " + str(confidence) + '% Confident it is '+ name
                    else:
                        display_string = ""

                cv2.putText(frame, display_string, (100, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)


            cv2.imshow("FACE RECOGNIZER", frame)
            # Put count on images and display live count

    except:

        cv2.imshow("FACE RECOGNIZER", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
