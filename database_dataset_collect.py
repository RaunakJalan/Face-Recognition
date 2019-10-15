
import cv2
import sqlite3



conn = sqlite3.connect("FaceBase.db")
c = conn.cursor()


count = 1
id = 1


def enter_data(id):

    name = input("ENTER USER NAME: ")
    gender = input("ENTER USER GENDER: ")
    age = input("ENTER USER AGE: ")
    recognized_times = 0

    c.execute("INSERT INTO People (ID, Name, Age, Gender, Recognized Times) VALUES (?, ?, ?, ?, ?)",
              (id, name, gender, age, recognized_times))

    conn.commit()


def id_generate(id):

    recordforid = 0
    id = input("ENTER USER ID: ")
    try:
        dataid = c.execute('SELECT * FROM People WHERE ID='+str(id))
        for row in dataid:
            recordforid = 1

        if recordforid == 1:

            print("ID already Present and in use....")
            print("Enter different id again: ")
            id_generate(id)

        else:
            enter_data(id)

    except:
        pass

    return id

def name_generate(id):
    c.execute('SELECT Name FROM People WHERE ID='+str(id))
    data = c.fetchone()
    print(data)


id = id_generate(id)
name = name_generate(id)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

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
    if faces is not () and count<100:

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cropped_face = gray[y:y+h, x:x+w]

        cropped_faces = cv2.resize(cropped_face,(200,200))
        cropped_faces = cropped_faces
        file_name_path = 'dataset/'+ name + ' ' +str(id) + ' ' +str(count) + '.jpg'
        cv2.imwrite(file_name_path, cropped_faces)
        count+=1

            # Put count on images and display live count
        cv2.putText(cropped_faces, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', cropped_faces)

    #cv2.imshow('FACE DETECT', frame)

    if count == 100:

        print("DATA SAMPLE COLLECTION SUCCESSFUL...")

        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
