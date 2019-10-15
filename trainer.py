#Creating Trainer for dataset

import os
import cv2
import numpy as np



recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'

def get_images_and_ID(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    face_to_train = []
    id = []

    for imagePath in imagePaths:

        #try again with imread
        faceImg = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        faceNp = np.asarray(faceImg, dtype=np.uint8)
        id.append(int(((imagePath.split("\\")[1]).split(' ')[1])))
        face_to_train.append(faceNp)


        cv2.imshow('Training', faceNp)
        cv2.waitKey(10)

    return face_to_train, np.array(id)

'''
        print(imagePath)
    print('*'*40)
    print(id)
'''



faces_to_train, id = get_images_and_ID(path)
recognizer.train(faces_to_train, id)

recognizer.save('recognizer/TrainingData.yml')
cv2.destroyAllWindows()

print('Model Trained Successfully')




#END
