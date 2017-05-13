import cv2
import glob
import random

##############    applying random rotations to images to augment data
count = 0
for filename in glob.glob("./benign/*.jpg"):

    img = cv2.imread(filename)
    img = cv2.resize(img, (256, 256))
    rows = img.shape[0]
    cols = img.shape[1]
    for i in range(1):
        angle = random.randint(1, 360)
        rotmat = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        newimg = cv2.warpAffine(img, rotmat, (cols, rows))
        cv2.imwrite('./temp_b/'+ str(count)+ '_' + str(i) + '_rand.jpg', newimg)
        count+=1
    
for filename in glob.glob("./malignant/*.jpg"):

    img = cv2.imread(filename)
    img = cv2.resize(img, (256, 256))
    rows = img.shape[0]
    cols = img.shape[1]
    for i in range(1):
        angle = random.randint(1, 360)
        rotmat = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        newimg = cv2.warpAffine(img, rotmat, (cols, rows))
        cv2.imwrite('./temp_m/'+ str(count)+ '_' + str(i) + '_rand.jpg', newimg)
        count+=1
print ('Done')
