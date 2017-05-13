import cv2
import glob

#################   applying mask to original data to filter images

for filename in glob.glob('../ISIC_cleaned_data/*.jpg'):
    img = cv2.imread(filename)
    img = cv2.resize(img, (256, 256))
    mask = cv2.imread('../ISIC_Contours_CLAHE/' + filename[21:], 0)
    blurred = cv2.GaussianBlur(img, (5, 5), 2)
    
    for i in range(256):
        for j in range(256):
            if (mask[i, j]!=0):
                blurred[i, j, :] = img[i, j, :]

    cv2.imwrite('../filtered_data/'+ filename[21:], blurred)

print ('Done')
