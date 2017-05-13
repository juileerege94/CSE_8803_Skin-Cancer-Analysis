READ ME
Aim of the project is to differentiate between benign and malignant images in the dataset and teach the network to classify correctly. 

We have 7 code files which perform different functionalities.

1. create_data_file.py separates the ISIC images into malignant and 
   benign and stores them into different folders.

2. image_filter.py filters the images according to the mask.

3. inception_feature_extraction.py extracts the features from the second last
   layer of the Inception V3 model using Tensorflow packages.

4. preprocess.py performs various pre-processing operations on the images
   such as CLAHE, median blur, Otzu's binarization and gaussian blur.

5. random_rotate.py applies random rotations on the images for augmentation.

6. statistics.py performs all the classification tasks and calculation of 
   various statistics from the results.

7. TSNE.ipynb performs the t-SNE visualization on the data in an ipython
   notebook.

Rest of the augmentations were done through command line using the package
as cited in the paper.

Data sources include: 
http://www.cs.rug.nl/~imaging/databases/melanoma_naevi/
https://challenge.kitware.com/#challenge/n/ISIC_2017%3A_Skin_Lesion_Analysis_Towards_Melanoma_Detection
