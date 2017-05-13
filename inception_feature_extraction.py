import os
import re
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import pickle

model_dir = '/home/kchanda3/BDH/models/TUTORIAL_DIR/imagenet'
images_dir = '/home/kchanda3/BDH/models/TUTORIAL_DIR/Data_V2/'


def create_graph():
    with gfile.FastGFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        ph_def.ParseFromString(f.read())
	_= tf.import_graph_def(graph_def, name='')

def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    labels = []
    names = []
    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):
            if (ind%10 == 0):
                print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)
            names.append(image.split("/")[-1])
	    labels.append(image.split("/")[-1][0])

    return features, labels, names


list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]
features,labels, names = extract_features(list_images)
pickle.dump(features, open('features_v2', 'wb'))
pickle.dump(labels, open('labels_v2', 'wb'))
pickle.dump(names, open('names_v2', 'wb'))
