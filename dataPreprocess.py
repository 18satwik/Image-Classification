# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:35:38 2021

@author: satwik
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

fashion_mnist = keras.datasets.fashion_mnist
#returns four numpy 28x28 arrays with pixel values ranging from 0 to #255
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',     
              'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle',    
               'boot']



train_images.shape

len(train_labels)

test_images.shape

len(test_labels)

train_images = train_images / 255.0
test_images = test_images / 255.0

datagen = ImageDataGenerator(zca_whitening=True)

datagen.fit(train_images)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

# scaler = StandardScaler()
# # Fit on training set only.
# scaler.fit(train_images)
# # Apply transform to both the training set and the test set.
# train_images = scaler.transform(train_images)
# test_images = scaler.transform(test_images)

pca = PCA(n_components=2)

pca = PCA(.95)
#pca.fit(train_images)

# train_images = pca.transform(train_images)
# test_images = pca.transform(test_images)

U, S, V = np.linalg.svd(train_images)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()





