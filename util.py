# -------------- #
# util.py        #
# Faerlin Pulido #
# 2020           #
# -------------- #
#
# This is a utility function used in jumio_challenge.ipynb
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_images(instances, image_dim, images_per_row=10, **options):
    """Plot images

    Args:
        instances (list of numpy arrays): Each array is an image of size image_size-by-image_size-by-m
                                          where m is the number of channels.
        images_per_row (int, optional): Number of images per row. Defaults to 10.

    Reference: code is adopted from 
        Geron A: 2017, "Hands-On Machine Learning with Scikit-Learn and Tensorflow". 
    """
   
    N = len(instances)
    instances = np.reshape(instances, [N, image_dim, image_dim])

    images_per_row = min(len(instances), images_per_row)
    images = [instance for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((image_dim, image_dim * n_empty)))

    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)
    plt.figure(figsize=(12,20))
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")
    
def plot_accuracy_loss(history):
    num_epochs = len(history['categorical_accuracy'])
    xs = [int(i) for i in range(1, num_epochs+1)]
    
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(xs, history['categorical_accuracy'], color='red', linestyle='solid')
    plt.plot(xs, history['val_categorical_accuracy'], color='blue', linestyle='solid')
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.title('Accuracy', fontsize=18)
    plt.grid(linestyle='--')
    plt.legend(['training', 'validation'], loc='lower right');
    
    plt.subplot(1, 2, 2)
    plt.plot(xs, history['loss'], color='red', linestyle='solid')
    plt.plot(xs, history['val_loss'], color='blue', linestyle='solid')
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('loss', fontsize=16)
    plt.title('Loss', fontsize=18)
    plt.grid(linestyle='--')
    plt.legend(['training', 'validation'], loc='upper right');