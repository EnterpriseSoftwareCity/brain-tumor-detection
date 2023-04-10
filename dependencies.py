import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os


STANDARD_DATA_DIRECTORY_STRUCTURE = 'data/Testing/'
STANDARD_RESIZE_VALUE = 200


def collect_images_from_files(X, Y):
    classifier_classes = {'no_tumor': 0, 'glioma_tumor': 1, 'pituitary_tumor': 1, 'meningioma_tumor': 1}
    
    for cls_class in classifier_classes:
        path = STANDARD_DATA_DIRECTORY_STRUCTURE + cls_class
        for file in os.listdir(path):
            img = cv2.imread(path + '/' + file, 0)
            img = cv2.resize(img, (STANDARD_RESIZE_VALUE, STANDARD_RESIZE_VALUE))
            X.append(img)
            Y.append(classifier_classes[cls_class])
            
            
def test_based_on_images(fileTestingClass, dir_source, size = 9):
    values_classifier = {0: 'No Tumor', 1: 'Is Tumor'}
    
    plt.figure(figsize=(12, 8))
    path = os.listdir(dir_source)
    
    subplot_index = 1

    for i in os.listdir(dir_source + fileTestingClass)[:size]:
        plt.subplot(3, 3, subplot_index)

        img = cv2.imread(dir_source + fileTestingClass + i, 0)
        img_resized = cv2.resize(img, (STANDARD_RESIZE_VALUE, STANDARD_RESIZE_VALUE))
        img_resized = img_resized.reshape(1, -1) / 255
        path = classifier.predict(img_resized)
        plt.title(values_classifier[path[0]])
        plt.imshow(img, cmap = 'gray')
        plt.axis('off')
        
        if subplot_index == 9:
            subplot_index = 1
        else:
            subplot_index += 1
            