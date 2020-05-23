import os
import numpy as np
import pickle as pk

def main():
    FEATURE_PATH = '../12.27_dataset/feature/'
    LABEL_PATH = '../12.27_dataset/result/'
    feature_files = os.listdir(FEATURE_PATH)
    label_files = os.listdir(LABEL_PATH)
    for feature_file in feature_files:
        file_name = FEATURE_PATH + feature_file
        f = open(file_name, 'rb')
        feature = pk.load(f)
        print(feature.shape)

    for label_file in os.listdir(LABEL_PATH):
        file_name = LABEL_PATH + label_file
        f = open(file_name, 'rb')
        param = pk.load(f)
        result = pk.load(f)


if __name__ == '__main__':
    main()