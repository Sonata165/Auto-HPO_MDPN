import os
import pickle as pk
FEATURE_PATH = '../12.27_dataset/feature/'
featureFiles = os.listdir(FEATURE_PATH)
for featureFile in featureFiles:
    print("read file: "+featureFile+'......')
    with open(FEATURE_PATH+featureFile,'rb') as f:
        feature = pk.load(f)
    print('OK')
