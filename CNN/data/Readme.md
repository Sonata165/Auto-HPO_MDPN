## Data

All data sets for CNN or MDPN need to be put here. Data sets are divided into two groups. 
Data sets in 'feature' and 'result' are for training MDPN and those in 'experiment' are for testing MDPN

### svhn

The original SVHN data set.

### subset

- If you want to train an MDPN for CNN , you need to run DataSplit.py first and all the results are in it.
  - type: mat
  - dict
  - have two keys: 'X' and 'y' 
  
### feature

- If you want to train an MDPN for CNN , you need to run DataEncode.py then and all the results are in it.
  - type: pkl
  - numeric
  
### result

- If you want to train an MDPN for CNN , you need to run FindLabel.py and get the hyperparameters for cnn. All the results are in it.
  - type: pkl
  - numeric
  
### experiment_data

- After run DataSplit.py, the datasets for testing the MDPN are reserved here. 
  - type: mat
  - dict
  - have two keys: 'X' and 'y' 
- train_data: the datasets for training the CNN models
- test_data: the datasets for testing the CNN models

