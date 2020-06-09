## Data

All data sets for XGBoost or MDPN need to be put here. Data sets are divided into two groups. Data sets in 'train' are for training MDPN and those in 'test' are for testing MDPN

### train

- raw: If you want to train an MDPN for XGBoost, you need to put a lot of data sets (for classification) here. Make sure they are all 
  - type: csv
  - numerical
  - have a column name: Label 
- after_cutting: The results of doing feature selecting or zero-padding will be saved here. The program will use the data sets here to compute labels and meta_features.
- labels
- meta_features

### test

- raw: If you want to test our auto-HPO system for XGBoost, put some data sets (for classification) here. Make sure they are all 
  - type: csv
  - numerical
  - have a column name: Label 
- after_cutting: The results of doing feature selecting or zero-padding will be saved here. Then the data sets here will be divided into training part and testing part, saved to the two folder below separately. 
- train_clf
- test_clf