### Folders:

​	data: You need to put your datasets here. For details, see data/readme.
​	results: The result of experiment will be saved here.

### The whole process is:

1. data preprocessing: Preprocess.py

2. Train the MDPN: Mdpn.py

3. Do MDPN experiment: MdpnExp.py

4. Further optimize with LOPT: LoptExp.py

   (Optional)

5. Control group 1: RandomComp.py

6. Control group 2: BayesianComp.py

### Other Modules:

​	Cosntants.py: Constants
​	EncoderTrainer.py: Definition and training code of auto-encoder
​	Lopt.py: LOPT algorithm
​	Utils.py: Some utility functions

​	TrainedMdpn.h5: an pretrained MDPN for XGBoost