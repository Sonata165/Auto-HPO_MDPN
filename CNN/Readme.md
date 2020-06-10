### Folders:

​	data: You need to put your datasets here. For details, see data/readme.
​	result: The result of experiment will be saved here.
    model: The models from Mdpn.py or RandomMdpnComp.py

### The whole process is:

1. data preprocessing: DataSplit.py  DataEncode.py  FindLabel.py

2. Train the MDPN: Mdpn.py

3. Do MDPN experiment: CnnExp.py

4. Control group: RndomMdpnComp.py

### Other Modules:


​	Utils.py and ReadDataSet.py: Some utility functions
    AnalysisCNN.py: Contain some visual methods after experiment.
    ZooptUtils.py: Search the best hyper-paramers for the given dataset Using ZOOpt 
    