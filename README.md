## introduction and flow
  Automatic machine learning (AutoML) has gained wide attention and applications in both industry and academia. Automatic hyper-parameter optimization(auto-HPO) is one of the most critical parts. The effectiveness of many machine learning algorithms is extremely sensitive to hyper-parameters. Without a good set of hyper-parameters, the task cannot be solved well even with the optimal algorithm.

Our contributions are summarized as follows.
- We consider the mapping from data to the optimal hyper-parameters and apply this mapping to the selection of the optimal hyper-parameters. On different tasks of an algorithm, the model has strong transferability, which greatly saves time overhead. For this reason, the model can achieve ultra-high-dimensional optimization of hyper-parameters.

- With XGBoost as an example, we design the neural network structure for the mapping as well as training approaches, which could be applied to other machine learning tasks with slight modification.

- Experimental results on real data demonstrate that the proposed approach significantly outperforms the state-of-art algorithms in both accuracy and efficiency.

### flow
  The components of our algorithm are shown in this figure. Part1 is to train MDPN with processed data come from part2 and part3. In part4, MDPN is used to predict parameters, which are optimized further in part5.
![image](https://github.com/Sonata165/NIPSProject/blob/master/ImgForReadme/system.png)

## operation method

## experiment

### data set

### baseline

In this part, we introduce the baseline we choose briefly and basic settings about the experiments.

[**Bayesian optimization**](https://en.wikipedia.org/wiki/Bayesian_optimization) is shorted in BO, which is a sequential design strategy for global optimization of black-box functions that doesn't require derivatives. BO is widely used in hyper-parameter optimization. Here we use BO to optimize XGBoost as control group.

[**Zeroth-order optimization**](https://arxiv.org/abs/1911.06317) is the process of minimizing an objective $f(x)$, given oracle access to evaluations at adaptive chosen input $x$, which is shorted as ZOOpt. Here we use ZOOpt as control group of evaluation of our model on CNN.

**Metrics** we recorded in our experiments contains time overhead and accuracy of each test data set. We evaluate our model on hundreds of data sets, so some global statistics are necessary to take analyses. We extract median, maximum, and two quartiles of all test sets' accuracy.
### environment


### result
  We design three groups experiments containing an experiment group, a blank control group(BCG) and a control group. In the experiment group, we use MDPN and MDPN + LOPT to take optimization. In the control group, BO and ZOOpt are used. In the blank control group, we optimize hyper-parameters with a MDPN or MDPN + LOPT model without pre-training.

As for partitioning of data sets. We partition raw data sets ![](http://latex.codecogs.com/gif.latex?\mathcal{D}) into ![](http://latex.codecogs.com/gif.latex?\mathcal{X}) and ![](http://latex.codecogs.com/gif.latex?\mathbb{X}) with size 9:1, and ![](http://latex.codecogs.com/gif.latex?\mathcal{X}) is used to train MDPN before experiments. Then we continue to partition each data set in ![](http://latex.codecogs.com/gif.latex?\mathbb{X}) by 9:1 into ![](http://latex.codecogs.com/gif.latex?X_{train}) and ![](http://latex.codecogs.com/gif.latex?X_{test}). We firstly use ![](http://latex.codecogs.com/gif.latex?X_{train}) to train XGBoost or CNN, then we use ![](http://latex.codecogs.com/gif.latex?X_{test}) to test the performances of XGBoost or CNN with the hyper-parameters MDPN and baseline give out.

![](https://github.com/Sonata165/NIPSProject/blob/master/ImgForReadme/cnn_accu.png)![](https://github.com/Sonata165/NIPSProject/blob/master/ImgForReadme/xg_accu.PNG)
![](https://github.com/Sonata165/NIPSProject/blob/master/ImgForReadme/cnn_time.jpg)![](https://github.com/Sonata165/NIPSProject/blob/master/ImgForReadme/xg_time.jpg)
 <center class="half">
    <img src="https://github.com/Sonata165/NIPSProject/blob/master/ImgForReadme/cnn_accu.png" width="200"/><img src="https://github.com/Sonata165/NIPSProject/blob/master/ImgForReadme/cnn_accu.png" width="200"/><img src="https://github.com/Sonata165/NIPSProject/blob/master/ImgForReadme/cnn_accu.png" width="200"/>
</center>
