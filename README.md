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
    \subsection{Experiment Settings}
        \label{sec:es}
		In this part, we introduce the baseline we choose briefly and basic settings about the experiments.
		
		\underline{Bayesian optimization}\footnote{\url{https://en.wikipedia.org/wiki/Bayesian_optimization}} is shorted in BO, which is a sequential design strategy for global optimization of black-box functions that doesn't require derivatives. BO is widely used in hyper-parameter optimization~\cite{li2019bayesian,snoek2012practical,wu2019hyperparameter}. Here we use BO to optimize XGBoost as control group.
		
		\underline{Zeroth-order optimization}~\cite{golovin2019gradientless} is the process of minimizing an objective $f(x)$, given oracle access to evaluations at adaptive chosen input $x$, which is shorted as ZOOpt. Here we use ZOOpt as control group of evaluation of our model on CNN.
        
        \underline{Metrics} we recorded in our experiments contains time overhead and accuracy of each test data set. We evaluate our model on hundreds of data sets, so some global statistics are necessary to take analyses. We extract median, maximum, and two quartiles of all test sets' accuracy.
        
        \underline{Basic settings} contain the software and hardware environment of our experiments, and these information is contained in the third section of supplementary material.


### result
    \subsection{Experiment Results}
    \label{sec:er}
	We design three groups experiments containing an experiment group, a blank control group(BCG) and a control group. In the experiment group, we use MDPN and MDPN + LOPT to take optimization. In the control group, BO and ZOOpt described in Section~\ref{sec:es} is used. In the blank control group, we optimize hyper-parameters with a MDPN or MDPN + LOPT model without pre-training.

    As for partitioning of data sets. We partition raw data sets $\mathcal{D}$ into $\mathcal{X}$ and $\mathbb{X}$ with size 9:1, and $\mathcal{X}$ is used to train MDPN before experiments. Then we continue to partition each data set in $\mathbb{X}$ by 9:1 into $X_{train}$ and $X_{test}$. We firstly use $X_{train}$ to train XGBoost or CNN, then we use $X_{test}$ to test the performances of XGBoost or CNN with the hyper-parameters MDPN and baseline give out.

    As talked in Section~\ref{sec:es}, we have recorded time overhead and accuracy. Next, we will compare and analyze the experimental results of the three groups from time overhead and accuracy.
    
    \subsubsection{Time overhead}
    We prepare 280 classification data sets as test sets when optimizing XGBoost. And when optimizing CNN, the number is 180 in which 90 of them are MNIST's subsets and 90 of them are SVHN's subsets. All the test set's time overhead has been recorded and can be seen in Figure~\ref{fig:xg_time} and Figure~\ref{fig:cnn_time}. Each discrete point on the horizontal axis represents a test set, and the vertical axis is the time overhead of each test set running under different models. And we choose not to record time overhead of blank control group, because it is meaningless to take this comparison.

    MDPN is the model without local optimization, all its time overhead is just a neural network's prediction. So it will be orders of magnitude faster than control group. While as for MDPN + LOPT, there is a trade-off between time and accuracy. We take local optimization to eliminate errors in some degree. But this will bring some time loss, which can be easily found in the comparison figure. Despite this, our model is still several times faster than the method in baseline. 
    \begin{figure}[t]
        \subfigure[XGBoost]{
            \centering
            \includegraphics[width=0.49\textwidth]{image/xg_time.jpg}
            \label{fig:xg_time}
        }
        \subfigure[CNN]{
            \centering
            \includegraphics[width=0.49\textwidth]{image/cnn_time.jpg}
            \label{fig:cnn_time}
        }
        \caption{This figure shows time overhead of different models. And the results of XGBoost and CNN are shown in Subfigure~\ref{fig:xg_time} and Subfigure~\ref{fig:cnn_time} respectively.}
        \label{fig:time}
    \end{figure}
    \begin{figure}[t]
		\setlength{\abovecaptionskip}{0.15cm}
        \setlength{\belowcaptionskip}{-0.5cm}
        \begin{minipage}{0.5\textwidth}
            \centering
            \includegraphics[width=0.95\textwidth]{image/xg_accu.png}
            % \caption{XGBoost}
            \label{fig:xg_accu}
        \end{minipage}
        \begin{minipage}{0.5\textwidth}
            \centering
            \includegraphics[width=0.95\textwidth]{image/cnn_accu.png}
            % \caption{CNN}
            \label{fig:cnn_accu}
        \end{minipage}
        \caption{The accuracy on different test sets have been arranged in this figure in the form of a violin diagram. Additionally, we have marked some key statistics in this figure. The results of XGBoost and CNN are shown in the left and right part respectively.}
        \label{fig:accu}
    \end{figure}
    \begin{figure}[t]
		\setlength{\abovecaptionskip}{0.15cm}
        \setlength{\belowcaptionskip}{-0.5cm}
        \begin{minipage}{0.5\textwidth}
           \centering
            \includegraphics[width=0.7\textwidth]{image/xg_bing.png}
            \label{fig:xg_accu_bing}
        \end{minipage}
        \begin{minipage}{0.5\textwidth}
            \centering
            \includegraphics[width=0.7\textwidth]{image/cnn_bing.png}
            \label{fig:cnn_accu_bing}
        \end{minipage}
        \caption{Pie charts about the incremental accuracy of our model to the baseline. The results of XGBoost and CNN are shown in the left and right part respectively.}
        \label{fig:accu_bing}
    \end{figure}
    \subsubsection{Accuracy}
    There is a difference between two experiments on XGBoost and CNN. We haven't conducted MDPN + LOPT model when optimizing hyper-parameters of CNN, because MDPN can already give a satisfactory hyper-parameter set for CNN, and local optimization incurred high time overhead. Here's the results of two experiments respectively.

    Firstly, an overview of accuracy of each test set is shown in Figure~\ref{fig:accu}. On the whole, MDPN can perform better than the methods in baseline, especially the results of optimizing XGBoost. It only fails on very few data sets. While for these data sets, the gap can be bridged by LOPT. In order to better show the comparison results, we have made pie charts about the incremental accuracy of our model to the baseline as Figure~\ref{fig:accu_bing}. On about 75\% data sets, our approach improves the classification accuracy of XGBoost and CNN. On over 20\% data sets, our approach improves the accuracy by over 20 percentage points. This is an exciting result!

