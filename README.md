## introduction
\section{Introduction}
Automatic machine learning (AutoML) has gained wide attention and applications in both industry~\cite{monje2008tuning} and academia~\cite{wang2018towards}. Automatic hyper-parameter optimization(auto-HPO) is one of the most critical parts~\cite{thornton2012auto,domhan2015speeding,kotthoff2017auto,maclaurin2015gradient}. The effectiveness of many machine learning algorithms is extremely sensitive to hyper-parameters~\cite{zkx_ref_1,Koprowski2018}. Without a good set of hyper-parameters, the task cannot be solved well even with the optimal algorithm.

Among the hyper-parameter optimization approaches, data-driven methods draw much attention~\cite{young2015optimizing,burger2014image} since they could achieve effective prediction of hyper-parameters based on historical experience implicit in the data. Data-driven method denotes that those methods optimize the hyper-parameters with a training set.

However, data-driven automatic hyper-parameter optimization faces three severe challenges. Firstly, existing systems may involve thousands of machine learning tasks with many hyper-parameters~\cite{zkx_ref_2}. Recalculating hyper-parameters for each task may cause large time overhead. Thus, the optimization process should be efficient. Secondly, a hyper-parameter optimization algorithm should be able to handle high-dimensional parameters~\cite{jankova2015confidence}, which is a common phenomenon for machine learning algorithms~\cite{zkx_ref_3}. Thirdly, while guaranteeing fast access to hyper-parameters, there are still high requirements for the quality of the hyper-parameters.

We propose an approach to solve these problems.
Intuitively, the optimal hyper-parameters are determined by two factors, i.e., the machine learning algorithm and the data. Therefore, under the same algorithm, the hyper-parameters are completely determined by the data. We attempt to investigate the relationship between hyper-parameters and data in this occasion. Considering that each data set corresponds to at least one set of optimal hyper-parameters, we believe that there is a mapping between data set and optimal hyper-parameters. As a result, it's possible to use this mapping to achieve prediction of hyper-parameters directly. And the effectiveness of this idea have been verified by experiments.

Our contributions of this paper are summarized as follows.
\begin{list}{\labelitemi}{\leftmargin=1em}\itemsep 0pt \parskip 0pt
\item We consider the mapping from data to the optimal hyper-parameters and apply this mapping to the selection of the optimal hyper-parameters. On different tasks of an algorithm, the model has strong transferability, which greatly saves time overhead. For this reason, the model can achieve ultra-high-dimensional optimization of hyper-parameters.

\item With XGBoost~\cite{chen2016xgboost} as an example, we design the neural network structure for the mapping as well as training approaches, which could be applied to other machine learning tasks with slight modification.

\item Experimental results on real data demonstrate that the proposed approach significantly outperforms the state-of-art algorithms in both accuracy and efficiency.
\end{list}

In the remaining of this paper,  Section~\ref{sec:method} describes the proposed approach. Experiments are conducted in Section~\ref{sec:exp}. We overview related work in Section~\ref{sec:related}. Section~\ref{sec:con} draws the conclusions.

## flow

## operation method

## experiment

### data set

### baseline

### result

