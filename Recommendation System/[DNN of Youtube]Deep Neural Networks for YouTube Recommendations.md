# [Paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)
# Introduction
1. YouTobe's challenge
	1. Scale
		- Highly specialized distributed learning algorithms and efficient serving systems are essential for handling YouTube’s massive user base and corpus.
	2. Freshness（冷启动）
		- Trade off between 'fresh' videos and well-established videos
		- an exploration/exploitation perspective(角度)
	3. Noise
		- implicit feedback 和 content feature中都有大量的噪声
2. Scale of model
	- one billion parameters
	- hundreds of billions of examples
# System Overview
##  1. Recomendation system architecture
![[Pasted image 20201013223757.png]]

##  2. Composition
1. A neural network for ***candidate generation* **and a neural network for ***ranking***
2. candidate generation network（粗排/召回）
	- provides broad personalizaiton via *collborative filtering*(协同过滤)
	- utilize simple features: **ID**, **search query tokens** and **demographics**(人口统计学特征)
3. ranking network（精排/排序）
	- distinguish relative importance among candidates with high recall
4. Metrics
	- offline：precision, racall, ranking loss, etc.
	- online：click-through rate (ctr), watch time and many other metrics that measure user engagement（用户参与度）.

## Candidate Generation
1. previous approach
	- FM with rank loss
2. Formulation
	1. a specific video watch $w_t$ 
	2. time $t$ 
	3. videos $i$
	4. from a corpus $V$ 
	5. user $U$
	6. context $C$
	7. $u \in R^N$ represents a high-dimensional “embedding” of the (user, context) pair
	8. $v_j \in R^N$ represent embeddings of each candidate video
	$$P(w_t=i|U,C)=\frac{e^{v_iu}}{\Sigma_{j\in V}e^{v_ju}}$$
![Pasted image 20201013225918.png](pic/Pasted image 20201013225918.png)
![[Pasted image 20201013225918.png]]

4. 任务：从用户观看的历史数据和当前上下文中学习用户的embedding $u$，并使用**softmax**分类器选择最有可能被观看的video（ The task of the deep neural network is to learn user embeddings u as a function of the user’s history and context that are useful for discriminating among videos with a softmax classifier.）
	- 第一反应：用softmax来做推荐在现在看来并不是特别合理（误）
	- 后面看懂了：先使用softmax学习video embedding。线上使用时，再与DNN学到的 user embedding 一起通过K近邻计算召回的视频
5. Efficient Extreme Multiclass
	- 每次优化正样本和sample一部分负样本（一般选几千个）的交叉熵（For each example the cross-entropy loss is minimized for the true label and the sampled negative classes）。*类似word2vec中Skip-Gram的负采样的方法*
	- 或者使用 hierarchical softmax [F. Morin and Y. Bengio. Hierarchical probabilistic neural network language model.] ，但本文在该方法上没能达到较好的精度
	- Serving：选出与user embedding 最相似的N个video embeddings
		- 早期方法：hashing（？？？局部敏感哈希（Locality Sensitive Hashing））
		- 对nearest neighbor search algorithm的选择对A/B test结果不敏感
		- serving阶段问题转化为 a nearest neighbor search in **the dot product space** for which general purpose libraries can be used 


## 