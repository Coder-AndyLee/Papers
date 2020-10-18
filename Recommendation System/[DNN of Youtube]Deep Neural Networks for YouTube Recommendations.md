# [Paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)
# Reference
1. [Deep Neural Network for YouTube Recommendation论文精读](https://zhuanlan.zhihu.com/p/25343518)
2. [DNN for YouTube Recommendations 论文总结](https://zhuanlan.zhihu.com/p/51440316)
3. [重读Youtube深度学习推荐系统论文，字字珠玑，惊为神文-王喆](https://zhuanlan.zhihu.com/p/52169807)
4. [YouTube深度学习推荐系统的十大工程问题-王喆](https://zhuanlan.zhihu.com/p/52504407)
5. [揭开YouTube深度推荐系统模型Serving之谜-王喆](https://zhuanlan.zhihu.com/p/61827629)
6. 
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
![Pasted image 20201013225918.png](https://github.com/Coder-AndyLee/Papers/tree/master/pic/Pasted%20image%2020201013225918.png)  
![[Pasted image 20201013225918.png]]

4. 任务：从用户观看的历史数据和当前上下文中学习用户的embedding $u$，并使用**softmax**分类器选择最有可能被观看的video（ The task of the deep neural network is to learn user embeddings u as a function of the user’s history and context that are useful for discriminating among videos with a softmax classifier.）
	- 第一反应：用softmax来做推荐在现在看来并不是特别合理（误）
	- 后面看懂了：先使用softmax学习video embedding。线上使用时，再与DNN学到的 user embedding 一起通过K近邻计算召回的视频
	- **trick：使用比较宽泛的特征，便于用户的相似性迁移**
5. Efficient Extreme Multiclass
	- 每次优化正样本和sample一部分负样本（一般选几千个）的交叉熵（For each example the cross-entropy loss is minimized for the true label and the sampled negative classes）。*类似word2vec中Skip-Gram的负采样的方法*
	- 或者使用 hierarchical softmax [F. Morin and Y. Bengio. Hierarchical probabilistic neural network language model.] ，但本文在该方法上没能达到较好的精度
	- Serving：选出与user embedding 最相似的N个video embeddings
		- 早期方法：hashing（？？？局部敏感哈希（Locality Sensitive Hashing））
		- 对nearest neighbor search algorithm的选择对A/B test结果不敏感
		- serving阶段问题转化为 a nearest neighbor search in **the dot product space** for which general purpose libraries can be used 
6. Model Architecture
![[Pasted image 20201018161917.png]]
7. Heterogeneous Signals
	1. 输入三部分特征
		1. User query: tokenized with **unigram or bigram**，过embedding层后求均值，得到query的summarized representation
		2. watch videos：过embedding层后求均值
		3. demographic features (binary or continuous)
			- the user’s gender and age
			- normalized to [0, 1]
			- concatenate with the above embeddings
		4. context infos：设备、登录状态（logged-in state）
	2. 视频上传时间特征（Example Age）
		- 考虑视频上传时间对视频推荐的影响：在相关性约束下，越新的视频理论上更受欢迎
		- 训练：```example_age = time_of_training(训练时间) - time_of_sample_log```
		- serving：将example_age置为0或者微负的值
		- 意义：例如在一段时间以前用户观看了某视频，在训练时该age特征得取一个正值label才能预测为1；而serving时该值为0/微负值，所以probability偏低一些，相当于考虑是时间对推荐倾向的影响
		- 实验证明使用了example_age特征后预测效力的分布更符合经验分布![[Pasted image 20201018165249.png]]
	3. Label and Context Selection
		- 训练样本
			- 不仅考虑YouTube内推荐给用户的数据，还考虑其他嵌入页面用户观看的数据
			- 为每个用户固定训练样本数量，排除活跃用户对loss和结果的影响
		- 抛弃序列信息：把watched video和query的embedding做加权
		- 不对称的浏览模式：图中a方法是以历史浏览的上下文作为特征，某一次视频观看作为label；b方法是以历史浏览作为特征，下一次观看作为label。理论上看b方法更合适，因为A方法取label的方式和实际场景有gap；且b方法在A/B测表现更好。
		![[Pasted image 20201018172626.png]]
## 