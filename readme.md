本项目比较了Adam、Adai、AdaptiveLR-Adai和SGDM四种优化算法的性能。


模型采用的是ResNet-18，batchsize为128，数据集为CIFAR-10，pkl中保存了这几种优化算法训练100个epoch的accuracy的list。


Adai算法基于论文Adaptive Inertia:Disentangling the Effects of Adaptive Learning Rate and Momentum。AdaptiveLR-Adai是我自己提出的将Adam的自适应学习率加入Adai的新算法。