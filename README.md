# SOME_TITLE
写了两个简单的model（model.py里），利用的是那个矩阵的特征值和特征向量的旋转不变；由于整个群体的特征计算太慢，因此局部
进行计算。目前修改的比较少，就是对于KNN里的局部计算上述特征，而且没有逐层计算。在传入的参数里`eig_topk`表示选择k个
特征值，`eig_knn_k`表示求特征值的KNN范围。
```commandline
python main.py --exp_name=dgcnn_eigval_1024 --model=dgcnn_eigval --num_points=1024 --k=20 --eig_topk=3 --eig_knn_k=20 --use_sgd=True --epochs=10
```
特别地，求特征值时使用的`torch.no_grad()`，即不从特征值向前传播梯度。
+ `DGCNN_Eigval`，就是取topk个特征值，拼接到最初始的特征（即三维坐标）上面；网络就是把最开始的通道数量改为(3 + topk)*2，
   *2是由于DGCNN用的是$f_j-f_i,f_i$的拼接。
+ `DGCNN_Eigvec`，写了两个版本
   + 上一个commit里是取topk特征值对应的特征向量，拼接到网络第一层卷积后的输出（`(batch_size, channel_num, k, num_points)`， 
   给`channel_num`加了`eigen_knn_k`的维度）。
   + 目前版本是把$f_j-f_i,f_i$换为$\text{eigen vector},f_i$，因为特征向量表达的也是局部特征。

这些都是只对最初的特征进行修改，没有在后续的动态图中继续重新求特征向量。

在`data.py`的Dataset里额外加了随机旋转。4080S里1个epoch只需要1min，还算比较快。

### TODO
1. 以上这些模型，包括DGCNN本体，可以train一下，看看加入这种旋转不变的特征有没有准确率的改善。
   DGCNN重新train是因为它提供的pretrain训练时没有随机旋转（当然也可以测试一下这个
   pretrain模型准确率，这个就是跑一下测试集，比较快）。
2. 由于之前说的局部计算特征会失去全局特征的唯一性（就是大卸八块，每块独立旋转的问题），因此
   这里还是保留了DGCNN原来的全局特征，旋转不变特征只是hint。可以考虑一下能不能构建出比较好的
   全局特征，这样就完全不用旋转敏感的全局特征，比如之前说的重心间再搞一下特征向量之类的。

