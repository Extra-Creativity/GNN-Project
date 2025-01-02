# Towards Rotation-Invariant Point Cloud Classification

首先进行环境安装（如遇到名称冲突或cuda版本不适合，可在yml中进行调整）：
```commandline
conda env create -f env.yml
cd dgcnn
```

我们的各个模型的运行方式如下；特别地，所有模型要额外加上如下两个参数：
```commandline
--train_rotate=z --test_rotate=SO3 
```
其中具体的取值可以为`z`和`SO3`，表示对训练集/测试集使用何种旋转，上面的参数就是z/z旋转。不加参数时默认为不施加旋转。

1. 使用每个点的localized Gram，特征向量压缩：
    ```commandline
    python main.py --exp_name=dgcnn_local_1024 --model=dgcnn_local --num_points=1024 --k=20 --use_sgd=True
    ```
2. 使用中心的localized Gram，特征向量压缩：
    ```commandline
    python main.py --exp_name=dgcnn_centered_1024 --model=dgcnn_centered --num_points=1024 --k=20 --use_sgd=True
    ```
3. 使用localized Gram的topk值（对应参数`eigen_topk`）：
    ```commandline
    python main.py --exp_name=dgcnn_naive_1024 --model=dgcnn_pca --only_naive=True --num_points=1024 --k=20 --eig_topk=32 --eig_knn_k=32 --use_sgd=True
    ```
4. 仅使用点云中心的localized Gram压缩后作为输入：
    ```commandline
    python main.py --exp_name=dgcnn_brute_1024 --model=dgcnn_brute --num_points=1024 --k=20 --use_sgd=True
    ```
   可以使用`--compact_feature=xx`来指定压缩到的维度。
5. 使用PCA后的点云作为输入：
    ```commandline
    python main.py --exp_name=dgcnn_pca_1024 --model=dgcnn_pca --num_points=1024 --k=20 --eig_topk=0 --use_sgd=True
    ```
6. 使用PCA作为输入，第一层补充使用localized Gram的topk值：
    ```commandline
    python main.py --exp_name=dgcnn_pca_naive_1024 --model=dgcnn_pca --num_points=1024 --k=20 --eig_topk=16 --eig_knn_k=32 --use_sgd=True
    ```
