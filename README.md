# Sparse Zoo

This repository is used to collect and store workloads for various sparse algorithms.

## How to contribute

1. fork本仓库的main分支，在本地新建一个文件夹添加新的workload

2. 添加readme，说明怎么使用代码（如,如何提取矩阵乘）

3. 每个workload需要满足以下的条件:

- 从零构建（或最简版本）
- 可以提取出稀疏计算的矩阵a和矩阵b（简单来说就是输出torch.matmul(a,b)的a和b）
- 支持由 buddy mlir lower 到linalg

4. 你可以在licence添加你的名字或者id，我们最后会一起合并到实验室仓库。

5. 提交pr :blush:
