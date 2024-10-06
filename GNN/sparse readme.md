关于如何提取稀疏矩阵
打开C:\Users\Lenovo\Documents\GitHub\Sparse-Zoo\gnn\examples\notebooks\code_tutorial_2.ipynb
找到add features
利用以下代码便可保存
# 保存稀疏矩阵
scipy.sparse.save_npz('sparse_matrix.npz', m)

# 加载稀疏矩阵
m_loaded = scipy.sparse.load_npz('sparse_matrix.npz')