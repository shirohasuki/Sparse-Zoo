import torch
import pandas as pd

class QuantLinearLUT(torch.nn.Module):
    def __init__(self, bits, infeatures, outfeatures, bias, include_sparse, numvals, topX, balanced, num_nonzero_per_thread):
        super(QuantLinearLUT, self).__init__()
        self.bits = bits
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bias = bias
        self.include_sparse = include_sparse
        self.numvals = numvals
        self.topX = topX
        self.balanced = balanced
        self.num_nonzero_per_thread = num_nonzero_per_thread

        if numvals > 0:
            self.register_buffer("rows", torch.zeros(outfeatures + 1, dtype=torch.int32))
            self.register_buffer("cols", torch.zeros(numvals, dtype=torch.int32))
            self.register_buffer("vals", torch.zeros(numvals, dtype=torch.float32))

    def pack2(self, linear, lookup_table, include_sparse, num_nonzero_per_thread):
        if include_sparse:
            # 生成虚拟稀疏矩阵数据
            rows = [0]
            cols = []
            vals = []
            for i in range(self.outfeatures):
                for j in range(num_nonzero_per_thread):
                    col = torch.randint(0, self.infeatures, (1,))
                    val = torch.randn(1)
                    cols.append(col.item())
                    vals.append(val.item())
                rows.append(len(cols))

            self.rows = torch.tensor(rows, dtype=torch.int32)
            self.cols = torch.tensor(cols, dtype=torch.int32)
            self.vals = torch.tensor(vals, dtype=torch.float32)

    def forward(self, x):
        if self.bits == 3:
            x = x.float()
            if self.include_sparse and self.topX > 0:
                raise NotImplementedError("Hybrid spmv not implemented")
            elif self.include_sparse and self.balanced:
                raise NotImplementedError("Balanced spmv not implemented")
            elif self.include_sparse:
                y = torch.zeros((x.shape[0], self.outfeatures), dtype=torch.float32)
                quant_cuda.vecquant3matmul_spmv_nuq_perchannel(
                    self.rows,
                    self.cols,
                    self.vals,
                    x,
                    y,
                    self.outfeatures,
                    self.qweight,
                    self.lookup_table,
                )
            else:
                raise NotImplementedError("Dense matmul not implemented")
        return y

# 创建 QuantLinearLUT 实例
quant_linear = QuantLinearLUT(
    bits=3,
    infeatures=1024,
    outfeatures=1024,
    bias=True,
    include_sparse=True,
    numvals=10000,
    topX=0,
    balanced=False,
    num_nonzero_per_thread=10
)

# 填充稀疏矩阵
quant_linear.pack2(None, None, include_sparse=True, num_nonzero_per_thread=10)

# 获取稀疏矩阵的 CSR 表示
rows = quant_linear.rows
cols = quant_linear.cols
vals = quant_linear.vals

# 转换为稀疏矩阵对象
sparse_matrix = torch.sparse_csr_tensor(rows, cols, vals, size=(quant_linear.outfeatures, quant_linear.infeatures))

# 转换为密集矩阵（可选）
dense_matrix = sparse_matrix.to_dense()

# 将稀疏矩阵保存为 CSV 文件
sparse_df = pd.DataFrame({
    'row': rows[:-1].repeat_interleave(rows[1:] - rows[:-1]),
    'col': cols,
    'value': vals
})
sparse_df.to_csv('sparse_matrix.csv', index=False)

# 将密集矩阵保存为 CSV 文件
dense_df = pd.DataFrame(dense_matrix.numpy())
dense_df.to_csv('dense_matrix.csv', index=False)

print("稀疏矩阵已保存到 sparse_matrix.csv")
print("密集矩阵已保存到 dense_matrix.csv")