# Warning
**The use of sparse Linear layer does reduce the number of parameters, but it will occupy more memory to maintain sparse matrices unless there is a more efficient way to store sparse matrices.**

# SparesModelForTorch
SparseLinear and other purning method using torch_sparse

# Features

## SparseLinear.py:

Torch_sparse library is used to implement the Linear Layer of sparse matrix, which can save memory cost and speed up the calculation in theory.

# Requirements
torch_sparse: https://github.com/rusty1s/pytorch_sparse

torch cu116
