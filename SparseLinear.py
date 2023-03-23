import torch.nn as nn
import torch
import torch_sparse


class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sparse_weight_indices, sparse_weight_values, bias, m, n):
        y = torch_sparse.spmm(sparse_weight_indices, sparse_weight_values, m, n, x).T
        if len(bias) > 0:
            y += bias.unsqueeze(0).expand_as(y)

        ctx.save_for_backward(x, sparse_weight_indices, sparse_weight_values, torch.tensor(m), torch.tensor(n), bias)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, sparse_weight_indices, sparse_weight_values, m, n, bias = ctx.saved_tensors
        grad_weight = grad_bias = None

        if ctx.needs_input_grad[2]:
            grad_weight = grad_output.t().mm(x.t())
            grad_weight = grad_weight[sparse_weight_indices[0], sparse_weight_indices[1]]

        if len(bias) > 0 and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return None, None, grad_weight, grad_bias, None, None


class SparseLinear(nn.Module):
    def __init__(self, weight, bias, sparsity):
        super(SparseLinear, self).__init__()
        self.sparsity = sparsity

        self.in_features = weight.size(1)
        self.out_features = weight.size(0)

        self.sparse_weight_indices = weight.to_sparse().indices()
        self.sparse_weight_values = torch.nn.Parameter(weight.to_sparse().values())
        self.register_parameter("sparse_weight_values", self.sparse_weight_values)

        if bias is not False:
            self.bias = torch.nn.Parameter(bias)
            self.register_parameter('bias', self.bias)
        else:
            self.bias = None

    def forward(self, x):
        if len(x.shape) is 3:
            result = None
            for single_batch in x:
                if result is not None:
                    result = result.cat(SparseLinearFunction.apply(single_batch.T, self.sparse_weight_indices,
                                                                   self.sparse_weight_values, self.bias,
                                                                   self.out_features, self.in_features), dim=0)
                else:
                    result = SparseLinearFunction.apply(single_batch.T, self.sparse_weight_indices,
                                                        self.sparse_weight_values, self.bias,
                                                        self.out_features, self.in_features).unsqueeze(0)
            return result
        else:
            return SparseLinearFunction.apply(x.T, self.sparse_weight_indices, self.sparse_weight_values, self.bias,
                                              self.out_features, self.in_features)

    def extra_repr(self):
        return 'input_features={}, output_features={}, bias={}, sparsity={}'.format(
            self.in_features, self.out_features,
            len(self.bias) > 0, self.sparsity)
