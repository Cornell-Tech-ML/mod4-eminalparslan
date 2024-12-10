from typing import Tuple, Optional

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    # reshape input to group pixels into kernel-sized blocks
    input = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # transpose to get final tiled tensor
    input = input.permute(0, 1, 2, 4, 3, 5)
    input = input.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return input, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D"""
    tiled, height, width = tile(input, kernel)
    pooled = tiled.mean(dim=4)
    return pooled.view(input.shape[0], input.shape[1], height, width)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D"""
    tiled, height, width = tile(input, kernel)
    pooled = max(tiled, dim=4)
    return pooled.view(input.shape[0], input.shape[1], height, width)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        """Forward pass for max reduction."""
        ctx.save_for_backward(t1, dim)
        if dim is None:
            # no dimension -> max across flattened tensor
            return t1.f.max_reduce(t1.contiguous().view(int(t1.size)), 0)
        else:
            return t1.f.max_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for max reduction.

        The gradient flows back only to the positions that were equal to
        the maximum value during the forward pass.
        """
        t1, dim = ctx.saved_values

        if dim is None:
            t1_flat = t1.contiguous().view(t1.size)
            max_val = t1_flat.f.max_reduce(t1_flat, 0)
            mask = t1_flat.f.eq_zip(t1_flat, max_val)
            # count number of max elements along reduction dimension
            count = mask.f.add_reduce(mask, 0)
            # distribute gradient to all max positions equally
            out = mask * (grad_output / count)
            return out.view(t1.shape), 0.0
        else:
            max_val = t1.f.max_reduce(t1, int(dim.item()))
            mask = t1.f.eq_zip(t1, max_val)
            # count number of max elements along reduction dimension
            count = mask.f.add_reduce(mask, int(dim.item()))
            # distribute gradient to all max positions equally
            out = mask * (grad_output / count)
            return out, 0.0


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Max of all elements"""
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor."""
    # numerically stable softmax using max subtraction
    max_val = max(input, dim=dim)
    exp_val = (input - max_val).exp()
    sum_val = exp_val.sum(dim=dim)
    return exp_val / sum_val


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor."""
    # numerically stable logsoftmax using log-sum-exp trick
    max_val = max(input, dim=dim)
    exp_val = (input - max_val).exp()
    sum_val = exp_val.sum(dim=dim)
    return input - max_val - sum_val.log()


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: input tensor
        rate: probability of dropping a position
        ignore: if True, don't apply dropout

    """
    if ignore:
        return input

    mask = rand(input.shape, input.backend) > rate
    # scale output to maintain expected value
    # scale = 1.0 / (1.0 - rate)
    return mask * input  # * scale
