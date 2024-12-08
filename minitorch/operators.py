"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Sequence, List, Callable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> float:
    """Less than."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Equals."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Maximum."""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Is close in value."""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """ReLU function."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Logarithm."""
    return math.log(x + 1e-6)


def exp(x: float) -> float:
    """Exponential."""
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg."""
    return y / (x + 1e-6)


def inv(x: float) -> float:
    """Inverse."""
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of inverse times a second arg."""
    return -(1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of relu times a second arg."""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float], ls: List[float]) -> List[float]:
    """Apply a function over the elements of a list."""
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float], ls1: List[float], ls2: List[float]
) -> List[float]:
    """Apply a function over the elements of two lists.

    Args:
    ----
        fn: The function to apply.
        ls1: The first list.
        ls2: The second list.

    Returns:
    -------
        The result of applying the function to the elements of the two lists.

    """
    assert len(ls1) == len(ls2)
    res = []
    for i in range(len(ls1)):
        res.append(fn(ls1[i], ls2[i]))
    return res


def reduce(fn: Callable[[float, float], float], ls: List[float]) -> float:
    """Reduce a list using a function.

    Args:
    ----
        fn: The function to reduce with.
        ls: The list to reduce.

    Returns:
    -------
        The reduced value.

    """
    if len(ls) == 0:
        return 0.0
    x = ls[0]
    for i in ls[1:]:
        x = fn(x, i)
    return x


def addLists(ls1: List[float], ls2: List[float]) -> List[float]:
    """Add two lists together."""
    return zipWith(add, ls1, ls2)


def negList(ls: List[float]) -> List[float]:
    """Negate a list."""
    return map(neg, ls)


def sum(ls: List[float]) -> float:
    """Sum a list."""
    return reduce(add, ls)


def prod(ls: Sequence[float | int]) -> float | int:
    """Take the product of a list."""
    return reduce(mul, list(ls))
