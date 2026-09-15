from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
from copy import deepcopy

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    rhs = list(vals)
    rhs[arg] += epsilon

    lhs = list(vals)
    lhs[arg] -= epsilon

    return (f(*rhs) - f(*lhs)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def dfs(v: Variable, top_sort: list, used: set):
    if v.is_constant() or v.unique_id in used:
        return

    used.add(v.unique_id)
    for p in v.parents:
        dfs(p, top_sort, used)
    top_sort.append(v)

def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    top_sort = list()
    used = set()
    dfs(variable, top_sort, used)
    top_sort = reversed(top_sort)
    return top_sort


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    derivatives = dict()
    derivatives[variable.unique_id] = deriv
    top_sort = topological_sort(variable)

    for v in top_sort:
        d_v = derivatives[v.unique_id]

        if v.is_leaf():
            v.accumulate_derivative(d_v)

        else:
            for p, d_p in v.chain_rule(d_v):
                if p.unique_id not in derivatives:
                    derivatives[p.unique_id] = 0.0
                derivatives[p.unique_id] += d_p


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
