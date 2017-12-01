from __future__ import division
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.exponential import ExpCone
import numpy as np
from scipy.special import xlogy


class xlogxy(Elementwise):
    """:math:`x\log(x/y)`

    """

    def __init__(self, x, y):
        super(xlogxy, self).__init__(x, y)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        x = values[0]
        y = values[1]
        # TODO return inf outside the domain
        return xlogy(x, x/y)

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (False, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        if np.min(values[0]) <= 0 or np.min(values[1]) <= 0:
            # Non-differentiable.
            return [None, None]
        else:
            div = values[0]/values[1]
            grad_vals = [1+np.log(div), -div]
            grad_list = []
            for idx in range(len(values)):
                rows = self.args[idx].size
                cols = self.size
                grad_list += [xlogxy.elemwise_grad_to_diag(grad_vals[idx],
                                                           rows, cols)]
            return grad_list

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >= 0, self.args[1] >= 0]

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        x = Elementwise._promote(arg_objs[0], shape)
        y = Elementwise._promote(arg_objs[1], shape)
        t = lu.create_var(shape)
        constraints = [ExpCone(t, x, y),
                       lu.create_geq(y)]  # 0 <= y
        # -t
        obj = lu.neg_expr(t)
        return (obj, constraints)
