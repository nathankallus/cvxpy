from cvxpy.atoms.affine.promote import promote
from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.variable import Variable


def xlogxy_canon(expr, args):
    shape = expr.shape
    x = promote(args[0], shape)
    y = promote(args[1], shape)
    t = Variable(shape)
    constraints = [ExpCone(t, x, y), y >= 0]
    obj = - t
    return obj, constraints
