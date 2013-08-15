"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy import *
import cvxopt
# Problem data
T = 10
n,p = (10,5)
A = cvxopt.normal(n,n)
B = cvxopt.normal(n,p)
x_init = cvxopt.normal(n)
x_final = cvxopt.normal(n)

# Object oriented optimal control problem.
class Stage(object):
    def __init__(self, A, B, x_prev):
        self.x = Variable(n)
        self.u = Variable(p)
        self.cost = sum(square(self.u)) + sum(abs(self.x))
        self.constraint = (self.x == A*x_prev + B*self.u)

stages = [Stage(A, B, x_init)] 
for i in range(T):
    stages.append(Stage(A, B, stages[-1].x))

obj = sum(s.cost for s in stages)
constraints = [stages[-1].x == x_final]
map(constraints.append, (s.constraint for s in stages))
print Problem(Minimize(obj), constraints).solve()