import numpy as np
import itertools
from cvxpy import *
import cvxpy as cp

A = np.loadtxt('A.txt')
B = np.loadtxt('B.txt')
ks = np.arange(1,5000,500)

def Mat_Comp(mat,k):
    indices = np.array(list(itertools.product(range(mat.shape[0]),range(mat.shape[1]))))
    N = mat.shape[0]*mat.shape[1]
    missing_ix_num = np.random.choice((np.arange(N)),size = k, replace=False)
    missing_ix = np.array(indices[missing_ix_num])
    known_ix_num = [item for item in range(N) if item not in missing_ix_num]
    known_ix = np.array(indices[known_ix_num])
    known_values = mat[tuple(known_ix.T)]
    to_recover = mat[tuple(missing_ix.T)]
    X = cp.Variable(mat.shape)
    obj = cp.Minimize(cp.norm(X, 'nuc'))
    cons = [ X[tuple(known_ix.T)] == known_values ]
    prob = cp.Problem(obj, cons)
    prob.solve()
    X_star = X.value
    recovered = np.sum(np.isclose(np.abs(X_star[tuple(missing_ix.T)] - to_recover), 0, atol=.01))
    recovered_per = recovered / k
    return recovered_per

valuesA = [matrix_complete_iter(A,kprime) for kprime in ks]
valuesB = [matrix_complete_iter(B,kprime) for kprime in ks]

