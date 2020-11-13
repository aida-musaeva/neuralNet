import numpy as np
import random as r
import math
import sympy as sp
import scipy.optimize
from scipy import optimize
from numpy import*



def make_sample(N):
    c = -28.305
    a0 = -0.000656
    a = np.array([-0.288, 0.628, 0.000141, 3.83])

    X = np.random.randint(0, 2, N)
    print(X)
    if X.all() == 0:
        X[random.randint(0, N - 1)] = 1
    print(X)



    average_profit = r.randint(10000, 120000)
    dist = r.randint(400, 12000)
    k = np.array([[r.uniform(dist/800, 20), r.randint(0, 1), average_profit, math.log(dist, math.e)] for j in range(N)])
    print(k)
    return X, k


def M(price, X, a0 = -0.000656, c = -28.305):
    a = np.array([-0.288, 0.628, 0.000141, 3.83])  # Вектор параметров
    ak = [np.dot(a, k[i]) for i in range(X.shape[0])]
    N = X.shape[0]
    m = np.zeros([N])
    s = 1
    for i in range(N):
        if X[i] == 1:
            s += exp(a0 * price[i] + ak[i] + c)
    for i in range(N):
        m[i] = exp(a0 * price[i] + ak[i]+ c)/s
        if X[i] == 0:
            m[i] = 0
    return m


def f(x, a0 = -0.000656, c = -28.305):
    global X
    N = X.shape[0]
    cost = np.array([4100 for i in range(X.shape[0])])
    a = np.array([-0.288, 0.628, 0.000141, 3.83])  # Вектор параметров
    ak = [np.dot(a, k[i]) for i in range(N)]
    N = X.shape[0]
    f = zeros([N])
    s = 1
    for i in range(N):
        if X[i] == 1:
            s += exp(a0*x[i] + ak[i]+c)
    for i in range(N):
        f[i] = (1 - exp(a0*x[i] + ak[i]+c)/s) * (cost[i] - x[i]) - 1/a0
        if X[i] == 0:
            f[i] = 0
    return f

N = 5
X, k = make_sample(N)
cost = np.array([4100 for it in range(X.shape[0])]) # Себестоимость на человека
x0 = np.zeros([N])
for i in range(N):
    x0[i] = (cost[i] + 1000)*X[i]

sol = optimize.root(f, x0, method='krylov')
P_opt = sol.x
M = M(sol.x, X)
w = (P_opt-cost)*M
print(P_opt)  # Цены
print(M)  # Доли
print(w)