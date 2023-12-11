from scipy.optimize import minimize, Bounds, LinearConstraint, SR1
import numpy as np

probs = [
    0.868131876,
    0.655844152,
    0.6065573692,
    0.6623376608,
    0.6756756902,
    0.6395938993,
    0.7714285851,
    0.7231404781
]

gains = [
    .087,
    .0952,
    .1,
    .0769,
    .0877,
    .087,
    .0952,
    .1
]

def coeffs(probabilities, gains):
    ret = []
    for i in range(len(probabilities)):
        ret.append(probabilities[i]*gains[i] - 1 + probabilities[i])
    
    return ret

def objectiveFunction(x):
    objective = 100
    c = coeffs(probs, gains)
    ret = 0.0

    if len(c) != len(x):
        raise Exception("Coefficients don't match dimension of space")
    else:
        for i in range(len(x)):
            ret += float(c[i]) * float(x[i])
    
    return 100 - ret

def generateSumConstraint(dim):
    mat = []
    tens = []
    zeroes = []
    for i in range(dim):
        currRow = []
        tens.append(2)
        zeroes.append(0)
        for j in range(dim):
            currRow.append(1)
        mat.append(currRow)
    return LinearConstraint(mat, tens, zeroes)
    
        
x0 = np.array([0 for i in range(len(probs))])
constraint1 = generateSumConstraint(len(probs))
bnd = [(0, None) for i in range(len(x0))]
res = minimize(objectiveFunction, x0, method='trust-constr', jac="8-point", hess=SR1(), bounds=bnd, constraints=[constraint1])
print(100 - res["fun"])
print(res["x"])
