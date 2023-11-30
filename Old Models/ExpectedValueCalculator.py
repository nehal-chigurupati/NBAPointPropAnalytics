from scipy.optimize import minimize, Bounds, LinearConstraint, SR1
import numpy as np

probs = [
    0.7529411912,
    0.7407407165,
    0.6540284157,
    0.78125,
    0.6728972197,
    0.6161616445,
    0.9024389982,
    0.7662337422

]

gains = [
    .0909,
    .087,
    .0909,
    .0833,
    .098,
    .082,
    .0847,
    .0909
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
        tens.append(14)
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
