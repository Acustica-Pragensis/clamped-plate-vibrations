import numpy as np
import re, sympy as sp

import matplotlib.pyplot as plt
from pysr import PySRRegressor, TemplateExpressionSpec

data = np.loadtxt('data/ef/ef_pysr.csv', skiprows=1)

y = data[:,0].astype(np.float64)
X = data[:,1:].astype(np.float64)
print(X.shape, y.shape)
print(X.dtype, y.dtype)

mask = np.isfinite(y)          # True where y is a usable number (also kills inf)
print(f"dropping {np.count_nonzero(~mask)} of {len(y)} rows")

X = X[mask]
y = y[mask]
print(X.shape, y.shape)


E = 170e+9      # in Pa Young modulus
nu = 0.28       # (-) Poisson ratio
rho = 2329      # kg/m^3 plate density
K = np.pi / 8 * np.sqrt(E / (12 * (1 - nu**2) * rho))

template = TemplateExpressionSpec(
    expressions=["f"],
    variable_names=["a", "b", "h", "m", "n"],
    combine=(
        f"{K:.9f}*h"
        "*(((m + f(a/b,m,n))/a)^2 + ((n + f(b/a,n,m))/b)^2)"
    ),
)

err = """function loss_fnc(prediction, target)
    scatter_loss = abs(prediction - target) / (abs(target))
    return scatter_loss
end
"""

model = PySRRegressor(
    populations=50,
    population_size=100,
    niterations=500,
    model_selection="best",
    binary_operators=["*", "+", "-", "/"],
    unary_operators=[
        "square",
        "inv"
    ],
    nested_constraints={
        "square": {"square": 0},
        "inv":   {"inv": 0}
    },
    constraints={"square": 2}, 
    maxsize = 20,
    expression_spec=template,
    elementwise_loss=err,
    precision=64,
)

model.fit(
    X, 
    y, 
    variable_names=["a", "b", "h", "m", "n"],
)

print(model.get_best().equation)

ypredict = model.predict(X)


plt.scatter(y, ypredict)
# plt.plot(np.linspace(0,0.002,100), np.linspace(0,0.002,100), 'r--')
# plt.plot(np.linspace(-0.1,1,100), np.linspace(-0.1,1,100), 'r--')
plt.xlabel('Truth')
plt.ylabel('Prediction')
plt.show()
