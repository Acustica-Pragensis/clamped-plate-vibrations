import numpy as np
from pysr import PySRRegressor
import matplotlib.pyplot as plt

csv_file = f"data/em/chosen_modes/unit_sq_X.csv"

data = np.loadtxt(csv_file, delimiter=',', skiprows=8)

x_axis = data[:, 0]
x_11 = data[:, 1] 
x_13 = data[:, 2]
x_31 = data[:, 3]
x_33 = data[:, 4]

ones = np.ones_like(x_axis)

weights = 2*np.abs(x_axis)**2 + ones
plt.figure(figsize=(8, 5))
plt.plot(x_axis, weights, "b")
plt.show()


y = x_33 /np.max(x_33)
plt.figure(figsize=(8, 5))
plt.plot(x_axis, y, "b")
plt.show()

# X = np.column_stack([
#     x_axis,
#     # x_axis - 0.5*ones,
#     # x_axis + 0.5*ones,
#     np.pi*x_axis,
# ])

X = np.column_stack([
    x_axis,
    x_axis**2,
    # 0.5*ones**2,
    # x_axis - 0.5*ones,
    # x_axis + 0.5*ones,
    # np.pi*x_axis,
    # np.cos(np.pi*x_axis),
    # np.sin(np.pi*x_axis),
    # np.cosh(np.pi*x_axis),
    # np.sinh(np.pi*x_axis),
])

# variable_names=['x']


print(y.shape, X.shape)


my_objective = """
function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    (prediction, completion) = eval_tree_array(tree, dataset.X, options)
    if !completion
        return L(Inf)
    end

    diffs = prediction .- dataset.y
    # err_L1 = sum(diffs .^ 2) / length(diffs)
    rms = 100*sqrt(sum(diffs .^ 2) / sum(dataset.y .^ 2))

    # Evaluate the tree specifically at x=-0.5, 0.5, expecting the output to be 0
    BC = copy(dataset.X)
    BC[1, 1:div(length(diffs), 2)] .= 0.5 
    BC[1, div(length(diffs), 2)+1:end] .= -0.5

    BC[2, 1:div(length(diffs), 2)] .= 0.25 
    BC[2, div(length(diffs), 2)+1:end] .= 0.25

    val_at_BC, flag_BC = eval_tree_array(tree, BC, options)
    
    if !flag_BC
        return L(Inf)
    end

    BC_penalty = sum((val_at_BC .- 0.0).^2)  / length(diffs)

    # return err_L1 + (300* BC_penalty)
    return rms + (300* BC_penalty)

end
"""


model = PySRRegressor(
    populations=128,
    niterations=200,
    population_size = 64,
    binary_operators=["+", "-", "*",  "/"],
    unary_operators=[
        # "sqrt",
        "square",
        # "exp",
        # "sin", 
        # "cos",
        # "sinh",
        # "cosh"
    ],
    constraints={
        # "/": (3, 4),
        # "^": (-3, 3),
        # "square": 3,
        # "cube": 3,
        # "sin": 3,
        # "cos": 3,
        # "sinh":3,
        # "cosh":3
    },
    nested_constraints = {
        # "sqrt":  {"sqrt": 0, "square": 0},
        # "square":  {"square": 0},
        # "sin":  {"sin": 0, "cos": 0},
        # "cos":  {"sin": 0, "cos": 0},
        # "sin":  {"sin": 0, "cos": 0, "sinh":0, "cosh":0},
        # "cos":  {"sin": 0, "cos": 0, "sinh":0, "cosh":0},
        # "sinh": {"sin": 0, "cos": 0, "sinh":0, "cosh":0},
        # "cosh": {"sin": 0, "cos": 0, "sinh":0, "cosh":0},
    },
    
    # elementwise_loss = "L1DistLoss()",
    # elementwise_loss=elementwise_loss,
    # maxdepth = 8,
    maxsize=25,
    loss_function=my_objective,
)


# model.fit(X, y)
model.fit(X, y, weights=weights)

# model.fit(X, y,  variable_names=variable_names)

print(model)


results_equations = model.equations_

for i in range(len(results_equations)):
    eq_row = results_equations.iloc[i]
    complexity = eq_row['complexity']
    loss = eq_row['loss']
    equation_str = str(eq_row['equation'])
    print(complexity, loss, model.latex(i))
    
    ypredict = model.predict(X, index=i)

    print("1st quartile of the error: ", np.quantile(np.abs(ypredict-y),0.25))
    print("2nd quartile of the error: ", np.quantile(np.abs(ypredict-y),0.50))
    print("3rd quartile of the error: ", np.quantile(np.abs(ypredict-y),0.75))
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(x_axis, ypredict, 'r+', label="Prediction", alpha=0.6)
    plt.plot(x_axis, y, 'b+', label="Truth", alpha=0.6)
    plt.ylabel('Value')
    plt.xlabel('Order in Dataset')
    plt.title(f"Fit Comparison")
    # plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    residuals = ypredict - y
    plt.plot(residuals, 'k+')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.ylabel('Absolute Error')
    plt.xlabel('Order in Dataset')
    # plt.yscale('log')
    # plt.title(f"Residual Analysis")

    plt.subplot(1, 3, 3)
    plt.scatter(y, ypredict, alpha=0.5, s=10)
    min_val = min(y.min(), ypredict.min())
    max_val = max(y.max(), ypredict.max())   
    plt.plot(np.linspace(min_val, max_val, 100), np.linspace(min_val, max_val, 100), 'r--')
    plt.xlabel('Truth')
    plt.ylabel('Prediction')
    # plt.xscale('log')
    # plt.yscale('log')

    plt.grid(True, alpha=0.3)

    plt.suptitle(f"Equation Index {i}: $y = {equation_str}$", fontsize=14)
    # plt.suptitle(f"Complexity  {complexity}, Loss: {loss:.2e}", fontsize=14)

    plt.tight_layout()
    
    # plt.savefig(f"diagnostic_eq_{i}.png")
    plt.show()