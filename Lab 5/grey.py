# Sample dataset
X = [
    [1.0, 2.0, None, 4.0],
    [2.0, None, 3.0, 4.0],
    [3.0, 2.0, 3.0, None],
    [None, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
]
y = [0, 1, 0, 1, 0]

# Replace None with 0 (manual handling of missing data)
def replace_none(data):
    return [[0 if val is None else val for val in row] for row in data]

X = replace_none(X)

# Split data into training and validation sets
X_train = X[:3]
y_train = y[:3]
X_val = X[3:]
y_val = y[3:]

# Calculate mean or median for a column
def calculate_statistic(data, col_index, strategy="mean"):
    values = [row[col_index] for row in data if row[col_index] != 0]
    if strategy == "mean":
        return sum(values) / len(values) if values else 0
    elif strategy == "median":
        sorted_values = sorted(values)
        n = len(sorted_values)
        return (sorted_values[n // 2] if n % 2 != 0 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2) if values else 0

# Fitness function for GWO
def fitness_function(config, X_train, y_train, X_val, y_val):
    imputation_strategy = ["mean", "median"][int(config[0])]
    scaling_strategy = ["none", "minmax"][int(config[1])]

    # Impute missing values in training and validation sets
    for col in range(len(X_train[0])):
        impute_value = calculate_statistic(X_train, col, imputation_strategy)
        for row in X_train:
            if row[col] == 0:
                row[col] = impute_value
        for row in X_val:
            if row[col] == 0:
                row[col] = impute_value

    # Apply scaling if specified
    if scaling_strategy == "minmax":
        min_vals = [min(row[col] for row in X_train) for col in range(len(X_train[0]))]
        max_vals = [max(row[col] for row in X_train) for col in range(len(X_train[0]))]

        for dataset in (X_train, X_val):
            for row in dataset:
                for col in range(len(row)):
                    row[col] = (row[col] - min_vals[col]) / (max_vals[col] - min_vals[col]) if max_vals[col] - min_vals[col] != 0 else 0

    # Dummy classifier: fitness = number of correct predictions
    correct_predictions = 0
    for i, row in enumerate(X_val):
        prediction = sum(row) % 2  # Simple rule for binary prediction
        if prediction == y_val[i]:
            correct_predictions += 1

    return correct_predictions

# Grey Wolf Optimizer implementation
def grey_wolf_optimizer(fitness_func, n_wolves, n_iterations, X_train, y_train, X_val, y_val):
    # Initialize wolves' positions randomly
    wolves = [[1, 0] for _ in range(n_wolves)]  # 2 parameters: imputation and scaling

    # Best solutions
    alpha, beta, delta = None, None, None
    alpha_fitness, beta_fitness, delta_fitness = float('-inf'), float('-inf'), float('-inf')

    for iteration in range(n_iterations):
        for i, wolf in enumerate(wolves):
            fitness = fitness_func(wolf, X_train, y_train, X_val, y_val)

            # Update alpha, beta, and delta
            if fitness > alpha_fitness:
                alpha, alpha_fitness = wolf[:], fitness
            elif fitness > beta_fitness:
                beta, beta_fitness = wolf[:], fitness
            elif fitness > delta_fitness:
                delta, delta_fitness = wolf[:], fitness

        # Update wolves' positions
        for i in range(n_wolves):
            for j in range(len(wolves[i])):
                A1, A2, A3 = (2 * (1 - iteration / n_iterations) * val for val in [0.5, 0.3, 0.2])
                C1, C2, C3 = 2, 2, 2

                D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                D_beta = abs(C2 * beta[j] - wolves[i][j])
                D_delta = abs(C3 * delta[j] - wolves[i][j])

                X1 = alpha[j] - A1 * D_alpha
                X2 = beta[j] - A2 * D_beta
                X3 = delta[j] - A3 * D_delta

                wolves[i][j] = (X1 + X2 + X3) / 3

    return alpha  # Return the best configuration found

# Run Grey Wolf Optimizer
best_config = grey_wolf_optimizer(fitness_function, n_wolves=5, n_iterations=5, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

# Decode the best configuration
imputation_methods = ["mean", "median"]
scaling_methods = ["none", "minmax"]

print("Best Configuration:")
print("Imputation Strategy:", imputation_methods[int(best_config[0])])
print("Scaling Strategy:", scaling_methods[int(best_config[1])])
