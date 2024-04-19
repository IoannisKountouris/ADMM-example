import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


# Constants and parameters
D = 100  # Total demand
c = np.array([1, 5, 2], dtype=float)  # Cost coefficients for each generator
capacities = np.array([40, 50, 40], dtype=float)  # Capacity constraints for each generator
rho = 1000  # Penalty parameter

# Initial values
x = np.array([0.0, 0.0, 0.0], dtype=float)   # Starting with a feasible production level for each generator
lambda_hat_i = np.array([0.0, 0.0, 0.0], dtype=float)  # Dual variables for each generator

# Optimization problem for each generator
def solve_qp(i, c_i, p_max_i, lambda_hat_old_i, z_old, rho):
    # Define the variable for generator's production level
    x_i = cp.Variable()
    
    # Define the quadratic cost function for the generator
    cost = c_i * x_i + lambda_hat_old_i * (x_i) + (rho / 2) * cp.square(x_i -z_old)
    
    # Define the constraints for the generator
    constraints = [0 <= x_i, x_i <= p_max_i]
    
    if i == 4:
        # This enforces x_i * 0.66 = 10 for i=2
        constraints.append(x_i * 0.66 == 10)

    # Set up the problem and solve it
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)
    
    # If problem is solved, return the value. If not, return previous value.
    return x_i.value

# ADMM iteration
max_iter = 100
x_history = [np.copy(x)]
lambda_history = [np.copy(lambda_hat_i)]  # Initialize lambda history
z=10 #initial
z_history = [np.copy(z)]

for iteration in range(max_iter):
    # Save current x values before they are updated
    x_old = np.copy(x)
    x_avg_old = np.mean(x)
    lambda_hat_old = np.copy(lambda_hat_i)
    z_old = np.copy(z)

    # x-update step for each generator
    for i in range(3):
        x[i] = solve_qp(i, c[i], capacities[i], lambda_hat_old[i], z_old , rho)

    z =  + np.sum(x)/3 + np.sum((1/rho)* lambda_hat_old)/3 
    z_history.append(np.copy(z))
    x_mean = np.mean(x)

  # Update lambda_hat_i for each generator
    for i in range(3):
        lambda_hat_i[i] += +rho * (x[i] - z)
       
    lambda_history.append(np.copy(lambda_hat_i))
    x_history.append(np.copy(x))
    
# Convergence check (simple version)
    if np.sum(np.linalg.norm(z - x)) < 1e-4:
        break

x_history = np.array(x_history)
lambda_history = np.array(lambda_history)


# Print results
print("Final Production levels:", x)
print("Total production:", np.sum(x))
print("Total demand:", D)
print("Iterations:", iteration + 1)

# Plotting
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(x_history[:, i], label=f'Generator {i+1}')
plt.xlabel('Iteration')
plt.ylabel('Production Level')
plt.title('Convergence of Production Levels by Generator')
plt.legend()
plt.grid(True)
plt.show()


print("price_clearing:", lambda_hat_i)
plt.plot(lambda_history)
plt.show()

print("Z value:", z)
plt.plot(z_history)
plt.show()
