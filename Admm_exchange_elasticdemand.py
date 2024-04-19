import numpy as np
import matplotlib.pyplot as plt
from gurobipy import GRB, Model
import time


# Start the timer
start_time = time.time()

# Constants and parameters
D0 = 100  # Total demand #here we assume inelastic
c = np.array([1, 5, 2], dtype=float)  # Cost coefficients for each generator
capacities = np.array([40, 50, 40], dtype=float)  # Capacity constraints for each generator
rho = 0.01 # Penalty parameter
 
# Initial values
x = np.array([0.0, 0.0, 0.0], dtype=float)  # Starting with a feasible production level for each generator
lambda_hat = 0  # Dual variables for each generator

# Assuming k is the rate at which demand decreases as price increases
k = 0.5  # This is a hypothetical value; adjust based on your scenario

# Function to calculate demand based on price
def demand_function(price, D0, k):
    D_hat = D0 - k * price
    return D_hat

def solve_qp_with_gurobi(c_i, p_max_i, lambda_hat, x_i_old, x_ave_old, rho, D):
    # Create a new Gurobi model
    m = Model()

    # Suppress the Gurobi solver output log
    m.setParam('OutputFlag', 0)

    # Add variable for generator's production level with lower bound 0 and upper bound p_max_i
    x_i = m.addVar(lb=0, ub=p_max_i, name="x_i")

    # Integrate new variables into the model
    m.update()

    # Define the quadratic term (the squared part of the norm expression)
    quadratic_term = (x_i - (x_i_old - 3 * x_ave_old) - D)**2

    # Define the full cost function for the generator
    cost = c_i * x_i + lambda_hat * x_i + (rho / 2) * quadratic_term

    # Set the objective to minimize cost
    m.setObjective(cost, GRB.MINIMIZE)

    # Solve the model
    m.optimize()

    # Check if the solution is found and return the optimal value of x_i
    if m.status == GRB.OPTIMAL:
        return x_i.X
    else:
        print("Optimal solution was not found.")
        return None

# ADMM iteration
max_iter = 300
x_history = [np.copy(x)]
lambda_history = [np.copy(lambda_hat)]  # Initialize lambda history


for iteration in range(max_iter):
    # Save current x values before they are updated
    x_old = np.copy(x)
    x_avg_old = np.mean(x_old)
    lambda_hat_old = np.copy(lambda_hat)

    D = demand_function(-lambda_hat_old, D0, k)
    
    # x-update step for each generator
    for i in range(3):
        x[i] = solve_qp_with_gurobi(c[i], capacities[i], lambda_hat, x_old[i],  x_avg_old,  rho, D)
    
    x_avg = np.mean(x)
    
    lambda_hat += rho * (-D + sum(x))
    #or 
    # lambda_hat += rho * (D - 3*x_avg)
    lambda_history.append(np.copy(lambda_hat))
    
    x_history.append(np.copy(x))


    # Check for lambda change for termination condition
    if np.linalg.norm(lambda_hat - lambda_hat_old) < 1e-4:
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

#recall the sing of lambda is based on the decomposition
print("price_clearing:", -lambda_hat)
plt.plot(-lambda_history)

# End the timer
end_time = time.time()

# Calculate the total duration
duration = end_time - start_time

print("Total execution time: {:.2f} seconds".format(duration))