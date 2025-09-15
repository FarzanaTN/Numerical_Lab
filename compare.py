import matplotlib.pyplot as plt
from lab_1.problem_1 import Bisection
from lab_2.a import FalsePosition

# Run both methods with the same parameters
a, b = 0, 1.3
eps_relative = 0.0005  # Consistent tolerance (matches problem_1.py)
bisection_data = Bisection(a, b, eps_relative)
false_position_data = FalsePosition(a, b, eps_relative)

# Extract data for plotting
bisection_iterations = [row[0] for row in bisection_data]
bisection_errors = [row[5] if row[5] is not None else 0 for row in bisection_data]
false_position_iterations = [row[0] for row in false_position_data]
false_position_errors = [row[5] if row[5] is not None else 0 for row in false_position_data]

# Plot both methods on the same graph
plt.figure(figsize=(10, 6))
plt.plot(bisection_iterations, bisection_errors, marker='o', label='Bisection Method', color='blue')
plt.plot(false_position_iterations, false_position_errors, marker='s', label='False Position Method', color='red')
plt.xlabel('Iteration')
plt.ylabel('Approximate Relative Error (%)')
plt.title('Convergence of Bisection vs False Position Methods')
plt.grid(True)
plt.legend()
plt.show()