import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Given sensor data
# -----------------------------
x_nodes = np.array([0, 10, 20, 35, 50, 65, 80, 90, 100], dtype=float)
y_nodes = np.array([25.0, 26.7, 29.4, 33.2, 35.5, 36.1, 37.8, 38.9, 40.0], dtype=float)

# -----------------------------------------------------------
# Utility: choose nearest (degree+1) nodes around a point x0
# -----------------------------------------------------------
def choose_nearest_nodes(x_nodes, y_nodes, x0, degree):
    k = degree + 1
    idx_sorted = np.argsort(np.abs(x_nodes - x0))
    chosen_idx = np.sort(idx_sorted[:k])
    return x_nodes[chosen_idx], y_nodes[chosen_idx]

# -----------------------------------------------------------
# Lagrange interpolation evaluation
# -----------------------------------------------------------
def lagrange_eval(xi, yi, x):
    n = len(xi)
    total = 0.0
    for j in range(n):
        term = yi[j]
        for m in range(n):
            if m != j:
                term *= (x - xi[m]) / (xi[j] - xi[m])
        total += term
    return total

# -----------------------------------------------------------
# Newton divided differences
# -----------------------------------------------------------
def newton_coeffs(xi, yi):
    n = len(xi)
    dd = np.zeros((n, n))
    dd[:, 0] = yi
    for j in range(1, n):
        for i in range(n - j):
            dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (xi[i + j] - xi[i])
    return dd[0, :], xi

def newton_eval(coeffs, xi, x):
    n = len(coeffs)
    result = coeffs[-1]
    for k in range(n - 2, -1, -1):
        result = result * (x - xi[k]) + coeffs[k]
    return result

# -----------------------------------------------------------
# Main computation for degrees 2 to 8
# -----------------------------------------------------------
degrees = range(2, 9)  # 2 to 8
x0 = 45.0  # prediction point
results = []

for deg in degrees:
    if deg == 8:
        xi, yi = x_nodes, y_nodes
    else:
        xi, yi = choose_nearest_nodes(x_nodes, y_nodes, x0, deg)

    # Lagrange interpolation
    T_lagrange = lagrange_eval(xi, yi, x0)

    # Newton interpolation
    coeffs, xi_new = newton_coeffs(xi, yi)
    T_newton = newton_eval(coeffs, xi_new, x0)

    results.append((deg, T_lagrange, T_newton))

# -----------------------------------------------------------
# Display table of predicted values
# -----------------------------------------------------------
df = pd.DataFrame(results, columns=["Degree", "Lagrange_T(45)", "Newton_T(45)"])
print("Predicted temperatures at x = 45 km:")
print(df.to_string(index=False))

# -----------------------------------------------------------
# Generate interpolation curves for visualization
# -----------------------------------------------------------
xx = np.linspace(0, 100, 400)
plt.figure(figsize=(10, 6))
plt.scatter(x_nodes, y_nodes, color='black', label='Sensor Data', zorder=5)

for deg in degrees:
    if deg == 8:
        xi, yi = x_nodes, y_nodes
    else:
        xi, yi = choose_nearest_nodes(x_nodes, y_nodes, x0, deg)

    # Lagrange and Newton curves for interpolation
    yy_L = [lagrange_eval(xi, yi, x) for x in xx]
    coeffs, xi_new = newton_coeffs(xi, yi)
    yy_N = [newton_eval(coeffs, xi_new, x) for x in xx]

    # Plot curves
    plt.plot(xx, yy_L, label=f"Lagrange deg {deg}")
    plt.plot(xx, yy_N, '--', label=f"Newton deg {deg}")

plt.title("Temperature Interpolation along 100 km Highway (Degrees 2-8)")
plt.xlabel("Distance (km)")
plt.ylabel("Temperature (°C)")
plt.legend(fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# Convergence curve: T(45) vs polynomial degree
# -----------------------------------------------------------
T_L, T_N = [], []
for deg in degrees:
    if deg == 8:
        xi, yi = x_nodes, y_nodes
    else:
        xi, yi = choose_nearest_nodes(x_nodes, y_nodes, x0, deg)
    T_L.append(lagrange_eval(xi, yi, x0))
    coeffs, xi_new = newton_coeffs(xi, yi)
    T_N.append(newton_eval(coeffs, xi_new, x0))

plt.figure(figsize=(8, 5))
plt.plot(degrees, T_L, 'o-', label='Lagrange T(45)')
plt.plot(degrees, T_N, 's--', label='Newton T(45)')
plt.title("Convergence of T(45) with Increasing Polynomial Degree (2 to 8)")
plt.xlabel("Polynomial Degree")
plt.ylabel("Predicted Temperature (°C)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
