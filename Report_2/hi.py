import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, expand, simplify

# -----------------------------
# Given sensor data
# -----------------------------
x_nodes = np.array([0, 10, 20, 35, 50, 65, 80, 90, 100], dtype=float)
y_nodes = np.array([25.0, 26.7, 29.4, 33.2, 35.5, 36.1, 37.8, 38.9, 40.0], dtype=float)

# -----------------------------
# Utility: choose nearest (degree+1) nodes around x0
# -----------------------------
def choose_nearest_nodes(x_nodes, y_nodes, x0, degree):
    k = degree + 1
    idx_sorted = np.argsort(np.abs(x_nodes - x0))
    chosen_idx = np.sort(idx_sorted[:k])
    return x_nodes[chosen_idx], y_nodes[chosen_idx]

# -----------------------------
# Lagrange interpolation
# -----------------------------
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

def lagrange_poly_sym(xi, yi):
    x = symbols('x')
    n = len(xi)
    poly = 0
    for j in range(n):
        term = yi[j]
        for m in range(n):
            if m != j:
                term *= (x - xi[m]) / (xi[j] - xi[m])
        poly += term
    return simplify(expand(poly))

# -----------------------------
# Newton divided differences
# -----------------------------
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

def newton_poly_sym(coeffs, xi):
    x = symbols('x')
    n = len(coeffs)
    poly = coeffs[-1]
    for k in range(n - 2, -1, -1):
        poly = poly * (x - xi[k]) + coeffs[k]
    return simplify(expand(poly))

# -----------------------------
# Main computation
# -----------------------------
degrees = range(2, 9)  # 2 to 8
x0 = 45.0  # prediction point
results = []

prev_T_L, prev_T_N = None, None

for deg in degrees:
    if deg == 8:
        xi, yi = x_nodes, y_nodes
    else:
        xi, yi = choose_nearest_nodes(x_nodes, y_nodes, x0, deg)

    # Lagrange
    T_lagrange = lagrange_eval(xi, yi, x0)
    poly_lagrange = lagrange_poly_sym(xi, yi)

    # Newton
    coeffs, xi_new = newton_coeffs(xi, yi)
    T_newton = newton_eval(coeffs, xi_new, x0)
    poly_newton = newton_poly_sym(coeffs, xi_new)

    # Absolute relative error Δ
    delta_L = abs(T_lagrange - prev_T_L)if prev_T_L is not None else 0
    delta_N = abs(T_newton - prev_T_N) if prev_T_N is not None else 0

    prev_T_L, prev_T_N = T_lagrange, T_newton

    results.append((deg, T_lagrange, T_newton, delta_L, delta_N, xi.tolist(), yi.tolist(), poly_lagrange, poly_newton))

# -----------------------------
# Display table
# -----------------------------
print("Predicted temperatures at x = 45 km with  Δ and points used:")
for r in results:
    deg, TL, TN, dL, dN, x_used, y_used, pL, pN = r
    print(f"\nDegree {deg}:")
    print(f"  Points used: {list(zip(x_used, y_used))}")
    print(f"  Lagrange T(45) = {TL:.4f}, Δ = {dL:.4e}")
    print(f"  Newton T(45)   = {TN:.4f}, Δ = {dN:.4e}")
    print(f"  Lagrange Poly: {pL}")
    print(f"  Newton Poly:   {pN}")

# -----------------------------
# Δ vs Degree graph
# -----------------------------
deg_list = [r[0] for r in results]
delta_L_list = [r[3] for r in results]
delta_N_list = [r[4] for r in results]

plt.figure(figsize=(8,5))
plt.plot(deg_list, delta_L_list, 'o-', label='Lagrange Δ', color='blue')
plt.plot(deg_list, delta_N_list, 's--', label='Newton Δ', color='green')
plt.title(" Δ vs Polynomial Degree at x=45")
plt.xlabel("Polynomial Degree")
plt.ylabel("Δ ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
