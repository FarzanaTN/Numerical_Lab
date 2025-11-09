import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([0, 1, 2.5, 3, 4.5, 5, 6], dtype=float)
y_data = np.array([2.0, 5.43750, 7.35160, 7.56250, 8.44530, 9.18750, 12.0], dtype=float)

#to calculate L
def lagrange_basis(x_nodes, j, x):
    L = 1.0
    for m, xm in enumerate(x_nodes):
        if m != j:
            L *= (x - xm) / (x_nodes[j] - xm)
    return L

def lagrange_poly(x_nodes, y_nodes, x):
    n = len(x_nodes)
    P = 0
    for j in range(n):
        P += y_nodes[j] * lagrange_basis(x_nodes, j, x)
    return P

# Task 1(a) Quadratic interpolant near x = 3.5
# Pick 3 nodes closest to x=3.5: x = [3, 2.5, 4.5]
nodes_indices_quad = [2, 3, 4]
x_quad = x_data[nodes_indices_quad]
y_quad = y_data[nodes_indices_quad]

P2_3_5 = lagrange_poly(x_quad, y_quad, 3.5)
print("P2(3.5) =", P2_3_5)

# Task 1(b) Cubic interpolant near x=3.5
# Pick 4 nodes closest to x=3.5: x = [2.5, 3, 4.5, 5]
nodes_indices_cubic = [2, 3, 4, 5]
x_cubic = x_data[nodes_indices_cubic]
y_cubic = y_data[nodes_indices_cubic]

P3_3_5 = lagrange_poly(x_cubic, y_cubic, 3.5)
print("P3(3.5) =", P3_3_5)

# Optional: Higher-degree interpolants using all points
degrees = [4, 5, 6]  # P4, P5, P6 using first n+1 points
Pk_values = [P2_3_5, P3_3_5]
for n in degrees:
    x_n = x_data[:n+1]
    y_n = y_data[:n+1]
    P_val = lagrange_poly(x_n, y_n, 3.5)
    Pk_values.append(P_val)

#calculate delta
deltas = [Pk_values[i+1] - Pk_values[i] for i in range(len(Pk_values)-1)]

print("\nDegree\tP(3.5)\t\tDelta")
for i, P_val in enumerate(Pk_values):
    delta = deltas[i-1] if i > 0 else 0
    print(f"{i+2}\t{P_val:.6f}\t{delta:.6f}")
    


x_plot = np.linspace(min(x_data), max(x_data), 200)
plt.figure(figsize=(8,5))

x_nodes_list = [x_quad, x_cubic, x_data[:5], x_data[:6], x_data[:7]]  # P2, P3, P4, P5, P6
y_nodes_list = [y_quad, y_cubic, y_data[:5], y_data[:6], y_data[:7]]
labels = ['P2', 'P3', 'P4', 'P5', 'P6']

for label, xn, yn in zip(labels, x_nodes_list, y_nodes_list):
    y_plot = np.array([lagrange_poly(xn, yn, xi) for xi in x_plot])
    plt.plot(x_plot, y_plot, label=label)

plt.scatter(x_data, y_data, color='black', zorder=5, label='Data points')

plt.axvline(3.5, color='red', linestyle='--', label='x=3.5')

plt.xlabel("x")
plt.ylabel("P(x)")
plt.title("Lagrange Interpolants: P2–P6")
plt.legend()
plt.grid(True)
plt.show()
