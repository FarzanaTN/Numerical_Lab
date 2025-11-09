import numpy as np

# Given data
x_data = np.array([0, 1, 2.5, 3, 4.5, 5, 6], dtype=float)
y_data = np.array([2.0, 5.43750, 7.35160, 7.56250, 8.44530, 9.18750, 12.0], dtype=float)

# For local accuracy near x = 3.5, choose nodes closest to 3.5
# Order by distance from 3.5: 3, 4.5, 2.5, 5
nodes_indices = [3, 4, 2, 5]
x_nodes = x_data[nodes_indices]
y_nodes = y_data[nodes_indices]

def divided_diff_table(x_nodes, y_nodes):
    n = len(x_nodes)
    table = np.zeros((n, n))
    table[:,0] = y_nodes
    for j in range(1, n):
        for i in range(n-j):
            table[i,j] = (table[i+1,j-1] - table[i,j-1]) / (x_nodes[i+j] - x_nodes[i])
    return table

def newton_poly(x_nodes, table, x):
    n = len(x_nodes)
    P = table[0,0]
    prod = 1.0
    for j in range(1, n):
        prod *= (x - x_nodes[j-1])
        P += table[0,j] * prod
    return P

# Compute divided difference table
table = divided_diff_table(x_nodes, y_nodes)
print("Divided difference table:")
print(np.round(table, 8))

# Evaluate Newton polynomial at x = 3.5 for k=2,3,4
Nk_values = []
for k in [2,3,4]:
    Nk = newton_poly(x_nodes[:k+1], table, 3.5)
    Nk_values.append(Nk)
print("\nNewton polynomial evaluations at x=3.5:")
for i, Nk in enumerate(Nk_values):
    delta = abs(Nk - Nk_values[i-1]) if i>0 else 0
    print(f"N{i+2}(3.5) = {Nk:.6f}, delta = {delta:.6f}")
