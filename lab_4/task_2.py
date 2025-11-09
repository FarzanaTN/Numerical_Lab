import numpy as np

x_data = np.array([0, 1, 2.5, 3, 4.5, 5, 6])
y_data = np.array([2.00000, 5.43750, 7.35160, 7.56250, 8.44530, 9.18750, 12.00000])

target_x = 3.5  

distances = np.abs(x_data - target_x)
sorted_indices = np.argsort(distances)

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

Nk_values = []
degrees = [2, 3, 4] 
for k in degrees:
    node_indices = sorted_indices[:k+1]
    x_nodes = x_data[node_indices]
    y_nodes = y_data[node_indices]
    
    table = divided_diff_table(x_nodes, y_nodes)
    
    # print(f"\nDivided difference table for degree {k}:")
    # print(np.round(table, 8))
    
    Nk = newton_poly(x_nodes, table, target_x)
    Nk_values.append(Nk)

print("\nNewton polynomial evaluations at x=3.5:")
for i, Nk in enumerate(Nk_values):
    delta = abs(Nk - Nk_values[i-1]) if i>0 else 0
    print(f"N{degrees[i]}(3.5) = {Nk:.6f}, delta = {delta:.6f}")
