import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Exact solution
# ------------------------------
def y_exact(t):
    return np.exp(-np.cos(t) + (np.cos(t)**3)/3 + 2/3)


# ODE dy/dt = y sin^3(t)
def f(t, y):
    return y * (np.sin(t)**3)


def euler(f, t0, y0, h, t_end):
    N = int((t_end - t0) / h)
    t = np.linspace(t0, t_end, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        y[n+1] = y[n] + h * f(t[n], y[n])
    return t, y


def heun(f, t0, y0, h, t_end):
    N = int((t_end - t0) / h)
    t = np.linspace(t0, t_end, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        y_pred = y[n] + h * f(t[n], y[n])
        y[n+1] = y[n] + (h/2) * (f(t[n], y[n]) + f(t[n+1], y_pred))
    return t, y

# ------------------------------
# Step sizes
# ------------------------------
h_values = [0.1, 0.005, 0.001, 0.0005, 0.0001]

tables = {}

# # ------------------------------
# # Compute tables + plots
# # ------------------------------
# for h in h_values:
#     t_eu, y_eu = euler(f, 0, 1, h, 5)
#     t_he, y_he = heun(f, 0, 1, h, 5)
#     exact_vals = y_exact(t_eu)

#     # Complete table
#     df = pd.DataFrame({
#         "t_n": t_eu,
#         "Euler y_n": y_eu,
#         "Heun y_n": y_he,
#         "Exact y(t_n)": exact_vals,
#         "Euler Error": np.abs(y_eu - exact_vals),
#         "Heun Error": np.abs(y_he - exact_vals)
#     })

#     tables[h] = df

#     # ------------------------------
#     # Print full table for this h
#     # ------------------------------
#     print(f"\n\n========== TABLE FOR h = {h} ==========\n")
#     print(df)

#     # ------------------------------
#     # Plot for this h
#     # ------------------------------
#     plt.figure(figsize=(10, 6))
#     plt.plot(t_eu, exact_vals, label="Exact", linewidth=2)
#     plt.plot(t_eu, y_eu, '--', label=f"Euler (h={h})")
#     plt.plot(t_eu, y_he, '-.', label=f"Heun (h={h})")

#     plt.title(f"Exact vs Euler vs Heun (h = {h})", fontsize=14)
#     plt.xlabel("t")
#     plt.ylabel("y")
#     plt.grid(True)
#     plt.legend()
#     plt.show()


# ------------------------------
# Compute tables + plots
# ------------------------------
for h in h_values:
    t_eu, y_eu = euler(f, 0, 1, h, 5)
    t_he, y_he = heun(f, 0, 1, h, 5)
    exact_vals = y_exact(t_eu)

    # Complete table
    df = pd.DataFrame({
        "t_n": t_eu,
        "Euler y_n": y_eu,
        "Heun y_n": y_he,
        "Exact y(t_n)": exact_vals,
        "Euler Error": np.abs(y_eu - exact_vals),
        "Heun Error": np.abs(y_he - exact_vals)
    })

    tables[h] = df

    # ------------------------------
    # Print full table for this h
    # ------------------------------
    print(f"\n\n========== TABLE FOR h = {h} ==========\n")
    print(df)


# ------------------------------
# Compute tables + plot all in one figure
# ------------------------------
plt.figure(figsize=(12, 8))
t_fine = np.linspace(0, 5, 5000)
plt.plot(t_fine, y_exact(t_fine), 'k', linewidth=2, label="Exact")  # Exact solution

for h in h_values:
    t_eu, y_eu = euler(f, 0, 1, h, 5)
    t_he, y_he = heun(f, 0, 1, h, 5)
    exact_vals = y_exact(t_eu)

    # Store table
    df = pd.DataFrame({
        "t_n": t_eu,
        "Euler y_n": y_eu,
        "Heun y_n": y_he,
        "Exact y(t_n)": exact_vals,
        "Euler Error": np.abs(y_eu - exact_vals),
        "Heun Error": np.abs(y_he - exact_vals)
    })
    tables[h] = df

    # Plot Euler and Heun for this h
    plt.plot(t_eu, y_eu, '--', label=f"Euler h={h}")
    plt.plot(t_he, y_he, '-.', label=f"Heun h={h}")

plt.title("Exact vs Euler vs Heun for Various Step Sizes", fontsize=14)
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
