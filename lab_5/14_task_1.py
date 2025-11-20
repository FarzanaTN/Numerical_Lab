import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

# ---------------------------
# Settings
# ---------------------------
h_values = [0.5, 0.25, 0.1, 0.01, 0.001]
t0, t_end = 0.0, 5.0
y0 = 1.0

OUT_DIR = "ivp_results"
os.makedirs(OUT_DIR, exist_ok=True)

# If True: prints full tables (can be very long for small h).
# Default False to avoid flooding console; you can set True if you want.
PRINT_ALL_TABLES = False

# ---------------------------
# Exact solution and ODE
# ---------------------------
def y_exact(t):
    return 4.0*t - 3.0 + 4.0 * np.exp(-t)

def f(t, y):
    # dy/dt = (1 + 4t) - y
    return (1.0 + 4.0*t) - y


def euler(f, t0, y0, h, t_end):
    N = int(round((t_end - t0) / h))
    t = np.linspace(t0, t_end, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        y[n+1] = y[n] + h * f(t[n], y[n])
    return t, y

def heun(f, t0, y0, h, t_end):
    N = int(round((t_end - t0) / h))
    t = np.linspace(t0, t_end, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        k1 = f(t[n], y[n])
        y_pred = y[n] + h * k1                   # Euler predictor
        k2 = f(t[n+1], y_pred)                  # slope at predicted point
        y[n+1] = y[n] + (h/2.0) * (k1 + k2)     # Heun corrector
    return t, y

# ---------------------------
# Containers
# ---------------------------
tables = {}
max_errors = []

# ---------------------------
# Compute for each h
# ---------------------------
for h in h_values:
    t_eu, y_eu = euler(f, t0, y0, h, t_end)
    t_he, y_he = heun(f, t0, y0, h, t_end)
    # t_eu and t_he are identical grids; use t_eu
    exact_vals = y_exact(t_eu)

    df = pd.DataFrame({
        "t_n": t_eu,
        "Euler y_n": y_eu,
        "Heun y_n": y_he,
        "Exact y(t_n)": exact_vals,
        "Euler Error": np.abs(y_eu - exact_vals),
        "Heun Error": np.abs(y_he - exact_vals)
    })

    tables[h] = df

    # Save CSV
    csv_name = os.path.join(OUT_DIR, f"table_h_{str(h).replace('.', 'p')}.csv")
    df.to_csv(csv_name, index=False)

    # Print table header + first rows (or entire table if opted)
    print("\n" + "="*60)
    print(f"TABLE for h = {h}  (saved -> {csv_name})")
    print("="*60)
    if PRINT_ALL_TABLES:
        pd.set_option('display.max_rows', None)
        print(df)
        pd.reset_option('display.max_rows')
    else:
        print(df.head(20).to_string(index=False))
        if len(df) > 20:
            print(f"... (only first 20 rows shown; total rows = {len(df)}). Set PRINT_ALL_TABLES=True to print all rows.)")

    # gather max errors for convergence plot
    max_euler_err = df["Euler Error"].max()
    max_heun_err = df["Heun Error"].max()
    max_errors.append((h, max_euler_err, max_heun_err))

# ---------------------------
# Combined plot: exact + all Euler & Heun approximations
# ---------------------------
# plt.figure(figsize=(12, 7))

# # Dense exact curve for smooth plotting
# t_dense = np.linspace(t0, t_end, 2000)
# plt.plot(t_dense, y_exact(t_dense), color='k', linewidth=1.2, label="Exact y(t)")

# # Plot Euler and Heun for each h with distinct styles
# markers = ['o', 's', 'v', '^', 'x']
# linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1,1))]
# for i, h in enumerate(h_values):
#     df = tables[h]
#     t = df["t_n"].values
#     y_e = df["Euler y_n"].values
#     y_hn = df["Heun y_n"].values

#     # Euler
#     plt.plot(t, y_e, linestyle=linestyles[i % len(linestyles)],
#              marker=markers[i % len(markers)], markersize=4,
#              label=f"Euler h={h}")

#     # Heun
#     plt.plot(t, y_hn, linestyle=linestyles[(i+1) % len(linestyles)],
#              marker=markers[(i+1) % len(markers)], markersize=3,
#              label=f"Heun h={h}")

# plt.title("Exact vs Euler vs Heun (all step sizes)", fontsize=14)
# plt.xlabel("t")
# plt.ylabel("y(t)")
# plt.grid(True)
# plt.legend(ncol=2, fontsize=9)
# plt.tight_layout()
# png_all = os.path.join(OUT_DIR, "all_solutions.png")
# plt.savefig(png_all, dpi=200)
# # print(f"\nSaved combined solution plot -> {png_all}")
# plt.show()

# # ---------------------------
# # Convergence / error summary
# # ---------------------------
# # Prepare arrays for log-log plot
# hs = np.array([item[0] for item in max_errors])
# euler_max_errs = np.array([item[1] for item in max_errors])
# heun_max_errs = np.array([item[2] for item in max_errors])

# # Print error table
# # print("\n" + "="*40)
# # print("Max absolute errors over [0,5] (for each h)")
# # print("="*40)
# # err_df = pd.DataFrame({
# #     "h": hs,
# #     "Max Euler Error": euler_max_errs,
# #     "Max Heun Error": heun_max_errs
# # })
# # print(err_df.to_string(index=False))

# # # Log-log plot of max error vs h
# # plt.figure(figsize=(7,5))
# # plt.loglog(hs, euler_max_errs, 'o-', label='Euler max error')
# # plt.loglog(hs, heun_max_errs, 's-', label='Heun max error')
# # plt.xlabel("step size h")
# # plt.ylabel("max absolute error on [0,5]")
# # plt.title("Convergence: max error vs h (log-log)")
# # plt.grid(True, which='both', ls='--')
# # plt.legend()
# # png_err = os.path.join(OUT_DIR, "max_error_vs_h.png")
# # plt.savefig(png_err, dpi=200)
# # print(f"Saved error plot -> {png_err}")
# # plt.show()

# # ---------------------------
# # Done
# # ---------------------------
# # print("\nAll tables saved in folder:", OUT_DIR)
# # print("If you want EVERY table printed to console, set PRINT_ALL_TABLES = True at the top of the script.")

plt.figure(figsize=(12, 7))

# Dense exact curve for smooth plotting
t_dense = np.linspace(t0, t_end, 2000)
plt.plot(t_dense, y_exact(t_dense), color='k', linewidth=2.0, label="Exact y(t)")

# Plot Euler and Heun for each h with same line thickness
markers = ['o', 's', 'v', '^', 'x']
linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1,1))]
for i, h in enumerate(h_values):
    df = tables[h]
    t = df["t_n"].values
    y_e = df["Euler y_n"].values
    y_hn = df["Heun y_n"].values

    step_marker = max(1, len(t)//20)  # markers only at some points

    # Euler
    plt.plot(t, y_e, linestyle=linestyles[i % len(linestyles)],
             marker=markers[i % len(markers)],
             markevery=step_marker,
             markersize=4, linewidth=2.0,  # same thickness as exact
             label=f"Euler h={h}")

    # Heun
    plt.plot(t, y_hn, linestyle=linestyles[(i+1) % len(linestyles)],
             marker=markers[(i+1) % len(markers)],
             markevery=step_marker,
             markersize=3, linewidth=2.0,  # same thickness
             label=f"Heun h={h}")

plt.title("Exact vs Euler vs Heun (all step sizes)", fontsize=14)
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(True)
plt.legend(ncol=2, fontsize=9)
# plt.xlim(0, 2)

plt.tight_layout()
png_all = os.path.join(OUT_DIR, "all_solutions_thick.png")
plt.savefig(png_all, dpi=200)
plt.show()
