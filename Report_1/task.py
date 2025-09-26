import numpy as np
import matplotlib.pyplot as plt

def f(h):
    return h**3 - 10*h + 5*np.exp(-h/2) - 2

def df(h):
    return 3*h**2 - 10 - (5/2)*np.exp(-h/2)

# ---------------- Bisection ----------------
def Bisection(a, b, eps_relative):
    print("\n=== Bisection Method ===")
    print(f"{'Iter':<6} {'xl':<10} {'xu':<10} {'xr':<10} {'f(xr)':<12} {'ea(%)':<10}")
    print("-"*60)

    iteration_data = []
    xr_old = a
    k = 0
    
    while True:
        k += 1
        xr = (a + b) / 2
        fxr = f(xr)
        ea = abs((xr - xr_old)/xr) * 100
        iteration_data.append([k, ea])
        
        print(f"{k:<6d} {a:.5f}   {b:.5f}   {xr:.5f}   {fxr:.5f}   {ea:.5f}")
        
        if f(xr) == 0 or ea < eps_relative:
            break
        if f(a) * fxr < 0:
            b = xr
        else:
            a = xr
        xr_old = xr
    return [row[0] for row in iteration_data], [row[1] for row in iteration_data]

# ---------------- False Position ----------------
def FalsePosition(a, b, eps_relative):
    print("\n=== False Position Method ===")
    print(f"{'Iter':<6} {'xl':<10} {'xu':<10} {'xr':<10} {'f(xr)':<12} {'ea(%)':<10}")
    print("-"*60)

    iteration_data = []
    xr_old = b
    k = 0
    while True:
        k += 1
        fa, fb = f(a), f(b)
        xr = (a*fb - b*fa) / (fb - fa)
        fxr = f(xr)
        ea = abs((xr - xr_old)/xr) * 100 if k > 1 else None
        if ea is not None:
            iteration_data.append([k, ea])
        
        ea_str = f"{ea:.5f}" if ea is not None else "N/A"
        print(f"{k:<6d} {a:.5f}   {b:.5f}   {xr:.5f}   {fxr:.5f}   {ea_str}")
        
        if ea is not None and ea < eps_relative:
            break
        if fa * fxr < 0:
            b = xr
        else:
            a = xr
        xr_old = xr
    return [row[0] for row in iteration_data], [row[1] for row in iteration_data]

# ---------------- Newton-Raphson ----------------
def newton_raphson(x0, tol=0.001, max_iter=100):
    print("\n=== Newton-Raphson Method ===")
    print("{:<6} {:<10} {:<12} {:<12} {:<10}".format("Iter", "x", "f(x)", "f'(x)", "ea(%)"))
    print("-"*60)

    ea_list, iter_list = [], []
    x = x0
    for i in range(1, max_iter+1):
        fx, dfx = f(x), df(x)
        if dfx == 0:
            break
        x_new = x - fx/dfx
        ea = abs((x_new - x)/x_new) * 100
        ea_list.append(ea)
        iter_list.append(i)
        
        print(f"{i:<6d} {x_new:.5f}   {fx:.5f}   {dfx:.5f}   {ea:.5f}")
        
        if ea < tol:
            break
        x = x_new
    return iter_list, ea_list

# ---------------- Secant ----------------
def secant_method(x0, x1, tol=0.001, max_iter=100):
    print("\n=== Secant Method ===")
    print(f"{'Iter':<6} {'x0':<10} {'x1':<10} {'x2':<10} {'f(x2)':<12} {'ea(%)':<10}")
    print("-"*70)

    ea_list, iter_list = [], []
    for i in range(1, max_iter+1):
        f0, f1 = f(x0), f(x1)
        denom = f1 - f0
        if denom == 0:
            break
        x2 = x1 - f1*(x1 - x0)/denom
        fx2 = f(x2)
        ea = abs((x2 - x1)/x2) * 100
        ea_list.append(ea)
        iter_list.append(i)
        
        print(f"{i:<6d} {x0:.5f}   {x1:.5f}   {x2:.5f}   {fx2:.5f}   {ea:.5f}")
        
        if ea < tol:
            break
        x0, x1 = x1, x2
    return iter_list, ea_list

# ---------------- Run all methods ----------------
iter_bis, err_bis = Bisection(0.1, 0.4, 0.001)
iter_fp, err_fp = FalsePosition(0.1, 0.4, 0.001)
iter_nr, err_nr = newton_raphson(1.5, 0.001)
iter_sec, err_sec = secant_method(1.5, 2.0, 0.001)

# ---------------- Single Plot ----------------
plt.figure(figsize=(10,6))
plt.plot(iter_bis, err_bis, marker='o', label="Bisection")
plt.plot(iter_fp, err_fp, marker='s', label="False Position")
plt.plot(iter_nr, err_nr, marker='^', label="Newton-Raphson")
plt.plot(iter_sec, err_sec, marker='d', label="Secant")

plt.xlabel("Iterations")
plt.ylabel("Approx. Relative Error (%)")
plt.title("Comparison of Root-Finding Methods")
plt.yscale("log")  # log scale for better visualization
plt.legend()
plt.grid(True)
plt.show()
