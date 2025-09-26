import numpy as np
import matplotlib.pyplot as plt

g = 9.81      
c = 15        
t = 10        
y_target = 36 

def f(m):
    return (g*m/c) * (1 - np.exp(-c*t/m)) - y_target

def df(m):
    return (g/c) * (1 - np.exp(-c*t/m)) + (g*t/m) * np.exp(-c*t/m)

def newton_raphson( x0, tol=0.001, max_iter=100):
    """
    Newton-Raphson method with iteration details printed as a table.
    """
    print("{:<6} | {:<12} | {:<15} | {:<15} | {:<12}".format(
        "Iter", "mk", "f(mk)", "f'(mk)", "εa(%)"))
    print("-"*70)
    
    x = x0
    ea_list = []
    iter_list = []

    for i in range(1, max_iter+1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivative zero. No convergence.")
        
        x_new = x - fx/dfx
        ea = abs((x_new - x)/x_new) * 100
        
        print("{:<6d} | {:<12.10f} | {:<15.10f} | {:<15.10f} | {:<12.10f}".format(
            i, x_new, fx, dfx, ea))
        
        ea_list.append(ea)
        iter_list.append(i)
        
        if ea < tol:
            print(f"\nConverged to root: {x_new:.10f} kg after {i} iterations")
            break
        
        x = x_new
        
        # if ea < tol:
        #     break


    # Plot error vs iterations
    plt.figure(figsize=(8,5))
    plt.plot(iter_list, ea_list, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Approx. Relative Error (%)")
    plt.title("Newton-Raphson: Error vs Iteration")
    plt.grid(True)
    plt.show()
    
    return x_new

m0 = 40.0  
root = newton_raphson( x0=m0, tol=0.001)
print(f"Approximate mass m = {root:.10f} kg")
