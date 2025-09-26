import numpy as np
import matplotlib.pyplot as plt

g = 9.81
c = 15
t = 10
y_target = 36

def f(m):
    return (g*m/c) * (1 - np.exp(-c*t/m)) - y_target

def secant_method( x0, x1, tol=0.001, max_iter=100):
    """
    Secant method for solving f(m) = 0
    """
    print("{:<6} {:<13} {:<13} {:<13} {:<13} {:<15}".format(
        "Iter", "x_{i-1}", "x_i", "x_{i+1}", "|e_a| (%)", "f(x_{i+1})"))
    print("-" * 75)
    
    ea_list = []
    iter_list = []
    ea = float('inf')  

    for i in range(1, max_iter + 1):
        f0 = f(x0)
        f1 = f(x1)
        denom = f1 - f0
        if denom == 0:
            raise ZeroDivisionError("Zero denominator in Secant formula.")
        
        x2 = x1 - f1 * (x1 - x0) / denom
        fx2 = f(x2)
        
        if i == 1:
            ea_str = "---"
        else:
            ea = abs((x1 - x0)/x1) * 100
            ea_str = f"{ea:.10f}"
            ea_list.append(ea)
            iter_list.append(i)
        
        print("{:<6d} {:<13.10f} {:<13.10f} {:<13.10f} {:<13} {:<15.10f}".format(
            i, x0, x1, x2, ea_str, fx2))
        
        # if i >= 2 and ea < tol:
        #     print(f"\nRoot found (approx): {x2:.10f} kg after {i} iterations")
        #     plt.figure(figsize=(8,5))
        #     plt.plot(iter_list, ea_list, marker='o')
        #     plt.xlabel("Iteration")
        #     plt.ylabel("Approx. Relative Error (%)")
        #     plt.title("Secant Method: Error vs Iteration")
        #     plt.grid(True)
        #     plt.show()
        #     return x2
        
        x0, x1 = x1, x2
        
        if ea < tol:
            print(f"\nRoot found (approx): {x2:.10f} kg after {i} iterations")
            break
        
        
    
    plt.figure(figsize=(8,5))
    plt.plot(iter_list, ea_list, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Approx. Relative Error (%)")
    plt.title("Secant Method: Error vs Iteration")
    plt.grid(True)
    plt.show()
    
    return x2
    
    #raise ValueError("Did not converge within the maximum number of iterations.")

root = secant_method( x0=30.0, x1=40.0, tol=0.001, max_iter=50)
print(f"\nApproximate mass m = {root:.10f} kg")
