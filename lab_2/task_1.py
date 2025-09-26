import numpy as np
import matplotlib.pyplot as plt

g = 9.81      
c = 15        
t = 10       
y_target = 36 

def f(m):
    return (g * m / c) * (1 - np.exp(-c*t/m )) - y_target

def isFalsePositionApplicable(a, b):
    return f(a) * f(b) < 0

def FalsePosition(a, b, eps_relative):
    if not isFalsePositionApplicable(a, b):
        print("f(a) * f(b) >= 0. No root guaranteed in [a, b].")
        return None
    
    iteration_data = []
    xr_old = b
    k = 0
    
    while True:
        k += 1
        fa = f(a)
        fb = f(b)
        xr = (a*fb - b*fa) / (fb - fa)
        fxr = f(xr)
        
        ea = abs((xr - xr_old)/xr) * 100 if k > 1 else None
        
        iteration_data.append([k, a, b, xr, fxr, ea])
        
        if ea is not None and ea < eps_relative:
            break
        
        if fa * fxr < 0:
            b = xr
        else:
            a = xr
        
        xr_old = xr
    
    # Print iteration table
    print(f"{'Iter':>4} | {'xl':>12} | {'xu':>12} | {'xr':>12} | {'f(xr)':>15} | {'εa(%)':>12}")
    print("-"*90)
    
    for row in iteration_data:
        k, xl, xu, xr, fxr, ea = row
        ea_str = f"{ea:.10f}" if ea is not None else "N/A"
        print(f"{k:4d} | {xl:12.10f} | {xu:12.10f} | {xr:12.10f} | {fxr:15.10f} | {ea_str:>12}")
    
    # Plot approximate relative error vs iterations
    abs_errors = [row[5] if row[5] is not None else 0 for row in iteration_data][1:]
    iterations = [row[0] for row in iteration_data][1:]
    
    plt.figure(figsize=(8,5))
    plt.plot(iterations, abs_errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Approximate Relative Error (%)')
    plt.title('Convergence of False Position Method')
    plt.grid(True)
    plt.show()
    
    return xr

xl = 40.0 
xu = 80.0  
es = 0.001 
root = FalsePosition(xl, xu, es)
print(f"\nApproximate root (mass m): {root:.10f} kg")
