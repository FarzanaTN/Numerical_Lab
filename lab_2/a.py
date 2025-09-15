import numpy as np
import matplotlib.pyplot as plt

# Function definition
def f(x):
    return x**3 - 6*x**2 + 11*x -6.1

    return x**10 -1
def isFalsePositionApplicable(a, b):
    return f(a) * f(b) < 0

def FalsePosition(a, b, eps_relative):
    if not isFalsePositionApplicable(a, b):
        print("f(a) * f(b) >= 0. No root guaranteed in [a, b].")
        return None
    
    iteration_data = []
    xr_old = b
    #xr_old = a
    k = 0
    
    while True:
        k += 1
        # False position formula
        fa = f(a)
        fb = f(b)
        xr = ((a*fb) - (b*fa)) / (fb - fa)
        #xr = b - fb * (a - b) / (fa - fb)
        fxr = f(xr)
        
        # Approximate relative error
        ea = abs((xr - xr_old)/xr) * 100 if k > 1 else None
        
        # Save iteration data
        iteration_data.append([k, a, b, xr, fxr, ea])
        
        # Check stopping criterion
        if ea is not None and ea < eps_relative:
            break
        
        # Update bounds
        if fa * fxr < 0:
            b = xr
        else:
            a = xr
        
        xr_old = xr
    
    # Print table header
    print(f"{'Iter':>4} | {'xl':>12} | {'xu':>12} | {'xr':>12} | {'f(xr)':>15} |  {'εa(%)':>12}")
    print("-"*90)
    
    for row in iteration_data:
        k, xl, xu, xr, fxr, ea = row
        ea_str = f"{ea:.10f}" if ea is not None else "N/A"
        print(f"{k:4d} | {xl:12.10f} | {xu:12.10f} | {xr:12.10f} | {fxr:15.10f} | {ea_str:>12}")
    
    # Plot absolute error vs iterations
    abs_errors = [row[5] if row[5] is not None else 0 for row in iteration_data]
    iterations = [row[0] for row in iteration_data]
    
    plt.figure(figsize=(8,5))
    plt.plot(iterations, abs_errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Approximate Relative Error (%)')
    plt.title('Convergence of False Position Method')
    plt.grid(True)
    plt.show()
    
    #return iteration_data

    
    return xr

# Run the False Position method with initial guesses and relative error < 0.05%
root = FalsePosition(1, 4, 0.001)
print(f"Root: {root}")
