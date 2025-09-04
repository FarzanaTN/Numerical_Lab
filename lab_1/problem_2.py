import numpy as np
from math import log2, ceil
import matplotlib.pyplot as plt

Q = 20.0
g = 9.81

# Function definition
def f(y):
    A = 3*y + (y**2)/2.0
    B = 3 + y
    return 1 - (Q**2) / (g * (A**3) * B)

def isBisectionApplicable(a, b):
    return f(a) * f(b) < 0

def FindStep(a, b, eps):
    L0 = b - a
    return ceil(log2(L0 / eps))

def Bisection(a, b, eps_relative):
    if not isBisectionApplicable(a, b):
        print("f(a) * f(b) >= 0. No root guaranteed in [a, b].")
        return None
    
    # # True root (for error calculation)
    # # Since we don't know the exact root, approximate it by a very small tolerance
    # true_root = BisectionApproxRoot(a, b, 1e-12)
    
    iteration_data = []
    xr_old = a
    k = 0
    step = FindStep(a, b, eps_relative)
    
    while k <= step:
        k += 1
        xr = (a + b) / 2
        fxr = f(xr)
        
       
        ea = abs((xr - xr_old)/xr) * 100
        # else:
        #     ea = None
        
        # True error (%)
        #et = abs((true_root - xr)/true_root) * 100
        
        # Save iteration data
        iteration_data.append([k, a, b, xr, fxr, ea])
        
        # Check stopping criterion (approximate relative error below threshold)
        # if ea is not None and ea < eps_relative:
        #     break
        if f(xr) == 0 or abs(b-a) < eps_relative:
            break
        
        # Update bounds
        if f(a) * fxr < 0:
            b = xr
        else:
            a = xr
        
        xr_old = xr
    
    print(f"{'Iter':>4} | {'xl':>12} | {'xu':>12} | {'xr':>12} | {'f(xr)':>15} |  {'εa(%)':>12}")
    print("-"*90)
    
    for row in iteration_data:
        k, xl, xu, xr, fxr,  ea = row
        print(f"{k:4d} | {xl:12.10f} | {xu:12.10f} | {xr:12.10f} | {fxr:15.10f} | {ea }")
    
    # Plot absolute error vs iterations
    abs_errors = [row[5] if row[5] is not None else 0 for row in iteration_data]
    iterations = [row[0] for row in iteration_data]
    
    plt.figure(figsize=(8,5))
    plt.plot(iterations, abs_errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Approximate Relative Error (%)')
    plt.title('Convergence of Bisection Method')
    plt.grid(True)
    plt.show()
    
    return xr

# # Helper function to get a very accurate approximate root for error calculation
# def BisectionApproxRoot(a, b, tol):
#     while (b - a)/2 > tol:
#         c = (a + b)/2
#         if f(a) * f(c) < 0:
#             b = c
#         else:
#             a = c
#     return (a + b)/2

# Run the bisection method with initial guesses and relative error < 0.05%
root = Bisection(0.5, 2.5, 0.01)
#root = Bisection (0, 1.3, 0.02)


