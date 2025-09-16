import matplotlib.pyplot as plt
def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Secant method printing:
      Iter | x_{i-1} | x_i | x_{i+1} | |e_a|% ( = |(x_i - x_{i-1})/x_i|*100 ) | f(x_{i+1})
    Error column is '---' for the first row (no previous pair).
    Stopping criterion: |e_a| < tol*100 (because e_a is in percent).
    """
    print("{:<6} {:<13} {:<13} {:<13} {:<13} {:<15}".format(
        "Iter", "x_{i-1}", "x_i", "x_{i+1}", "|e_a| (%)", "f(x_{i+1})"))
    print("-" * 75)
    
    ea_list = []   # store errors
    iter_list = [] # store iteration numbers


    for i in range(1, max_iter + 1):
        f0 = f(x0)
        f1 = f(x1)

        denom = (f1 - f0)
        if denom == 0:
            raise ZeroDivisionError("Zero denominator encountered in secant formula.")

        x2 = x1 - f1 * (x1 - x0) / denom
        fx2 = f(x2)

        # error computed between current middle (x1) and previous (x0):
        if i == 1:
            ea_str = "---"
        else:
            if x1 != 0:
                ea = abs((x1 - x0) / x1) * 100.0
                ea_str = f"{ea:.5f}"
                ea_list.append(ea)
                iter_list.append(i)
            else:
                ea = float("inf")
                ea_str = "inf"

        print("{:<6d} {:<13.8f} {:<13.8f} {:<13.8f} {:<13} {:<15.5e}".format(
            i, x0, x1, x2, ea_str, fx2))

        # stopping: only meaningful from iteration 2 onward (ea computed)
        if i >= 2 and x1 != 0 and ea < tol * 100.0:
            print("\nRoot found (approx): {:.8f} after {} iterations".format(x2, i))
            # plot error vs iteration
             # ✅ FIX: use ea_list instead of ea
            plt.plot(iter_list, ea_list, marker='o')
            plt.xlabel("Iteration")
            plt.ylabel("Approx. Relative Error (%)")
            plt.title("Secant Method: Error vs Iteration")
            plt.grid(True)
            plt.show()
            return x2

        # shift for next iteration
        x0, x1 = x1, x2
    
        
    plt.plot(iter_list, ea_list, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Approx. Relative Error (%)")
    plt.title("Secant Method: Error vs Iteration")
    plt.grid(True)
    plt.show()

    raise ValueError("Did not converge within the maximum number of iterations.")


# Define the function
f = lambda x: x**4 - 5*x - 2
    #x**3 - 0.165*x**2 + 3.993e-4

# Run using the initial guesses you used before
secant_method(f, x0=3, x1=5, tol=1e-6, max_iter=50)
