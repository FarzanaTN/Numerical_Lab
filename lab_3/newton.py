from matplotlib import pyplot as plt


def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson method with iteration details printed as a table.
    Iteration 0 shows the initial guess. Error is calculated starting from iteration 1.
    """
    print("{:<10} {:<15} {:<15} {:<15}".format("Iter", "xi", "f(xi)", "Abs Error"))
    print("-" * 60)

    x = x0
    print("{:<10} {:<15.8f} {:<15.8f} {:<15}".format(0, x, f(x), "---"))  # iteration 0


    ea_list = []   # store errors
    iter_list = [] # store iteration numbers
    
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            raise ValueError("Derivative is zero. No convergence.")

        x_new = x - fx / dfx
        error = abs((x_new - x) / x_new) * 100

        print("{:<10} {:<15.8f} {:<15.8f} {:<15}".format(i, x_new, f(x_new), error))

        ea_list.append(error)
        iter_list.append(i)
        
        if error < tol:
            print("\nRoot found: {:.8f} after {} iterations".format(x_new, i))
            # plot error vs iteration
            plt.plot(iter_list, ea_list, marker='o')
            plt.xlabel("Iteration")
            plt.ylabel("Approx. Relative Error (%)")
            plt.title("Newton-Raphson: Error vs Iteration")
            plt.grid(True)
            plt.show()

            
            return x_new

        x = x_new
    
    plt.plot(iter_list, ea_list, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Approx. Relative Error (%)")
    plt.title("Newton-Raphson: Error vs Iteration")
    plt.grid(True)
    plt.show()

    raise ValueError("Did not converge within the maximum number of iterations.")


# Function and derivative
# f  = lambda x: x**3 - 0.165*x**2 + 3.993e-4
# df = lambda x: 3*x**2 - 0.33*x
f  = lambda x: x**4 - 5*x - 2
df = lambda x: 4*x**3 - 5

# Run Newton-Raphson
newton_raphson(f, df, x0=1)
