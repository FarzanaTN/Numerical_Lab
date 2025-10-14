import numpy as np
import time
import matplotlib.pyplot as plt

def input_matrix(n):
    print("Enter coefficient matrix A (row-wise):")
    A = np.zeros((n, n))
    for i in range(n):
        while True:
            try:
                row = list(map(float, input(f"Row {i+1}: ").split()))
                if len(row) != n:
                    raise ValueError("Row must have exactly n elements.")
                A[i] = row
                break
            except ValueError as e:
                print("Invalid input:", e)
    return A

def input_vector(n, name):
    print(f"Enter {name} vector ({n} values):")
    v = np.zeros(n)
    for i in range(n):
        while True:
            try:
                v[i] = float(input(f"{name}[{i+1}]: "))
                break
            except ValueError:
                print("Invalid number, try again.")
    return v

def check_diagonal(A):
    if np.any(np.isclose(np.diag(A), 0)):
        raise ZeroDivisionError("Matrix has zero diagonal element(s).")

def jacobi(A, b, x0, tol, max_iter):
    n = len(b)
    check_diagonal(A)
    x = x0.copy()
    error_list = []
    residual_list = []
    
    print("\nIter | " + " ".join([f"x{i+1:>10}" for i in range(n)]) + " | Residual")
    print("-" * (15 + 12 * n))

    for k in range(1, max_iter + 1):
        x_new = np.zeros_like(x)
        
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]

        abs_err = np.abs(x_new - x)
        max_err = np.max(abs_err)
        residual = np.linalg.norm(A @ x_new - b)

        error_list.append(max_err)
        residual_list.append(residual)
        
        print(f" {k:3d} | {k:4d} | " + " ".join([f"{val:10.6f}" for val in x]) + f" | {residual:10.6e}")


        # print(f"[Jacobi] Iter {k:3d} | Residual: {residual:.6e}")
        # print(f"[Jacobi] Iter {k:3d} | Residual: {residual:.6e}, Max Error: {max_err:.6e}")


        if residual < tol:
            print("→ Jacobi converged!\n")
            return x_new, k, residual, error_list, residual_list
        x = x_new

    print("Jacobi did NOT converge within max iterations.\n")
    return x, max_iter, residual, error_list, residual_list

def gauss_seidel(A, b, x0, tol, max_iter):
    n = len(b)
    check_diagonal(A)
    x = x0.copy()
    error_list = []
    residual_list = []
    
    print("\nIter | " + " ".join([f"x{i+1:>10}" for i in range(n)]) + " | Residual")
    print("-" * (15 + 12 * n))

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i, j] * x[j] for j in range(i))
            s2 = sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]

        diff = x - x_old
        abs_err = np.abs(x - x_old)
        max_err = np.max(abs_err)
        residual = np.linalg.norm(A @ x - b)

        error_list.append(max_err)
        residual_list.append(residual)

        # print(f"[Gauss-Seidel] Iter {k:3d} | Residual: {residual:.6e}, Max Error: {max_err:.6e}")
        # print(f"[Gauss-Seidel] Iter {k:3d} | Residual: {residual:.6e}")
        print(f" {k:3d} | {k:4d} | " + " ".join([f"{val:10.6f}" for val in x]) + f" | {residual:10.6e}")

        if residual < tol:
            print("→ Gauss-Seidel converged!\n")
            return x, k, residual, error_list, residual_list

    print("Gauss-Seidel did NOT converge within max iterations.\n")
    return x, max_iter, residual, error_list, residual_list

def plot_comparison(errors_jacobi, errors_gs, res_jacobi, res_gs):
    # # ---- Plot 1: Iteration vs Maximum Absolute Error ----
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, len(errors_jacobi)+1), errors_jacobi, 'o-', label="Jacobi")
    # plt.plot(range(1, len(errors_gs)+1), errors_gs, 's-', label="Gauss-Seidel")
    # plt.xlabel("Iteration")
    # plt.ylabel("Max Absolute Error")
    # plt.title("Iteration vs Max Error")
    # plt.yscale("log")
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.legend()

    # ---- Plot 2: Iteration vs Residual Error ----
    # plt.subplot(1, 2, 2)
    plt.plot(range(1, len(res_jacobi)+1), res_jacobi, 'o-', label="Jacobi")
    plt.plot(range(1, len(res_gs)+1), res_gs, 's-', label="Gauss-Seidel")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Error (‖Ax - b‖₂)")
    plt.title("Iteration vs Residual Error")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    print("=== Iterative Linear Equation Solver ===\n")
    try:
        n = int(input("Enter number of equations (n ≥ 2): "))
        if n < 2:
            raise ValueError("n must be ≥ 2.")
    except ValueError as e:
        print("Invalid input:", e)
        return

    A = input_matrix(n)
    b = input_vector(n, "b")
    x0 = input_vector(n, "initial guess x(0)")

    try:
        max_iter = int(input("Enter maximum iterations: "))
        tol = float(input("Enter tolerance: "))
        if max_iter <= 0 or tol <= 0:
            raise ValueError("Both must be positive.")
    except ValueError as e:
        print("Invalid numeric input:", e)
        return

    print("\nMatrix A:\n", A)
    print("Vector b:", b)
    print("Initial guess x(0):", x0)
    print(f"Max iterations = {max_iter}, Tolerance = {tol}\n")

    print("=== Running Jacobi Method ===")
    t1 = time.time()
    xj, kj, rj, err_j, res_j = jacobi(A, b, x0, tol, max_iter)
    t2 = time.time()

    print("=== Running Gauss-Seidel Method ===")
    t3 = time.time()
    xg, kg, rg, err_g, res_g = gauss_seidel(A, b, x0, tol, max_iter)
    t4 = time.time()

    print("\n=== Final Results ===")
    print("Jacobi Solution:\n", np.round(xj, 6))
    print(f"Iterations: {kj}, Residual: {rj:.6e}, Time: {t2-t1:.6f}s")
    print("\nGauss-Seidel Solution:\n", np.round(xg, 6))
    print(f"Iterations: {kg}, Residual: {rg:.6e}, Time: {t4-t3:.6f}s")

    print("\n=== Comparative Analysis ===")
    if kj < kg:
        print("Jacobi converged faster.")
    elif kg < kj:
        print("Gauss-Seidel converged faster.")
    else:
        print("Both required the same number of iterations.")
    print(f"Residual comparison → Jacobi: {rj:.2e}, Gauss-Seidel: {rg:.2e}")
    print(f"Time comparison → Jacobi: {t2-t1:.6f}s, Gauss-Seidel: {t4-t3:.6f}s")

    # Plot both error graphs
    plot_comparison(err_j, err_g, res_j, res_g)

    print("\nExecution complete.")

if __name__ == "__main__":
    main()

