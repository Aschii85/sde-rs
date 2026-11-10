import numpy as np
import sde_rs
from _util import plot_mean, plot_paths


def main():
    # Initial state for the processes
    initial_values = {"X0": 0.5, "X1": 100.0, "X2": 0.0, "X3": 0.0}
    # Equations demonstrating lagged terms:
    # 1. X0 incorporates a mean-reverting drift based on its value 10 steps ago: X0[-10]
    # 2. X1 incorporates a moving-average style diffusion and drift driven by X1[-5] and X0[-2]
    # 3. X2 tracks a spread between X1's current value and its past value 20 steps ago: X1[-20]
    processes_equations = [
        "dX0 = ( 2.0 * (0.5 - X0[-2]) ) * dt + ( 0.1 ) * dN1(X0)",
        "dX1 = ( 0.05 * X1 ) * dt + ( 0.2 * X1[-1] ) * dW1 + ( 0.5 ) * dN1(X0[-2])",
        "dX2 = ( X1 ) * dt",
        "X3 = max(X2/t - 100.0, 0.0)",
    ]
    result = sde_rs.simulate(
        processes_equations=processes_equations,
        time_steps=list(np.arange(0.0, 10.0, 0.01)),
        number_scenarios=10_000,
        initial_values=initial_values,
    )
    print(result)
    plot_paths(result)
    plot_mean(result)


if __name__ == "__main__":
    main()
