import numpy as np
import sde_rs
from _util import plot_mean, plot_paths


def main():
    initial_values = {"X0": 0.5, "X1": 100.0}
    result = sde_rs.simulate(
        processes_equations=[
            "dX0 = ( 2.0 * (0.5 - X0) ) * dt + ( 0.1 ) * dN1(X0)",
            "dX1 = ( 0.05 * X1 ) * dt + ( 0.2 * X1 ) * dW1 + ( 0.5 ) * dN1(X0)",
            "X2 = max(X1 - 100.0, 0.0)",
        ],
        time_steps=list(np.arange(0.0, 10.0, 0.01)),
        number_scenarios=10_000,
        initial_values=initial_values,
        iteration_scheme="euler",
        rng_method="sobol",
    )
    print(result)
    plot_paths(result)
    plot_mean(result)


if __name__ == "__main__":
    main()
