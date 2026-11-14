import numpy as np
import sde_rs
from _util import plot_mean, plot_paths


def main():
    mu = 0.05
    sigma = 0.1
    start_value = 1.0

    result = sde_rs.simulate(
        processes_equations=[
            f"dX1 = ( {mu} * X1 ) * dt + ( {sigma} * X1) * dW1",
        ],
        time_steps=list(np.arange(0.0, 10.0, 0.01)),
        number_scenarios=10_000,
        initial_values={"X1": start_value},
        rng_method="sobol",
    )
    print(result)
    plot_paths(result)
    plot_mean(
        result,
        np.array([start_value * np.exp(mu * t) for t in result.coords["time"].values]),
    )


if __name__ == "__main__":
    main()
