import plotly.express as px
import plotly.graph_objects as go
from sde_rs import SimulationResult


def plot_paths(result: SimulationResult, n_scenarios: int = 10):
    for name in result.coords["process"].values:
        df = (
            result.sel(
                process=name, scenario=slice(0, n_scenarios - 1)
            )  # label-based, inclusive
            .transpose("time", "scenario")
            .to_pandas()  # index = time, one column per scenario
        )
        fig = px.line(
            df,
            labels={"value": "value", "time": "time", "scenario": "scenario"},
            title=f"Simulated SDE Process {name}",
        )
        fig.show()


def plot_mean(result: SimulationResult, band: tuple[float, float] = (0.05, 0.95)):
    mean = result.mean("scenario")  # dims: time, process
    q = result.quantile(list(band), dim="scenario")  # dims: quantile, time, process
    t = result.coords["time"].values

    for name in result.coords["process"].values:
        fig = go.Figure()
        fig.add_scatter(
            x=t,
            y=q.sel(process=name, quantile=band[1]).values,
            mode="lines",
            line={"width": 0},
            showlegend=False,
        )
        fig.add_scatter(
            x=t,
            y=q.sel(process=name, quantile=band[0]).values,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(99,110,250,0.2)",
            name=f"{int(band[0] * 100)}-{int(band[1] * 100)}% band",
        )
        fig.add_scatter(
            x=t,
            y=mean.sel(process=name).values,
            mode="lines",
            name="mean",
            line={"color": "rgba(99,110,250)"},
        )
        fig.update_layout(
            title=f"Mean Simulated SDE Process for {name}",
            xaxis_title="time",
            yaxis_title="value",
        )
        fig.show()
