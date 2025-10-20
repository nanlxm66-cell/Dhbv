# HBV Conceptual Model Overview

This document summarizes the behavior of the `HBV` class that was provided for review. The
implementation mirrors the hydrological HBV model described by Feng et al. (2022) and Seibert
(2005) and is designed to run in PyTorch so that it remains differentiable.

## High-level structure

* The class derives from `BaseConceptualModel` and exposes a `forward` method that simulates
  discharge for one or more basins in parallel (`n_models`).
* Model parameters can be static or dynamic over time depending on how they are supplied in the
  `parameters` dictionary. Each parameter is represented as a tensor with the shape
  `[batch_size, time_steps, n_models]`.
* The method returns
  * `y_hat`: simulated outflow (mean of quick, intermediate, and slow runoff components),
  * `parameters`: the parameter tensor dictionary passed in,
  * `internal_states`: the full time evolution of each state variable,
  * `final_states`: the state values in the last simulated time step.

## Differentiability in practice

The implementation stays entirely within PyTorch tensor operations, so every arithmetic update in
the snow, soil, and groundwater modules participates in automatic differentiation. No NumPy
conversions or detached tensors appear inside the time-step loop. Even conditional behavior such as
freeze/melt partitioning is handled through differentiable primitives like boolean masking and
`torch.clamp`, so gradients continue to flow to all upstream inputs and parameters. The only hard
floor that is applied is `SM = torch.clamp(SM - ETact, min=1e-5)`, which prevents the soil-moisture
store from reaching exactly zero (a situation that can cause vanishing gradients) while remaining
compatible with autograd.

Because the entire `forward` routine is differentiable, `torch.autograd` can compute gradients of a
loss with respect to both the conceptual-model parameters and any neural network that provides those
parameters. This is the mechanism that makes the HBV component “differentiable” and suitable for
hybrid learning setups.

## Forcing inputs and preprocessing

* `x_conceptual` is expected to contain precipitation, potential evapotranspiration, and
  temperature forcings. If both maximum and minimum temperature are supplied, the average is used.
* Precipitation and temperature inputs are broadcast so the same forcings can be consumed by
  multiple conceptual model replicas (`n_models`).
* Solid and liquid precipitation are separated using the threshold temperature parameter `TT`.
  * Values below `TT` are treated as snow, values above as liquid precipitation.

## Initial state handling

* When `initial_states` are not provided, the model fills every state with a small default value
  (`0.001`) so that gradients remain well-defined.
* Provided `initial_states` must include the five state tensors: `SNOWPACK`, `MELTWATER`, `SM`
  (soil moisture), `SUZ` (upper groundwater zone), and `SLZ` (lower groundwater zone).

## Time-step loop

The model advances states sequentially over the time dimension.

### Snow routine

1. Add solid precipitation to `SNOWPACK`.
2. Compute melt as `CFMAX * (temperature - TT)`, clamped to `[0, SNOWPACK]`.
3. Accumulate melt in `MELTWATER` and reduce `SNOWPACK` accordingly.
4. Compute refreezing with coefficient `CFR`, again clamped to available meltwater.
5. Allow excess meltwater beyond `CWH * SNOWPACK` to percolate to the soil column (`tosoil`).

### Soil moisture and evapotranspiration

1. Soil wetness is `(SM / FC)^BETA` (optionally with an exponent `BETAET` for evapotranspiration).
2. Recharge equals `(liquid precipitation + tosoil) * soil_wetness`.
3. Soil moisture is updated by adding inputs and subtracting recharge and actual ET.
4. Excess water beyond field capacity `FC` becomes `excess` runoff to the upper groundwater store.
5. Actual ET scales potential ET (`et`) by the soil wetness factor, capped by current soil moisture.
6. Soil moisture is clamped to remain above `1e-5` to keep gradients stable.

### Groundwater and runoff

1. Upper zone storage `SUZ` receives recharge and excess water.
2. Percolation `PERC` is the minimum of `SUZ` and the parameter `PERC`.
3. Quick flow `Q0` is controlled by `K0` and `UZL`; intermediate flow `Q1` by `K1`.
4. Lower zone storage `SLZ` accumulates percolation and releases slow flow `Q2` via `K2`.
5. Total discharge is `mean(Q0 + Q1 + Q2)` across conceptual model replicas.

### State bookkeeping

* After each time step the updated states are written into `states` so their full time series can be
  returned.
* `final_states` captures only the values from the last step for use as warm starts in subsequent
  runs.

## Parameter bounds

`parameter_ranges` provides recommended lower and upper bounds for all calibration parameters:

| Parameter | Purpose | Range |
|-----------|---------|-------|
| `BETA` | Soil infiltration shape | 1.0–6.0 |
| `FC` | Soil field capacity (mm) | 50–1000 |
| `K0` | Quick runoff coefficient | 0.05–0.9 |
| `K1` | Intermediate runoff coefficient | 0.01–0.5 |
| `K2` | Slow runoff coefficient | 0.001–0.2 |
| `LP` | Soil moisture limit for ET | 0.2–1.0 |
| `PERC` | Percolation rate | 0–10 |
| `UZL` | Threshold for quick runoff | 0–100 |
| `TT` | Snow melt threshold (°C) | −2.5–2.5 |
| `CFMAX` | Degree-day melt factor | 0.5–10 |
| `CFR` | Refreezing coefficient | 0–0.1 |
| `CWH` | Water holding capacity of snow | 0–0.2 |
| `BETAET` | Exponent for ET reduction | 0.3–5.0 |

These ranges mirror typical HBV calibration limits and can be used for parameter regularization or
constrained optimization during training.

## Static versus dynamic parameterization

`parameter_type` controls whether each HBV parameter is treated as static (time-invariant within a
simulation window) or dynamic (allowed to vary at every step). Regardless of type, the parameters are
passed into `forward` as tensors, so they remain differentiable.

* **Static parameters** typically originate from learnable tensors registered on the
  `BaseConceptualModel` (for example as `nn.Parameter`s). During training, gradient signals coming
  from the loss backpropagate through the HBV computations to those tensors. Updating them is then a
  matter of running an optimizer such as Adam or SGD over the conceptual-model parameters—exactly the
  same workflow as calibrating a neural network weight.
* **Dynamic parameters** are usually predicted by an upstream network (e.g., an LSTM or transformer)
  that consumes meteorological forcings or catchment attributes. The chosen entries in
  `parameter_type` tell the base class to request a full `[batch, time, n_models]` trajectory instead
  of a single static value. Because the HBV equations are differentiable, gradients propagate back
  through the dynamic parameter tensors into the weights of that upstream network, enabling their
  end-to-end training.

In short, both static and dynamic parameter updates leverage the same backpropagation path; the only
difference is whether the learnable quantities live directly on the HBV module (static) or inside an
external network that outputs time-varying parameter fields (dynamic).
