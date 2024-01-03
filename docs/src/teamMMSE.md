# Team Mininum Mean Square Error Precoding

Here we provide a simple interface and sample generation tool for producing a toy simulation that we can test our solvers on.

## General Team Minimum Mean Square Error Problem Description

The team MMSE precoding problem is in general a static quadratic team decision problem. In particular, if we have ``N`` transmitters each equipped with ``L`` antennas and ``K`` single antenna receivers, then for every ``k \in \{1,\dots, K\}``, the team MMSE problem reads

```math
\argmin_{\gamma_k \in \Gamma^1 \times \dots \times \Gamma^N} \mathrm{MSE}_k(\gamma) := \mathbb{E} \left[ \left\lVert \mathbf{H}^{\mathsf{H}}\gamma_k(\mathbf{Y}) - \mathbf{e}_k \right\rVert^2 + \frac{1}{P}\left\lVert \gamma_k(\mathbf{Y}) \right\rVert^2 \right].
```

The random variable

```math
\mathbf{H} = [\mathbf{H}_1, \dots, \mathbf{H}_N]
```

is called the **channel** and is centrally symmetric complex gaussian with realizations in ``\mathbb{C}^{K, LN}`` and arbitrary covariance.

For each transmitter (agent) ``i \in \{1,\dots, N\}`` , the measurement random vector ``\mathbf{Y}_i`` is given by

```math
\mathbf{Y}_i := \mathbf{H} + \mathbf{Z}_i
```

where ``\mathbf{Z}_i`` is the **channel estimation error** at transmitter (agent) ``i`` and is also centrally symmetric complex gaussian with realizations in ``\mathbb{C}^{K,LN} and arbitrary covariance.

For further details on team MMSE precoding, please refer to [team-precoding](@cite) and [miretti2022joint](@cite).

## Our Situation

The previously defined measurement model is quite general. While the solvers we provide should be able to handle all measurement schemes that can be constructed from this model in this library, we implement the following model:

```math
\mathbf{Y}_i = \begin{bmatrix} \dots & \varepsilon\mathbf{H}_{i-1} + (1 - \varepsilon)\mathbf{N}_{i-1} & \mathbf{H}_i & \varepsilon\mathbf{H}_{i+1} + (1 - \varepsilon)\mathbf{N}_{i+1} & \dots \end{bmatrix}
```

where ``\mathbf{H}_i`` is the ``i``-th block column of the channel ``\mathbf{H}`` and ``\mathbf{N}`` is centrally symmetric gaussian noise with diagonal covariance matrix.

## Types

```@docs
teamMMSEproblem
```

## Functions

```@docs
sampleComplexNormal(σ::AbstractFloat, K::Int, L::Int)
generateTeamMMSEsamples(t::teamMMSEproblem, m::Int)
```
