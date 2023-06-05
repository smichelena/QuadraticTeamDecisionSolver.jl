# QuadraticTeamDecisionSolver.jl Documentation

```@contents
```

## General Problem Description

The quadratic team decision problem aims to optimize a team of agents' actions to minimize a quadratic objective function. The problem can be formulated as follows:

Minimize:

```math
J(\gamma) = \mathbb{E}_\mathbf{X} \left[ \sum_{i,j = 1}^N \gamma^i(\mathbf{Y}_i)^{\top} \mathbf{R}_{i,j}(\mathbf{X}) \gamma^j (\mathbf{Y}_j)+ 2 \Re \left( \sum_{i = 1}^N \gamma^j (\mathbf{Y}_j)^{\top} \mathbf{r}_i(\mathbf{X})  \right) + \mathbf{c}(\mathbf{X}) \right]
```

where:

- ``\mathbf{X}`` is a random variable representing the state of the world, which is not directly accessible to the agents.
- ``\mathbf{Y}_i`` is the random variable the agent ``i`` measures. It has realizations in ``\mathbb{F}^{m_i}``.
- ``\gamma_i(\mathbf{Y}_i)`` is the control policy function for agent ``i`` that maps a realization of ``\mathbf{Y}_i`` to the action ``\mathbf{U}_i \in \mathbb{F}^{a_i}`` taken by agent ``i``
- ``\mathbf{c}(\mathbf{X})`` is a scalar valued random variable that depends on ``\mathbf{X}``.
- ``\mathbf{R}_{i,j}(\mathbf{X})`` is the matrix capturing the quadratic term for the interaction between agent ``i`` and agent ``j``, which depends on the state ``\mathbf{X}``. It has realizations in ``\mathcal{L}(\mathbb{F}^{a_j}, \mathbb{F}^{a_i})``.


## Index

```@index
```