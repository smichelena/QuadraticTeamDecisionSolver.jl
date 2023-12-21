# QuadraticTeamDecisionSolver.jl Documentation

## Theoretical Problem Description

The quadratic team decision problem aims to optimize a team of agents' actions to minimize a quadratic objective function.

Our exact formulation can be found in [yuksel](@cite), however the problem was initially formulated in [radner](@cite).

Importantly, in any static team decision problem, every random variable at play depends on **the state of the world** ``\mathbf{X}``, which is the variable that adds "randomness" to the system and thus cannot be influenced by the agents' actions.

With this in mind, the static quadratic team decision problem can be formulated as follows:

```math
\argmin_{\gamma \in \Gamma^1 \times \dots \times \Gamma^N} J(\gamma) := \mathbb{E} \left[ \sum_{i,j = 1}^N \gamma^i(\mathbf{Y}_i)^{\mathsf{H}} \mathbf{Q}_{i,j} \gamma^j (\mathbf{Y}_j) + 2 \mathrm{Re} \left( \sum_{i = 1}^N \gamma^j (\mathbf{Y}_j)^{\mathsf{H}} \mathbf{R}_i  \right) + \mathbf{c} \right]
```

where:

- ``\mathbf{Y}_i`` is the random variable the agent ``i`` measures. It has realizations in ``\mathbb{F}^{m_i}``.
- ``\mathbf{Q}_{i,j}`` is a random matrix that captures the quadratic term for the interaction between agent ``i`` and agent ``j``, which depends on the state ``\mathbf{X}``. It has realizations in ``\mathcal{L}(\mathbb{F}^{a_j}, \mathbb{F}^{a_i})``.
- ``\mathbf{R}_i`` is a random vector that depends on the state ``\mathbf{X}``.
- ``\mathbf{c}`` is a scalar valued random variable that depends on ``\mathbf{X}``.
- ``\gamma_i`` is the policy function for agent ``i`` that maps a realization of ``\mathbf{Y}_i`` to the action ``\mathbf{U}_i \in \mathbb{F}^{a_i}`` taken by agent ``i``. In this setting, we require ``\gamma^i`` to have bounded second moment with respect to ``\mathbf{Y}_i``.

Note that every random variable previously mentioned is taken as a deterministic function of the state of the world ``\mathbf{X}``.

NFurthermore,the block matrix

```math
\mathbf{Q} := \begin{bmatrix} \mathbf{Q}_{1,1} & \dots & \mathbf{Q}_{1,N} \\
                              \vdots & \ddots & \vdots \\
                              \mathbf{Q}_{N,1} & \dots & \mathbf{Q}_{N,N}
                              \end{bmatrix}
```

fulfills the following critical conditions:

- ``\mathbf{Q}`` is **uniformly bounded above**, that is, there exists ``0 < \lambda_{\mathrm{max}} < \infty`` such that

```math
\mathbb{P}\left( \lambda_{\mathrm{max}}I - \mathbf{Q} \succ 0 \right) = 1.
```

- ``\mathbf{Q}`` is **uniformly bounded below**, that is, there exists ``\lambda_{\mathrm{min}} > 0`` such that

```math
\mathbb{P}\left( \mathbf{Q} - \lambda_{\mathrm{min}}I \succ 0 \right) = 1.
```

These conditions ensure a solution to our problem exists and is unique. Furthermore, they determine the convergence of the policy update schemes that we employ in our numerics.

## Data Structures Employed

We employ nested vectors for simplicity.

### Measurement Vectors ``\mathbf{Y}``

We store the samples of our measurement vectors as a `vector{<:vector}`. In particular, if we have `m` samples, then `Y` is a vector of length `N` (number of agents) of vectors of length `m`. Each `m` length vector then contains `m` vectors, each of length ``m_i`` (dimension of measurements for agent ``i``) of real or complex numbers.

For example, suppose we have 3 agents with ``m_1 = 1, m_2 = 2`` and ``m_3 = 3`` and we have 10 samples. Then `Y` is a vector of length 3 with

```math
\mathbf{Y}[1] = \left[ \left[y[1]_1^{(1)} \right], \dots , \left[y[1]_1^{(10)}\right] \right]
```

```math
\mathbf{Y}[2] = \left[ \left[y[2]_1^{(1)}, y[2]_2^{(1)}\right], \dots , \left[y[2]_1^{(10)}, y[2]_2^{(10)}\right] \right]
```

```math
\mathbf{Y}[3] = \left[ \left[y[3]_1^{(1)}, y[3]_2^{(1)}, y[3]_2^{(1)}\right], \dots , \left[y[3]_1^{(10)}, y[3]_2^{(10)}, y[3]_2^{(10)}\right] \right]
```

### System Random Vectors ``\mathbf{R}``

These employ the same exact structure as the measurement vectors ``\mathbf{Y}``.

### System Random Matrices ``\mathbf{Q}``

In this case, we use the same basic structure as before, but double indexed. In particular, if we have `m` samples, then `Q` is an `N` by `N` matrix of vectors of length `m`. The vector in the ``i,j``-th position then contains `m` matrices, each of dimensions ``a_j \times a_i`` (dimension of measurements for agent ``i``) of real or complex numbers.

For example, suppose we have 2 agents with ``a_1 = 1`` and ``a_2 = 2`` and we have 10 samples. Then `Q` is a 2 by 2 matrix with

```math
\mathbf{Q}[1,1] = \left[ \left[Q[1,1]_1^{(1)} \right], \dots , \left[Q[1,1]_1^{(10)}\right] \right]
```

```math
\mathbf{Q}[1,2] = \left[ \begin{bmatrix} Q[1,2]_1^{(1)} & Q[1,2]_2^{(1)} \end{bmatrix} , \dots , \begin{bmatrix} Q[1,2]_1^{(10)} & Q[1,2]_2^{(10)} \end{bmatrix} \right]
```

```math
\mathbf{Q}[2,1] = \left[ \begin{bmatrix} Q[2,1]_1^{(1)} \\ Q[2,1]_2^{(1)} \end{bmatrix} , \dots , \begin{bmatrix} Q[2,1]_1^{(10)} \\ Q[2,1]_2^{(10)} \end{bmatrix} \right]
```

```math
\mathbf{Q}[2,2] = \left[ \begin{bmatrix} Q[2,2]_{1,1}^{(1)} & Q[2,2]_{1,2}^{(1)} \\ Q[2,2]_{2,1}^{(1)} & Q[2,2]_{2,2}^{(1)} \end{bmatrix} , \dots , \begin{bmatrix} Q[2,2]_{1,1}^{(10)} & Q[2,2]_{1,2}^{(10)} \\ Q[2,2]_{2,1}^{(10)} & Q[2,2]_{2,2}^{(10)} \end{bmatrix} \right]
```

## References

```@bibliography
```

## Index

```@index
```
