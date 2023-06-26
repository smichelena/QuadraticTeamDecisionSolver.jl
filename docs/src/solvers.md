# Solvers

Methods and Interface for building solvers for the quadratic team decision problem using empirical methods.

## Fixed Point Solvers

For a team decision problem with convex loss, the Nash equilibrium becomes a system of linear operator equations known as the 
'Stationary Equations'. For our problem, these read:

```math
\begin{align*}
	\mathbb{E} \left[ \mathbf{R}_{ii} \vert \mathbf{Y}_i \right]\gamma^i(\mathbf{Y}_i) &= \sum_{i,j=1  , j \neq i} \mathbb{E} \left[ \mathbf{R}_{ij}  \gamma_{(k)}^j \left( \mathbf{Y}_j \right) \vert \mathbf{Y}_i \right] + \mathbb{E} \left[ \mathbf{r}_i \vert \mathbf{Y}_i \right] \\
	i = 1, \dots, N
\end{align*}
```

The following fix point schemes approximate the solution of these equations, after restricting the problem to a finite dimensional subspace.

### Jacobi Iteration

This is an iteration of the form:

```math
\begin{align*}
	\gamma_{(k+1)}^i(\mathbf{Y}_i) &= -\left[ \mathbb{E} \left[ \mathbf{R}_{ii} \vert \mathbf{Y}_i \right] \right]^{-1} \left\{ \sum_{i,j=1  , j \neq i} \mathbb{E} \left[ \mathbf{R}_{ij}  \gamma_{(k)}^j \left( \mathbf{Y}_j \right) \vert \mathbf{Y}_i \right] + \mathbb{E} \left[ \mathbf{r}_i \vert \mathbf{Y}_i \right] \right\} \\
	\gamma_{(0)}^i &\equiv 0  \qquad  i = 1, \dots, N
\end{align*}
```

-`empiricalJacobiSolver` performs this iteration restricted to the finite dimensional reproducing kernel Hilbert space spanned by basis samples, that is, we set:

```math
	\mathbb{V}_i^{(m)} = \text{span}\left\{ k( \ \cdot \ , y_i^{(1)}), \dots, k( \ \cdot \ , y_i^{(m)}) \right\}
```

### Sequential Iteration

Similar to the Jacobi iteration, but now we update the agents in strict order:

```math
\begin{align*}
	\gamma_{(k+1)}^i(\mathbf{Y}_i) &= -\left[ \mathbb{E} \left[ \mathbf{R}_{ii} \vert \mathbf{Y}_i \right] \right]^{-1} \left\{ \sum_{i,j=1,j < i} \mathbb{E} \left[ \mathbf{R}_{ij}  \gamma_{(k)}^j \left( \mathbf{Y}_j \right) \vert \mathbf{Y}_i \right] + \sum_{i,j=1,j > i} \mathbb{E} \left[ \mathbf{R}_{ij}  \gamma_{(k)}^j \left( \mathbf{Y}_j \right) \vert \mathbf{Y}_i \right] + \mathbb{E} \left[ \mathbf{r}_i \vert \mathbf{Y}_i \right] \right\} \\
	\gamma_{(0)}^i &\equiv 0  \qquad  i = 1, \dots, N
\end{align*}
```

-`empiricalJacobiSolver` performs this iteration restricted to the finite dimensional reproducing kernel Hilbert space spanned by basis samples, that is, we set:

```math
	\mathbb{V}_i^{(m)} = \text{span}\left\{ k( \ \cdot \ , y_i^{(1)}), \dots, k( \ \cdot \ , y_i^{(m)}) \right\}
```

-`naiveIterativeSolver` performs this iteration for any arbitrary function class parametrized by some ``\theta``.

### Note

The empirical solvers will be generalized to work with any kind of basis, not only with a kernel basis.

## Direct Solvers

Compute the fix point directly over an empirical basis, coming soon!

## Functions

```@docs
naiveIterativeSolver(
		p::QuadTeamProblem,
		S::Vector{<:Sample},
		functionClass::Vector{<:Function},
		functionNorms::Vector{<:Function},
		outputMap::Function,
		interpolation::Function;
		iterations = 5,
		preprocessData = true,
	)
empiricalJacobiSolver!(
	p::QuadTeamProblem,
	w::Vector{<:Vector},
	Yᵃ::Vector{<:Vector{<:Vector}},
	Yᵇ::Vector{<:Vector{<:Vector}},
	Rblocks::Vector{<:Vector{<:Vector{<:Matrix}}},
	rblocks::Vector{<:Vector{<:Vector}},
	kernels::Vector{<:Function},
	conditionalMean::Function;
	iterations = 5,
	λ = 1,
)
empiricalAlternatingSolver!(
		p::QuadTeamProblem,
		w::Vector{<:Vector},
		Yᵃ::Vector{<:Vector{<:Vector}},
		Yᵇ::Vector{<:Vector{<:Vector}},
		Rblocks::Vector{<:Vector{<:Vector{<:Matrix}}},
		rblocks::Vector{<:Vector{<:Vector}},
		kernels::Vector{<:Function},
		conditionalMean::Function;
		iterations = 5,
		λ = 1,
	)
GeneralOutputMap(
		Y::Vector{<:Vector},
		crossSamples::Vector,
		squareBlocks::Vector,
		r::Vector,
		conditionalMean::Function
	)
```

## Detailed Description of Function Arguments The Naive Solver Takes

The purpose of this solver is to provide a reasonable interface to solve this problem using deep learning methods.

### `functionClass::Function`

#### Arguments
- `weights`: A vector that correspond to the basis function weights / function class parameters
#### Returns
- ``f``: A function in the hypothesis function class ``\mathcal{F}``.

#### Remarks
This function is used to construct each entry of the solution function vector  `γ`, as such, great care must be taken that it can properly handle 
vector valued outputs. Moreover the  `interpolation` and `functionNorm` methods must be tailor made to suit this function class.

Note that during each iteration, samples of the random variable ``\mathbf{R}_{i,j}\gamma_{(k)}^j(\mathbf{Y}_j)`` are approximately generated using the last
estimate of ``\gamma^j``, so a very good choice of ``\mathcal{F}`` is critical to the success of the scheme.

### `functionNorm::Function`

#### Arguments
-  `weights`: A vector that correspond to the basis function weights / function class parameters.

#### Returns
- ``||f||``: The function norm of the function ``f`` described by the weights `weights`.

### `interpolation::Function`

#### Arguments
- `Y[i]`: Measurement data for the agent ``i``.
- `O[i]`: Output map data for the agent ``i``.

#### Returns
- `weights`: The weights that correspond to the function ``f`` in our function class ``\mathcal{F}`` that best interpolates `Y[i]` and `O[i]`.

#### Remarks
As with `functionClass`, care most be taken that this can hanble both vector and scalar valued input and output data.

### `outputMap::Function`

#### Arguments
- `Y[i]`: Measurement data for the agent ``i``. 
- `crossSamples`: *generated* sample data from the random variable given by: ``\mathbf{R}_{i,j}\gamma_{(k)}^j(\mathbf{Y}_j)``.
- `squareBlocks`: Samples from ``\mathbf{R}_{i,i}``. These should be Hermitian and Positive Definite.
- `r_blocks[i]`: Samples from ``\mathbf{r}_i``

#### Returns
- `O`: Vector of approximated samples of the random variable ``\left( \mathbb{E}[\mathbf{R}_{i,i}|\mathbf{Y}_i] \right)^{-1}\left( \sum_{i,j=1 \ , \ i \neq j}^N \mathbb{E}[\mathbf{R}_{i,j}\gamma_{(k)}^j(\mathbf{Y}_j)|\mathbf{Y}_i] + \mathbb{E}[\mathbf{r}_i|\mathbf{Y}_i] \right)``

#### Remarks
This method must have within it a method that estimates conditional expectations of the form ``\mathbb{E}[\mathbf{X}|\mathbf{Y}=y]`` see: [`GeneralOutputMap(Y::Vector{<:Vector}, crossSamples::Vector{<:Vector}, squareBlocks::Vector{<:Matrix}, r::Vector{<:Vector}, conditionalMean::Function)`](@ref)

Otherwise, an `outputMap` can also be implemented with a method that estimates conditional expectation operators of the form ``\mathbb{E}[\mathbf{X}|\mathbf{Y}]``.

