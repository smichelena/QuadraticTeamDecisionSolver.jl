# Fixed Point Solvers

Methods and Interface for building solvers for the quadratic team decision problem using empirical 
fixed point iterations.

## Functions

```@docs
generateCrossSamples(N::Int, i::Int, γ::Vector{<:Function}, Y::Vector{<:Vector}, R::Vector{<:Vector})
parallelIterationSolver(
		p::QuadTeamProblem,
		S::Vector{<:Sample},
		functionClass::Vector{<:Function},
		functionNorms::Vector{<:Function},
		outputMap::Function,
		interpolation::Function;
		iterations = 5,
		preprocessData = true,
	)
GeneralOutputMap(
		Y::Vector{<:Vector},
		crossSamples::Vector,
		squareBlocks::Vector,
		r::Vector,
		conditionalMean::Function
	)
```

## Detailed Description of Function Arguments The Solver Takes

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

