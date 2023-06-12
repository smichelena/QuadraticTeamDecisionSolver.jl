"""
	generateCrossSamples(N::Int, i::Int, γ::Vector{<:Function}, Y::Vector{<:Vector}, R::Vector{<:Vector})

Generate cross samples for a quadratic team decision problem.

# Arguments
- `N::Int`: Total number of agents in the problem.
- `i::Int`: Index of the current agent.
- `γ::Vector{<:Function}`: Vector of function class instances representing the decision variables.
- `R::Vector{<:Vector}`: Vector of vectors representing the data samples.

# Returns
- `crossSamples::Vector`: Vector of cross samples for the current agent.

"""
function generateCrossSamples(
	N::Int,
	i::Int,
	γ::Vector{<:Function},
	Y::Vector{<:Vector},
	R::Vector{<:Vector},
)
	crossRange = if i == 1
		2:N
	elseif i == N
		1:(N-1)
	else
		vcat(1:(i-1), (i+1):N)
	end

	return [sum([Rₛ[j] * γ[j](y) for j in crossRange]) for (Rₛ, y) in zip(R, Y)]
end


using ProgressLogging

"""
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

Approximate the solution to a quadratic team decision problem using a parallel iteration scheme.

# Arguments
- `p::QuadTeamProblem`: A `QuadTeamProblem` object defining the problem's specifications.
- `S::Vector{<:Sample}`: A vector of `Sample` objects representing the data samples.
- `functionClass::{<:Function}`: A vector of functions that for each agent provides a function that constructs an instance of the function class used for optimization.
- `functionNorm::{<:Function}`: A vector of functions that for each agent provides a function that computes the norm of a function class instance.
- `outputMap::Function`: A function that maps performs conditional mean estimation.
- `interpolation::Function`: A function that performs interpolation for updating the decision variables.
- `iterations::Int = 5`: The number of fixed-point iterations to perform.
- `preprocessData::Bool = true`: Whether to perform data preprocessing and validation.

# Returns
- `weights::Vector`: A vector of weight vectors representing the decision variables at each iteration.
- `norms::Vector{Vector{Float64}}`: A vector of vectors containing the norms of the weight vectors at each iteration for each agent.
- `empiricalCost::Vector{Float64}`: A vector containing the empirical cost at each iteration.

"""
function parallelIterationSolver(
	p::QuadTeamProblem,
	S::Vector{<:Sample},
	functionClass::Vector{<:Function},
	functionNorms::Vector{<:Function},
	outputMap::Function,
	interpolation::Function;
	iterations = 5,
	preprocessData = true,
)

	#check data is fine
	if preprocessData
		p = checkProblem(p)
		S = checkData(p, S)
	end

	#split data blocks
	Y, R_blocks, r_blocks = splitDataSetIntoBlocks(p, S)

	#initialize gamma
	m = length(S)
	weights = [zeros(p.T, m) for _ in 1:p.N]
	γ = [functionClass[i](weights[1]) for i in 1:p.N] #as many gammas as total output dimensions

	#initialize convergence tracking 
	norms = [Vector{Float64}() for _ in 1:p.N]
	empiricalCost = Vector{Float64}()
	#append!(empiricalCost, risk(S, γ))

	@progress for k in 1:iterations  #fixed point iterations

		for i in 1:p.N #solve for each agent

			#generate iteration data
			crossSamples = generateCrossSamples(p.N, i, γ, Y[i], R_blocks[i])
			squareBlocks = [Rₛ[i] for Rₛ in R_blocks[i]]

			#estimate conditional means
			O = outputMap(Y[i], crossSamples, squareBlocks, r_blocks[i])

			#interpolate and update \gamma
			weights[i] = interpolation(Y[i], O)
			γ[i] = functionClass[i](weights[i])

			#update norms
			append!(norms[i], functionNorms[i](weights[i]))

		end

		#update empirical riskß
		append!(empiricalCost, risk(S, γ))
	end

	return weights, norms, empiricalCost

end

"""
	GeneralOutputMap(
		Y::Vector{<:Vector},
		crossSamples::Vector,
		squareBlocks::Vector,
		r::Vector,
		conditionalMean::Function
	)


Computes the output map for the agent ``i`` in the current iteration.

# Arguments: 

- Samples of ``\\mathbf{Y}_i``
- Samples of ``\\mathbf{R}_{i,j}(\\mathbf{X})\\gamma_{(k)}^j(\\mathbf{Y}_j)``
- Samples of ``\\mathbf{R}_{i,j}``
- Samples of ``\\mathbf{r}_{i}``
- `conditionalMean`:A method to compute a contional mean of the form ``\\mathbb{E}[\\mathbf{X}|\\mathbf{Y}=y]``

# Usage:

Pass an anonymous function of the form:
```julia
	(Yᵢ, Rᵢⱼ, Rᵢᵢ, rᵢ) -> GeneralOutputMap(Y, Rᵢⱼ, Rᵢᵢ, rᵢ, conditionalMean)
```
as the `outputMap` argument to any of the fix point solvers. 

# Note:

Check / fine tune argument types!

"""
function GeneralOutputMap(
	Y::Vector{<:Vector},
	crossSamples::Vector,
	squareBlocks::Vector,
	r::Vector,
	conditionalMean::Function,
)
	crossOutput = Y .|> y -> conditionalMean(crossSamples, Y, y)
	Routput = Y .|> y -> conditionalMean(squareBlocks, Y, y)
	rOutput = Y .|> y -> conditionalMean(r, Y, y)

	#if output is not scalar valued then division becomes matrix inversion
	return (ndims(crossOutput[1]) == 0) ? (crossOutput + rOutput) ./ Routput :
		   Routput .\ (crossOutput + rOutput)
end
