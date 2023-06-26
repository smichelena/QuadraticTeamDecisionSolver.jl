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

"""
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

Approximate the solution to a quadratic team decision problem using a sequential iteration scheme.
Should work for any function class parametrized by some parameter ``\\theta``.

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
function naiveIterativeSolver(
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
	append!(empiricalCost, risk(S, γ))

	for k in 1:iterations  #fixed point iterations

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

function assembleCrossBlock(
	T::Type,
	m::Int,
	Yᵢᵃ::Vector{<:Vector},
	Yᵢᵇ::Vector{<:Vector},
	Yⱼᵃ::Vector{<:Vector},
	Yⱼᵇ::Vector{<:Vector},
	Rᵢⱼ::Vector{<:Matrix},
	kʲ::Function,
	conditionalMean::Function,
)

	Eᵢⱼ = zeros(T, m, m)

	for (k, yᵢ) in zip(1:m, Yᵢᵇ)
		for (l, yⱼ) in zip(1:m, Yⱼᵇ)
			Eᵢⱼ[l, k] = conditionalMean(
				[R * kʲ(y, yⱼ) for (R, y) in zip(Rᵢⱼ, Yⱼᵃ)],
				Yᵢᵃ,
				yᵢ,
			)
		end
	end

	return Eᵢⱼ

end

function assembleDiagonalBlock(
	T::Type,
	m::Int,
	Rᵢᵢ::Vector{<:Matrix},
	Yᵢᵃ::Vector{<:Vector},
	Yᵢᵇ::Vector{<:Vector},
	conditionalMean::Function,
)

	Eᵢᵢ = zeros(T, m)

	for (k, yᵢ) in zip(1:m, Yᵢᵇ)
		Eᵢᵢ[k] = conditionalMean(Rᵢᵢ, Yᵢᵃ, yᵢ)
	end

	return Eᵢᵢ

end

function assembleAffineBlock(
	T::Type,
	m::Int,
	rᵢ::Vector{<:Vector},
	Yᵢᵃ::Vector{<:Vector},
	Yᵢᵇ::Vector{<:Vector},
	conditionalMean::Function,
)

	r = zeros(T, m)

	for (k, yᵢ) in zip(1:m, Yᵢᵇ)
		r[k] = conditionalMean(rᵢ, Yᵢᵃ, yᵢ)
	end

	return r

end

"""
	assembleSystem(
		T::Type, 
		i::Int, 
		m::Int, 
		N::Int, 
		Rᵢ::Vector{<:Vector{<:Matrix}}, 
		rᵢ::Vector{<:Vector}, 
		Yᵃ::Vector{<:Vector{<:Vector}}, 
		Yᵇ::Vector{<:Vector{<:Vector}}, 
		kernels::Vector{<:Function}, 
		conditionalMean::Function
	)

Assembles and returns various blocks of a system.

# Arguments
- `T::Type`: The type of the system.
- `i::Int`: The index of the current block.
- `m::Int`: The size of the system.
- `N::Int`: The total number of blocks.
- `Rᵢ::Vector{<:Vector{<:Matrix}}`: A vector of matrices.
- `rᵢ::Vector{<:Vector}`: A vector of vectors.
- `Yᵃ::Vector{<:Vector{<:Vector}}`: A vector of 3D arrays.
- `Yᵇ::Vector{<:Vector{<:Vector}}`: A vector of 3D arrays.
- `kernels::Vector{<:Function}`: A vector of kernel functions.
- `conditionalMean::Function`: The conditional mean function.

# Returns
- `Kᵢ`: The gramian matrix calculated using the kernel function `kernels[i]` and `Yᵇ[i]`.
- `Eᵢᵢ`: The diagonal block assembled using the `assembleDiagonalBlock` function.
- `Eᵢ`: An array of cross blocks assembled using the `assembleCrossBlock` function.
- `rᵢ`: The affine block assembled using the `assembleAffineBlock` function.
"""
function assembleSystem(
	T::Type,
	i::Int,
	m::Int,
	N::Int,
	Rᵢ::Vector{<:Vector{<:Matrix}},
	rᵢ::Vector{<:Vector},
	Yᵃ::Vector{<:Vector{<:Vector}},
	Yᵇ::Vector{<:Vector{<:Vector}},
	kernels::Vector{<:Function},
	conditionalMean::Function,
)

	Kᵢ = gramian(kernels[i], Yᵇ[i])

	Eᵢᵢ = assembleDiagonalBlock(
		T,
		m,
		[R[i] for R in Rᵢ],
		Yᵃ[i],
		Yᵇ[i],
		conditionalMean,
	)

	crossRange = if i == 1
		2:N
	elseif i == N
		1:(N-1)
	else
		vcat(1:(i-1), (i+1):N)
	end

	Eᵢ = []

	for j in crossRange
		append!(Eᵢ,
			[
				assembleCrossBlock(
					T,
					m,
					Yᵃ[i],
					Yᵇ[i],
					Yᵃ[j],
					Yᵇ[j],
					[R[j] for R in Rᵢ],
					kernels[j],
					conditionalMean,
				),
			])
	end

	rᵢ = assembleAffineBlock(T, m, rᵢ, Yᵃ[i], Yᵇ[i], conditionalMean)

	return Kᵢ, Eᵢᵢ, Eᵢ, rᵢ

end

"""
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

Approximate the solution to a quadratic team decision problem using a parallel iteration scheme. 
Restricts the operator Jacobi iteration to a finite dimensional reproducing kernel Hilbert space.

# Parameters:
- `p: QuadTeamProblem` - The QuadTeamProblem object.
- `w: Vector{<:Vector}` - The weight vector. Must be pre-initialized and will be modified in place.
- `Yᵃ: Vector{<:Vector{<:Vector}}` - The vector of agent samples used to generate conditional mean approximations.
- `Yᵇ: Vector{<:Vector{<:Vector}}` - The vector of agent samples used to generate the finite-dimensional reproducing kernel Hilbert space, which serves as our ansatz space.
- `Rblocks: Vector{<:Vector{<:Vector{<:Matrix}}}` - The vector of matrix samples corresponding to each agent.
- `rblocks: Vector{<:Vector{<:Vector}}` - The vector of samples of ``\\mathbf{r}`` corresponding to each agent.
- `kernels: Vector{<:Function} `- The kernel functions. (Vector as it may differ per agent).
- `conditionalMean: Function `- Function to approximate conditional means.

# Keyword Arguments:
- `iterations: Int` - The number of iterations for the empirical Jacobi method. Default is 5.
- `λ: Float` - The regularization parameter for kernel gramian inversion. Default is 1.

# Returns:
- `w: Vector{<:Vector}` - The updated weight vector.
"""
function empiricalJacobiSolver!(
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
	K = []
	Eₛ = []
	Eᵣ = []
	rₜ = [] #dont like the subscript

	m = length(Yᵇ[1])

	for i in 1:p.N
		k, eₛ, eᵣ, r = assembleSystem(
			p.T,
			i,
			m,
			p.N,
			Rblocks[i],
			rblocks[i],
			Yᵃ,
			Yᵇ,
			kernels,
			conditionalMean,
		)
		append!(K, [k])
		append!(Eₛ, [eₛ])
		append!(Eᵣ, [eᵣ])
		append!(rₜ, [r])
	end

	for _ in 1:iterations

		E = []

		for i in 1:p.N
			crossRange = if i == 1
				2:p.N
			elseif i == p.N
				1:(p.N-1)
			else
				vcat(1:(i-1), (i+1):p.N)
			end

			append!(
				E,
				[sum([E * w[j][end] for (E, j) in zip(Eᵣ[i], crossRange)])],
			)
		end

		for i in 1:p.N
			append!(w[i], [(K[i] + λ * I) \ -(Eₛ[i] .\ (E[i] + rₜ[i]))])
		end

	end

	return w

end

"""
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

Approximate the solution to a quadratic team decision problem using a parallel iteration scheme. 
Restricts the operator alternating iteration to a finite dimensional kernel basis.

# Parameters:
- `p: QuadTeamProblem` - The QuadTeamProblem object.
- `w: Vector{<:Vector}` - The weight vector. Must be pre-initialized and will be modified in place.
- `Yᵃ: Vector{<:Vector{<:Vector}}` - The vector of agent samples used to generate conditional mean approximations.
- `Yᵇ: Vector{<:Vector{<:Vector}}` - The vector of agent samples used to generate the finite-dimensional reproducing kernel Hilbert space, which serves as our ansatz space.
- `Rblocks: Vector{<:Vector{<:Vector{<:Matrix}}}` - The vector of matrix samples corresponding to each agent.
- `rblocks: Vector{<:Vector{<:Vector}}` - The vector of samples of ``\\mathbf{r}`` corresponding to each agent.
- `kernels: Vector{<:Function} `- The kernel functions. (Vector as it may differ per agent).
- `conditionalMean: Function `- Function to approximate conditional means.

# Keyword Arguments:
- `iterations: Int` - The number of iterations for the empirical Jacobi method. Default is 5.
- `λ: Float` - The regularization parameter for kernel gramian inversion. Default is 1.

# Returns:
- `w: Vector{<:Vector}` - The updated weight vector.

"""
function empiricalAlternatingSolver!(
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
	K = []
	Eₛ = []
	Eᵣ = []
	rₜ = [] #dont like the subscript

	m = length(Yᵇ[1])

	for i in 1:p.N
		k, eₛ, eᵣ, r = assembleSystem(
			p.T,
			i,
			m,
			p.N,
			Rblocks[i],
			rblocks[i],
			Yᵃ,
			Yᵇ,
			kernels,
			conditionalMean,
		)
		append!(K, [k])
		append!(Eₛ, [eₛ])
		append!(Eᵣ, [eᵣ])
		append!(rₜ, [r])
	end


	for _ in 1:iterations
		for i in 1:p.N
			crossRange = if i == 1
				2:p.N
			elseif i == p.N
				1:(p.N-1)
			else
				vcat(1:(i-1), (i+1):p.N)
			end

			E = sum([E * w[j][end] for (E, j) in zip(Eᵣ[i], crossRange)])

			append!(w[i], [(K[i] + λ * I) \ -(Eₛ[i] .\ (E + rₜ[i]))])
		end
	end

	return w

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
	 
	return -Routput .\ (crossOutput + rOutput)
end
