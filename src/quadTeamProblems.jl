"""
	QuadTeamProblem{T <: Number}

The `QuadTeamProblem` struct represents a quadratic team decision problem. In essence, it stores the correct problem dimensions and the field over which the problem is solved.

# Fields
- `N::Int`: Number of agents in the team.
- `m::Vector{Int}`: Array of measurement dimensions for each agent.
- `a::Vector{Int}`: Array of action space dimensions for each agent.
- `T::Type{T}`: Numeric type for the problem.
"""
struct QuadTeamProblem{T <: Number}
	N::Int
	m::Vector{Int}
	a::Vector{Int}
	T::Type{T}
end

"""
	checkProblem(p::QuadTeamProblem)

Check the consistency and correctness of a `QuadTeamProblem` object `p`.

# Arguments
- `p::QuadTeamProblem`: The QuadTeamProblem object representing the problem specification.

# Errors
- `AssertionError`: Throws an error if the problem specification is incorrect or inconsistent.

# Example
```julia
P = QuadTeamProblem(...)
checkProblem(P)
```
"""
function checkProblem(p::QuadTeamProblem)

	@assert size(p.m)[1] == p.N && size(p.m)[1] == size(p.a)[1] && size(p.a)[1] == p.N "Problem specification is wrong! sizes dont match. Sizes are: \n" *
																					   "Lengh of array of measurement dimensions: $(size(p.m)[1] ) \n" *
																					   "Lenght of array of action space dimensions: $(size(p.a)[1]) \n Number of agents: $(p.N) "

	return p
end


"""
	checkGamma(p::QuadTeamProblem, γ::Vector{<:Function})

Check the output dimensions of the functions γ for each agent in the QuadTeamProblem `P`.

# Arguments
- `P::QuadTeamProblem`: The QuadTeamProblem specifying the problem.
- `γ::Vector{Function}`: Vector of functions γ for each agent.

# Returns
- `γ::Vector{Function}`: The input vector of functions γ.

# Errors
- Throws an error if the output dimensions of the functions γ do not match the specified dimensions in P.

"""
function checkGamma(p::QuadTeamProblem, γ::Vector{<:Function})
	# Generate random measurement vector with correct dimensions
	Y = [rand(n) for n in p.m]

	for (i, g) in zip(1:p.N, γ)
		result = g(Y[i])
		@assert (ndims(result) == 0 && 1 == p.a[i]) || (size(result)[1] == p.a[i]) "Wrong output dimension for γ^$(i)!" *
																				   " Output dimension is $((size(result) == ()) ? 1 : size(result)[1]) but should be $(p.a[i])"

	end

	return γ
end

"""
	residual(
		m::Int,
		p::QuadTeamProblem,
		kernels::Vector{<:Function},
		γ::Vector{<:Vector{<:Vector}},
		Y::Vector{<:Vector},
		Q::Matrix{<:Vector},
		R::Vector{<:Vector},
		λ::Vector{<:AbstractFloat},
	)

Compute the residual function

```math
\\mathrm{res}(\\gamma) := \\mathbf{A}\\gamma + \\mathbf{\\tilde{R}}
```

for a given tuple of policies ``\\gamma = (\\gamma^1, \\dots, \\gamma^N)``.

# Arguments
- `m::Int`: Number of training samples.
- `p::QuadTeamProblem`: A quad team problem object.
- `kernels::Vector{<:Function}`: Vector of kernel functions.
- `γ::Vector{<:Vector{<:Vector}}`: Vector of gamma kernel function coefficients.
- `Y::Vector{<:Vector}`: Vector of measurement vector samples.
- `Q::Matrix{<:Vector}`: Matrix of system matrix samples, organized block-wise.
- `R::Vector{<:Vector}`: Vector of linear term samples, organized block-wise.

# Returns
- `res`: vector of kernel function coefficient vectors corresponding to residual function.

"""
function residual(
	m::Int,
	p::QuadTeamProblem,
	kernels::Vector{<:Function},
	γ::Vector{<:Vector{<:Vector}},
	Y::Vector{<:Vector},
	Q::Matrix{<:Vector},
	R::Vector{<:Vector},
	λ::Vector{<:AbstractFloat},
)
	#initialize solution
	res = Vector{Vector{Vector{p.T}}}(undef, length(γ))


	#diagonal terms
	fii = [kernelRegression(kernels[i], Q[i, i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fii_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fii[i], Y[i], x) for i ∈ 1:p.N]

	#linear term
	fi = [kernelRegression(kernels[i], R[i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fi_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fi[i], Y[i], x) for i ∈ 1:p.N]

	#gamma samples
	U = [Y[i] .|> x -> kernelFunction(kernels[i], γ[i], Y[i], x) for i in 1:p.N]

	#cross terms
	for i in 1:p.N
		#sample cross terms
		S = [zeros(p.T, p.a[i]) for _ in 1:m]
		for j in setdiff(1:p.N, i)
			S += Q[i, j] .* U[j]
		end

		fij = kernelRegression(kernels[i], S, Y[i], λ = λ[i])
		fij_samples = Y[i] .|> x -> kernelFunction(kernels[i], fij, Y[i], x)

		fiig = [f * u for (f, u) in zip(fii_samples[i], U[i])]

		res_samples = fiig .+ fij_samples .+ fi_samples[i]

		temp = kernelRegression(kernels[i], res_samples, Y[i], λ = λ[i])

		res[i] = temp .|> x -> vcat(x...)
	end

	return res
end

"""
	gammaNorm(f::Function, Y::AbstractVector)

Compute the ``\\Gamma^i`` norm of a given function ``f \\in \\Gamma^i`` approximated 
using measurement vector data, that is, samples of ``\\mathbf{Y}_i``.

# Arguments
- `f::Function`: The function for which the norm is to be computed.
- `Y::AbstractVector`: Input data vector.

# Returns
- `norm`: ``\\Gamma^i``-norm of f.

# Details
The gamma norm is calculated as the squared Euclidean norm of the function's evaluations on the input data, normalized by the length of the input data.

"""
function gammaNorm(f::Function, Y::AbstractVector)
	R = f.(Y)
	return real(mean([r' * r for r in R]))
end

"""
	GammaNorm(F::Vector{<:Function}, Y::AbstractVector)

Compute the ``\\Gamma`` norm of a given function tuple ``F = (f_1, \\dots, f_i) \\in \\Gamma = \\Gamma^i\\times\\dots\\times\\Gamma^N`` 
approximated using measurement vector data, that is, samples of ``\\mathbf{Y} = (\\mathbf{Y}_i,\\dots,\\mathbf{Y}_N)``.

# Arguments
- `F::Vector{<:Function}`: Vector of functions for which the gamma norm is to be computed.
- `Y::AbstractVector`: Input data vector.

# Returns
- `norm`: ``\\Gamma``-norm of F.
"""
function GammaNorm(F::Vector{<:Function}, Y::AbstractVector)
	return sqrt(sum([gammaNorm(f, y) for (f, y) in zip(F, Y)]))
end

"""
	reformatData(Y_data::Vector{<:Vector}, Q_data::Matrix{<:Vector}, R_Data::Vector{<:Vector})

Reformat data to efficient data structure
"""
function reformatData(Y_data::Vector{<:Vector}, Q_data::Matrix{<:Vector}, R_data::Vector{<:Vector})
    Y = [vcat(vec.(Y_data[i])...) for i in eachindex(Y_data)]
    R = [vcat(vec.(R_data[i])...) for i in eachindex(R_data)]
    Q   = [
        i == j ? vcat(Q_data[i, j]...) : BlockDiagonal(Q_data[i, j]) for
        i in axes(Q_data, 1), j in axes(Q_data, 2)
    ]
    return Y, Q, R
end

