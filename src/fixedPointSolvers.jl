"""
	jacobiSolver(
		p::QuadTeamProblem,
		m::Int,
		Y::Vector{<:Vector},
		Q::Matrix{<:Vector},
		R::Vector{<:Vector},
		kernels::Vector{<:Function},
		λ::Vector{Float64};
		iterations = 5,
		random_init = false,`
	)

Implements the Jacobi operator splitting method to generate policy updates.

Converges to the unique team optimal ``\\gamma^*`` whenever
	
```math
\\lambda_{\\mathrm{min}}  > \\varrho(\\mathbf{T})
```

"""
function jacobiSolver(
	p::QuadTeamProblem,
	m::Int,
	Y::Vector{<:Vector},
	Q::Matrix{<:Vector},
	R::Vector{<:Vector},
	kernels::Vector{<:Function},
	λ::Vector{Float64};
	iterations = 5,
	random_init = false,
)
	#diagonal terms
	fii = [kernelRegression(kernels[i], Q[i, i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fii_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fii[i], Y[i], x) for i ∈ 1:p.N]

	#linear term
	fi = [kernelRegression(kernels[i], R[i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fi_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fi[i], Y[i], x) for i ∈ 1:p.N]

	#initialize gammas
	γ = [[[random_init ? rand(p.T, p.a[i]) : zeros(p.T, p.a[i]) for _ in 1:m]] for i in 1:p.N]
	U = [[Y[i] .|> x -> kernelFunction(kernels[i], γ[i][end], Y[i], x)] for i in 1:p.N]

	for _ ∈ 1:iterations

		temp = []

		for i ∈ 1:p.N

			#sample cross terms
			S = [zeros(p.T, p.a[i]) for _ in 1:m]
			for j in setdiff(1:p.N, i)
				S += Q[i, j] .* U[j][end]
			end

			fij = kernelRegression(kernels[i], S, Y[i], λ = λ[i])
			fij_samples = Y[i] .|> x -> kernelFunction(kernels[i], fij, Y[i], x)

			append!(temp, [-fii_samples[i] .\ (fij_samples .+ fi_samples[i])])
		end

		for i ∈ 1:p.N
			append!(γ[i], [kernelRegression(kernels[i], temp[i], Y[i], λ = λ[i])])
			append!(U[i], [Y[i] .|> x -> kernelFunction(kernels[i], γ[i][end], Y[i], x)])
		end

	end

	return γ

end



"""
	gaussSeidelSolver(
		p::QuadTeamProblem,
		m::Int,
		Y::Vector{<:Vector},
		Q::Matrix{<:Vector},
		R::Vector{<:Vector},
		kernels::Vector{<:Function},
		λ::Vector{Float64};
		iterations = 5,
		random_init = false,
	)

Implements the Gauss-Seidel operator splitting method to generate policy updates.

Guaranteed to converge to the unique team optimal ``\\gamma^*``.

"""
function gaussSeidelSolver(
	p::QuadTeamProblem,
	m::Int,
	Y::Vector{<:Vector},
	Q::Matrix{<:Vector},
	R::Vector{<:Vector},
	kernels::Vector{<:Function},
	λ::Vector{Float64};
	iterations = 5,
	random_init = false,
)
	#diagonal terms
	fii = [kernelRegression(kernels[i], Q[i, i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fii_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fii[i], Y[i], x) for i ∈ 1:p.N]

	#linear term
	fi = [kernelRegression(kernels[i], R[i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fi_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fi[i], Y[i], x) for i ∈ 1:p.N]

	#initialize gammas
	γ = [[[random_init ? rand(p.T, p.a[i]) : zeros(p.T, p.a[i]) for _ in 1:m]] for i in 1:p.N]
	U = [[Y[i] .|> x -> kernelFunction(kernels[i], γ[i][end], Y[i], x)] for i in 1:p.N]

	for _ ∈ 1:iterations

		for i ∈ 1:p.N

			#sample cross terms
			S = [zeros(p.T, p.a[i]) for _ in 1:m]
			for j in setdiff(1:p.N, i)
				S += Q[i, j] .* U[j][end]
			end

			fij = kernelRegression(kernels[i], S, Y[i], λ = λ[i])
			fij_samples = Y[i] .|> x -> kernelFunction(kernels[i], fij, Y[i], x)

			append!(
				γ[i],
				[kernelRegression(
					kernels[i],
					-fii_samples[i] .\ (fi_samples[i] .+ fij_samples),
					Y[i],
					λ = λ[i],
				)],
			)
			append!(U[i], [Y[i] .|> x -> kernelFunction(kernels[i], γ[i][end], Y[i], x)])
		end

	end

	return γ

end

"""
	SORSolver(
		p::QuadTeamProblem,
		m::Int,
		Y::Vector{<:Vector},
		Q::Matrix{<:Vector},
		R::Vector{<:Vector},
		kernels::Vector{<:Function},
		λ::Vector{Float64};
		iterations = 5,
		random_init = false,
		omega = 1.0,
	)

Implements the Successive-Over-Relaxation (SOR) operator splitting method to generate policy updates.

Converges to the unique team optimal ``\\gamma^*`` for all ``\\omega \\neq 0`` that fulfill 

```math
	1 - \\frac{\\lambda_{\\mathrm{min}}}{\\varrho(\\mathbf{T})} < \\omega < 1 + \\frac{\\lambda_{\\mathrm{min}}}{\\varrho(\\mathbf{T})}
```

"""
function SORSolver(
	p::QuadTeamProblem,
	m::Int,
	Y::Vector{<:Vector},
	Q::Matrix{<:Vector},
	R::Vector{<:Vector},
	kernels::Vector{<:Function},
	λ::Vector{Float64};
	iterations = 5,
	random_init = false,
	omega = 1.0,
)
	#diagonal terms
	fii = [kernelRegression(kernels[i], Q[i, i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fii_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fii[i], Y[i], x) for i ∈ 1:p.N]

	#linear term
	fi = [kernelRegression(kernels[i], R[i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fi_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fi[i], Y[i], x) for i ∈ 1:p.N]

	#initialize gammas
	γ = [[[random_init ? rand(p.T, p.a[i]) : zeros(p.T, p.a[i]) for _ in 1:m]] for i in 1:p.N]
	U = [[Y[i] .|> x -> kernelFunction(kernels[i], γ[i][end], Y[i], x)] for i in 1:p.N]

	for _ ∈ 1:iterations

		for i ∈ 1:p.N

			#sample cross terms
			S = [zeros(p.T, p.a[i]) for _ in 1:m]
			for j in setdiff(1:p.N, i)
				S += Q[i, j] .* U[j][end]
			end

			fij = kernelRegression(kernels[i], S, Y[i], λ = λ[i])
			fij_samples = Y[i] .|> x -> kernelFunction(kernels[i], fij, Y[i], x)

			append!(
				γ[i],
				[kernelRegression(
					kernels[i],
					-omega * fii_samples[i] .\ (fi_samples[i] .+ fij_samples) .+ (1 - omega) * U[i][end],
					Y[i],
					λ = λ[i],
				)],
			)
			append!(U[i], [Y[i] .|> x -> kernelFunction(kernels[i], γ[i][end], Y[i], x)])
		end

	end

	return (1 / omega) * γ

end
