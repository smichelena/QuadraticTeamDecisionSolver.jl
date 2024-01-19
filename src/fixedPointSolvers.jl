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


"""
	solverPreprocessing(
		Y::Vector{<:Vector},
		Q::Matrix{<:AbstractMatrix},
		λ::Vector{Float64},
		kernels::Vector{<:Function},
		L::Int,
		K::Int,
		N::Int,
		S::Int,
	)

	Assemble gramians/covariances and pre factorize everything for solver. 
"""
function solverPreprocessing(
	Y::Vector{<:Vector},
	Q::Matrix{<:AbstractMatrix},
	λ::Vector{Float64},
	kernels::Vector{<:Function},
	L::Int,
	K::Int,
	N::Int,
	S::Int,
)
	m   = Int(ceil(S / 2))
	n   = L * N * K
	Y_o = [[Y[i][l:(l+n-1)] for l in 1:n:n*m] for i in eachindex(Y)]
	Y_p = [[Y[i][l:(l+n-1)] for l in m*n:n:(length(Y[i])-n)] for i in eachindex(Y)]
	G_o = [covariance(kernels[i], Y_o[i], Y_o[i]) for i in eachindex(Y_o)]
	G_p = [covariance(kernels[i], Y_o[i], Y_p[i]) for i in eachindex(Y_p)]
	C   = [cholesky(G_o[i] + λ[i] * I) for i in eachindex(G_o)]
	D   = [G_p[i] * (C[i] \ Q[i, i]) for i in axes(Q, 1)]
	Q_f = deepcopy(Q)
	for i in axes(Q, 1)
		Q_f[i, i] =
			BlockDiagonal([D[i][l:(l+L-1), :] for l in 1:L:size(D[1], 1)])
	end
	return G_o, G_p, C, Q_f
end

"""
	optimizedGaussSeidel(
		G_p::Vector{<:Matrix},
		C::Vector,
		Q_f::Matrix{<:AbstractMatrix},
		R::Vector{<:Vector};
		iterations = 10,
	)

Implements a much faster version of the Gauss Seidel solver that uses better data structures and pre factorized gramians.

Other methods can be implemented by using the function `solverPreprocessing` and simply changing the inner loop of this function.

"""
function optimizedGaussSeidel(
	G_p::Vector{<:Matrix},
	C::Vector,
	Q_f::Matrix{<:AbstractMatrix},
	R::Vector{<:Vector};
	iterations = 10,
)
	g = [[zeros(ComplexF64, size(G_p[1], 1))] for _ in eachindex(R)]
	for _ in 1:iterations
		for i in eachindex(R)
			append!(
				g[i],
				[
					-Q_f[i, i] \ (
						G_p[i] * (
							C[i] \ (
								sum([
									Q_f[i, j] * g[j][end] for
									j in setdiff(eachindex(R), i)
								]) + R[i]
							)
						)
					),
				],
			)
		end
	end
	return g
end