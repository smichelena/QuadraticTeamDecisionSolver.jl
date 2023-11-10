"""
	jacobiSolver(
		p::QuadTeamProblem,
		U::Vector{<:Vector},
		Y::Vector{<:Vector},
		R::Matrix{<:Vector},
		r::Vector{<:Vector}
		regression::Function;
		iterations = 5,
	)

Approximately sample the solution to a quadratic team decision problem using a Jacobi iteration scheme.

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
)
	#diagonal terms
	fii = [kernelRegression(kernels[i], Q[i, i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fii_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fii[i], Y[i], x) for i ∈ 1:p.N]

	#linear term
	fi = [kernelRegression(kernels[i], R[i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fi_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fi[i], Y[i], x) for i ∈ 1:p.N]

	#initialize gammas
	U = [[[zeros(p.T, p.a[i]) for _ in 1:m]] for i in 1:p.N]
	γ = [[[zeros(p.T, p.a[i]) for _ in 1:m]] for i in 1:p.N]

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
			append!(U[i], Y[i] .|> x -> kernelFunction(kernels[i], γ[i], Y[i], x))
		end

	end

	return γ

end



"""
	gaussSeidelSolver(
		p::QuadTeamProblem,
		U::Vector{<:Vector},
		Y::Vector{<:Vector},
		R::Matrix{<:Vector},
		r::Vector{<:Vector}
		regression::Function;
		iterations = 5,
	)

Approximately sample the solution to a quadratic team decision problem using a Jacobi iteration scheme.

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
)
	#diagonal terms
	fii = [kernelRegression(kernels[i], Q[i, i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fii_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fii[i], Y[i], x) for i ∈ 1:p.N]

	#linear term
	fi = [kernelRegression(kernels[i], R[i], Y[i], λ = λ[i]) for i ∈ 1:p.N]
	fi_samples = [Y[i] .|> x -> kernelFunction(kernels[i], fi[i], Y[i], x) for i ∈ 1:p.N]

	#initialize gammas
	U = [[[zeros(p.T, p.a[i]) for _ in 1:m]] for i in 1:p.N]
	γ = [[[zeros(p.T, p.a[i]) for _ in 1:m]] for i in 1:p.N]

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
				[
					kernelRegression(
						kernels[i],
						-fii_samples[i] .\ (fi_samples[i] .+ fij_samples),
						Y[i],
						λ = λ[i],
					),
				],
			)
			append!(U[i], Y[i] .|> x -> kernelFunction(kernels[i], [γ[i][end]], Y[i], x))
		end

	end

	return γ

end