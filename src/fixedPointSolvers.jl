using Distributed

"""
	empiricalJacobiSolver!(
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
function empiricalJacobiSolver!(
	L::Int,
	U::Vector{<:Vector},
	Y::Vector{<:Vector},
	R::Matrix{<:Vector},
	r::Vector{<:Vector},
	regression::Function,
	regressor::Function;
	iterations = 5,
)
	wE = [regression(R[i, i], Y[i]) for i ∈ 1:L]
	wr = [regression(r[i], Y[i]) for i ∈ 1:L]
	E = [Y[i] .|> x -> regressor(wE[i], Y[i], x) for i ∈ 1:L]
	hatr = [Y[i] .|> x -> regressor(wr[i], Y[i], x) for i ∈ 1:L]

	for _ ∈ 1:iterations

		temp = []

		for i ∈ 1:L

			crossRange = if i == 1
				2:L
			elseif i == L
				1:(L-1)
			else
				vcat(1:(i-1), (i+1):L)
			end

			S = zeros(ComplexF64, length(r[1])) #think of the type later
			for j in crossRange
				S += R[i, j] .* U[j][end]
			end

			crossWeights = regression(S, Y[i])

			crossSamples = Y[i] .|> x -> regressor(crossWeights, Y[i], x)

			append!(temp, [crossSamples])
		end

		for i ∈ 1:L
			append!(U[i], [-E[i] .\ (temp[i] .+ hatr[i])])
		end

	end

	return U

end

"""
	jacobiPrecodingSolver!(
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
function jacobiPrecodingSolver!(
	p::QuadTeamProblem,
	kernel::Function,
	U::Vector{<:Vector},
	Y::Vector{<:Vector},
	R::Matrix{<:Vector},
	r::Vector{<:Vector};
	iterations = 5,
	λ = 0.01,
	history=true
)

	if history 
		H = Vector{Vector{Vector}}(undef, p.N)
	end

	m = length(R[1, 1])

	range = 1:p.N

	K = [kernelMatrix(kernel, Y[i]) for i in range]

	for iter ∈ 1:iterations

		temp = Vector{Vector}(undef, p.N)

		for i ∈ range #distributed doesnt work yet

			S = [zeros(ComplexF64, p.a[i], p.a[i]) for _ in 1:m]
			for j in setdiff(range, i)
				S += R[i, j] .* U[j]
			end

			crossWeights = kernelRegression((K[i] + λ * I), S) #make regression method handle this inherently, gramian assembly done already for speed

			crossSamples = Y[i] .|> x -> kernelRegressor(kernel, crossWeights, Y[i], x)

			append!(temp, [crossSamples])
		end

		for i ∈ range
			U[i] = -R[i, i] .\ (temp[i] .+ r[i])
			if history
				append!(H[i], kernelRegression((K[i] + λ * I), U[i]))
			end
		end

	end


	if history
		return [kernelRegression((K[i] + λ * I), U[i]) for i in range]
	else
		return H
	end

end

"""
	empiricalAlternatingSolver!(
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
function empiricalAlternatingSolver!(
	p::QuadTeamProblem,
	U::Vector{<:Vector},
	Y::Vector{<:Vector},
	R::Matrix{<:Vector},
	r::Vector{<:Vector},
	regression::Function,
	regressor::Function;
	iterations = 5,
)

	wE = [regression(R[i, i], Y[i]) for i ∈ 1:p.N]
	wr = [regression(r[i], Y[i]) for i ∈ 1:p.N]
	E = [Y[i] .|> x -> regressor(wE[i], Y[i], x) for i ∈ 1:p.N]
	hatr = [Y[i] .|> x -> regressor(wr[i], Y[i], x) for i ∈ 1:p.N]

	for _ ∈ 1:iterations

		for i ∈ 1:p.N

			crossRange = if i == 1
				2:p.N
			elseif i == p.N
				1:(p.N-1)
			else
				vcat(1:(i-1), (i+1):p.N)
			end

			S = zeros(p.T, length(r[1]))
			for j in crossRange
				S += R[i, j] .* U[j][end]
			end

			crossWeights = regression(S, Y[i])

			crossSamples = Y[i] .|> x -> regressor(crossWeights, Y[i], x)

			append!(U[i], [-E[i] .\ (crossSamples .+ hatr[i])])
		end

	end

	return U

end


"""
	alternatingPrecodingSolver!(
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
function alternatingPrecodingSolver!(
	p::QuadTeamProblem,
	U::Vector{<:Vector},
	Y::Vector{<:Vector},
	R::Matrix{<:Vector},
	r::Vector{<:Vector},
	regression::Function,
	regressor::Function;
	iterations = 5,
)
	for _ ∈ 1:iterations

		for i ∈ 1:p.N

			crossRange = if i == 1
				2:p.N
			elseif i == p.N
				1:(p.N-1)
			else
				vcat(1:(i-1), (i+1):p.N)
			end

			S = zeros(p.T, length(r[1]))
			for j in crossRange
				S += R[i, j] .* U[j][end]
			end

			crossWeights = regression(S, Y[i])

			crossSamples = Y[i] .|> x -> regressor(crossWeights, Y[i], x)

			append!(U[i], [-R[i, i] .\ (crossSamples .+ r[i])])
		end

	end

	return U

end
