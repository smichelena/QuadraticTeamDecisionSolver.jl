"""
	assembleSystem(
		N::Int,
		Y::Vector{<:Vector},
		R::Matrix{<:Vector},
		r::Vector{<:Vector},
		conditionalMean::Function,
	)

Assembles and returns various blocks of a system.

"""
function assembleSystem(
	N::Int,
	Y::Vector{<:Vector},
	R::Matrix{<:Vector},
	r::Vector{<:Vector},
	conditionalMean::Function,
)
	E = Vector{Vector}(undef, N)
	hatr = Vector{Vector}(undef, N)
	for i in 1:N
		E[i] = [conditionalMean(R[i, i], Y[i], y) for y in Y[i]]
		hatr[i] = [conditionalMean(r[i], Y[i], y) for y in Y[i]]
	end

	return E, hatr

end

"""
	empiricalJacobiSolver!(
		p::QuadTeamProblem,
		w::Vector{<:Vector},
		Y::Vector{<:Vector},
		R::Matrix{<:Vector},
		r::Vector{<:Vector},
		conditionalMean::Function;
		iterations = 5,
	)

Approximately sample the solution to a quadratic team decision problem using a Jacobi iteration scheme.

"""
function empiricalJacobiSolver!(
	p::QuadTeamProblem,
	w::Vector{<:Vector},
	Y::Vector{<:Vector},
	R::Matrix{<:Vector},
	r::Vector{<:Vector};
	iterations = 5,
	h = 1.0,
	λ = 0.1
)

	for _ in 1:iterations 

		temp = []

		for i in 1:p.N

			crossRange = if i == 1
				2:p.N
			elseif i == p.N
				1:(p.N-1)
			else
				vcat(1:(i-1), (i+1):p.N)
			end

			S = zeros(ComplexF64, length(r[1])) 
			for j in crossRange
				S += R[i, j] .* w[j][end]
			end

			kernel(x,y) = exponentialKernel(x, y, h=h) 

			crossWeights = kernelInterpolation(kernel, S, Y[i], λ=λ)

			crossSamples = Y[i] .|> y -> kernelFunction(kernel, crossWeights, Y[i], y)

			append!(temp, [crossSamples])
		end

		for i in 1:p.N
			append!(w[i], [-R[i,i] .\ (temp[i] .+ r[i])])
		end

	end

	return w

end


"""
	empiricalJacobiSolver!(
		p::QuadTeamProblem,
		w::Vector{<:Vector},
		Y::Vector{<:Vector},
		R::Matrix{<:Vector},
		r::Vector{<:Vector},
		conditionalMean::Function;
		iterations = 5,
	)

Approximately sample the solution to a quadratic team decision problem using an alternating iteration scheme.

"""
function empiricalAlternatingSolver!(
	p::QuadTeamProblem,
	w::Vector{<:Vector},
	Y::Vector{<:Vector},
	R::Matrix{<:Vector},
	r::Vector{<:Vector},
	conditionalMean::Function;
	iterations = 5,
)

	E, hatr = assembleSystem(p.N, Y, R, r, conditionalMean)

	for k in 1:iterations 

		for i in 1:p.N

			crossRange = if i == 1
				2:p.N
			elseif i == p.N
				1:(p.N-1)
			else
				vcat(1:(i-1), (i+1):p.N)
			end

			crossSamples(y) = sum([conditionalMean(R[i, j] .* w[j][end] , Y[i], y) for j in crossRange])

			append!(w[i], [-E[i] .\ (crossSamples.(Y[i]) .+ hatr[i])])
		end

	end

	return w

end