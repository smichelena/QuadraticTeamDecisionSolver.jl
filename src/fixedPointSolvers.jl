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
    p::QuadTeamProblem,
    U::Vector{<:Vector},
    Y::Vector{<:Vector},
    R::Matrix{<:Vector},
    r::Vector{<:Vector},
    regression::Function,
    regressor::Function;
    iterations = 5,
)
    wE = [regression(R[i, i], Y[i]) for i = 1:p.N]
    wr = [regression(r[i], Y[i]) for i = 1:p.N]
    E = [Y[i] .|> x -> regressor(wE[i], Y[i], x) for i ∈ 1:p.N]
    hatr = [Y[i] .|> x -> regressor(wr[i], Y[i], x) for i ∈ 1:p.N]

    for _ ∈ 1:iterations

        temp = []

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

            append!(temp, [crossSamples])
        end

        for i ∈ 1:p.N
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
    U::Vector{<:Vector},
    Y::Vector{<:Vector},
    R::Matrix{<:Vector},
    r::Vector{<:Vector},
    regression::Function,
    regressor::Function;
    iterations = 5,
)
    for _ ∈ 1:iterations

        temp = []

        for i ∈ 1:p.N #distributed doesnt work yet

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

            append!(temp, [crossSamples])
        end

        for i ∈ 1:p.N
            append!(U[i], [-R[i, i] .\ (temp[i] .+ r[i])])
        end

    end

    return U

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

    wE = [regression(R[i, i], Y[i]) for i = 1:p.N]
    wr = [regression(r[i], Y[i]) for i = 1:p.N]
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
