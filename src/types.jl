"""
	QuadTeamProblem{T <: Number}

The `QuadTeamProblem` struct represents a quadratic team decision problem. In essence, it stores the correct problem dimensions and the field over which the problem is solved.

# Fields
- `N::Int`: Number of agents in the team.
- `m::Vector{Int}`: Array of measurement dimensions for each agent.
- `a::Vector{Int}`: Array of action space dimensions for each agent.
- `T::Type{T}`: Numeric type for the problem.
"""
struct QuadTeamProblem{T<:Number}
    N::Int
    m::Vector{Int}
    a::Vector{Int}
    T::Type{T}
end

"""
	Sample{T <: Number}

The `Sample` struct represents a sample data point for the quadratic team decision problem.

# Fields
- `Y::Vector{Vector{T}}`: Array of measurement vectors for each agent, where `Y[i]` represents the measurement vector for agent `i`.
- `R::Matrix{T}`: Quadratic matrix capturing the interaction between agents.
- `r::Vector{T}`: Vector representing the linear term for the control policies.
- `c::T`: Constant term for the objective function.

## Sample Data

In the quadratic team decision problem, a sample represents a specific data point with measurements, quadratic matrices, and linear and constant terms. These samples are used to estimate the performance and optimize the control policies.

The sample data includes the following components:
- ``\\mathbf{Y}``: Array of measurement vectors for each agent, where `Y[i]` represents the measurement vector for agent `i`.
- ``\\mathbf{R}``: Hermitian Positive Definite matrix capturing the interaction between agents. It represents the quadratic term for the control policies.
- ``\\mathbf{r}``: Vector representing the linear term for the control policies.
- ``\\mathbf{c}``: Constant term for the objective function.
"""
struct Sample{T<:Number}
    Y::Vector{Vector{T}}
    R::Matrix{T}
    r::Vector{T}
    c::T
end
