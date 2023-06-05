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

	@assert size(p.m)[1] == p.N && size(p.m)[1] == size(p.a)[1] &&
			size(p.a)[1] == p.N "Problem specification is wrong! sizes dont match. Sizes are: \n" *
								"Lengh of array of measurement dimensions: $(size(p.m)[1] ) \n" *
								"Lenght of array of action space dimensions: $(size(p.a)[1]) \n Number of agents: $(p.N) "

	return p
end

"""
	checkSample(p::QuadTeamProblem, s::Sample)

Check the consistency and correctness of a `Sample` object `s` with respect to a `QuadTeamProblem` `p`.

# Arguments
- `p::QuadTeamProblem`: The QuadTeamProblem object representing the problem specification.
- `s::Sample`: The Sample object to be checked.

# Errors
- `AssertionError`: Throws an error if any inconsistency or mismatch is found in the sample.

# Example
```julia
P = QuadTeamProblem(...)
s = Sample(...)
checkSample(P, s)
```

"""
function checkSample(p::QuadTeamProblem, s::Sample)

	#check types are correct
	@assert eltype(s.c) == p.T "Sample has wrong numeric type: Sample type is $(eltype(s.c))," *
							   " expected type $(p.T)"

	#check measurement sizes are correct
	for (y, n, i) in zip(s.Y, p.m, 1:p.N)
		@assert size(y)[1] == n "measurement for agent $(i) is of wrong size! \n" *
								" size is $(size(y)[1]) but should be $(n)"
	end

	totalSize = sum(p.a)

	#check R 
	@assert size(s.R)[1] == totalSize && size(s.R)[2] == totalSize "Sample of R(X) has wrong dimensions! \n" *
																   "Row dimension is $(size(s.R)[1]) \n" *
																   "Column dimension is $(size(s.R)[2]) \n" *
																   "Both should be $(totalSize) !"


	#check r
	@assert size(s.r)[1] == totalSize "sample of r(X) has wrong dimensions! \n " *
									  "dim is $(size(s.r)[1]) \n " *
									  "but should be $(totalSize) !"

	return s

end

"""
	checkData(p::QuadTeamProblem, S::Vector{<:Sample})

Check the validity of a vector of Sample data for a given QuadTeamProblem `p`.

# Arguments
- `p::QuadTeamProblem`: The QuadTeamProblem specifying the problem.
- `S::Vector{Sample}`: Vector of Sample data to be checked.

# Returns
- `result::Vector{Sample}`: A vector of Sample data with validity checks applied.

"""
function checkData(p::QuadTeamProblem, S::Vector{<:Sample})
	return S .|> (s -> checkSample(p, s))
end

"""
	checkGamma(P::QuadTeamProblem, γ::Vector{<:Function})

Check the output dimensions of the functions γ for each agent in the QuadTeamProblem `P`.

# Arguments
- `P::QuadTeamProblem`: The QuadTeamProblem specifying the problem.
- `γ::Vector{Function}`: Vector of functions γ for each agent.

# Returns
- `γ::Vector{Function}`: The input vector of functions γ.

# Errors
- Throws an error if the output dimensions of the functions γ do not match the specified dimensions in P.

"""
function checkGamma(P::QuadTeamProblem, γ::Vector{<:Function})
	# Generate random measurement vector with correct dimensions
	Y = [rand(n) for n in P.m]

	for (i, g) in zip(1:P.N, γ)
		result = g(Y[i])
		@assert (ndims(result) == 0 && 1 == P.a[i]) ||
				(size(result)[1] == P.a[i]) "Wrong output dimension for γ^$(i)!" *
											" Output dimension is $((size(result) == ()) ? 1 : size(result)[1]) but should be $(P.a[i])"

	end

	return γ
end

using IterTools, LinearAlgebra

"""
	loss(s::Sample, γ::Vector{<:Function})

Compute the loss function for a given Sample `s` using a vector of functions `γ`.

# Arguments
- `s::Sample`: The Sample data.
- `γ::Vector{Function}`: Vector of functions specifying the control policies.

# Returns
- `loss::Real`: The computed loss value.

"""
function loss(s::Sample, γ::Vector{<:Function})
	v = collect(Iterators.flatten([g(y) for (g, y) in zip(γ, s.Y)]))
	return real(dot(v, s.R * v) + 2 * real(dot(v, s.r)) + s.c)
end

"""
	risk(S::Vector{<:Sample}, γ::Vector{<:Function})

Compute the risk function for a given vector of Samples `S` using a vector of functions `γ`.

# Arguments
- `S::Vector{Sample}`: Vector of Sample data.
- `γ::Vector{Function}`: Vector of functions specifying the control policies.

# Returns
- `risk::Real`: The computed risk value.
"""
function risk(S::Vector{<:Sample}, γ::Vector{<:Function})
	return (S .|> (s -> loss(s, γ)) |> sum) / length(S)
end
