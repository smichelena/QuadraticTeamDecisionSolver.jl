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

Additionally checks samples of ``\\mathbf{R}(\\mathbf{X})`` are Hermitian and positive definite.

# Arguments
- `p::QuadTeamProblem`: The QuadTeamProblem object representing the problem specification.
- `s::Sample`: The Sample object to be checked.

# Errors
- `AssertionError`: Throws an error if any inconsistency or mismatch is found in the sample, or if R isn't positive definite 

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

	#check R has correct dimensions
	@assert size(s.R)[1] == totalSize && size(s.R)[2] == totalSize "Sample of R(X) has wrong dimensions! \n" *
																   "Row dimension is $(size(s.R)[1]) \n" *
																   "Column dimension is $(size(s.R)[2]) \n" *
																   "Both should be $(totalSize) !"

	#check R is positive definite and Hermitian/Symettric
	@assert eigvals(s.R) .|> (x -> isapprox(imag(x), 0) ? real(x) : -1) |>
			vec -> all(x -> x > 0, vec) "R(X) is not positive definite!"

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

"""
	splitSampleIntoBlocks(p::QuadTeamProblem, s::Sample)

Split a sample `s` into blocks based on the dimensions defined in the `QuadTeamProblem` `p`.

# Arguments
- `p::QuadTeamProblem`: A `QuadTeamProblem` object defining the dimensions of the blocks.
- `s::Sample`: A `Sample` object to be split into blocks.

# Returns
- `R_blocks::Vector{SubArray}`: An array of subarrays containing the blocks of `s.R` based on the block dimensions defined in `p`.
- `r_blocks::Vector{SubArray}`: An array of subarrays containing the blocks of `s.r` based on the block dimensions defined in `p`.

"""
function splitSampleIntoBlocks(p::QuadTeamProblem, s::Sample)
	beginnings = accumulate(+, vcat(1, p.a)[1:end-1])
	endings = accumulate(+, p.a)
	R_blocks = [
		s.R[a:b, c:d] for (a, b) in zip(beginnings, endings),
		(c, d) in zip(beginnings, endings)
	]
	r_blocks = [s.r[a:b] for (a, b) in zip(beginnings, endings)]
	return R_blocks, r_blocks
end

"""
	splitDataSetIntoBlocks(p::QuadTeamProblem, S::Vector{<:Sample})

Split a vector of samples `S` into blocks based on the dimensions defined in the `QuadTeamProblem` `p`.

# Arguments
- `p::QuadTeamProblem`: A `QuadTeamProblem` object defining the dimensions of the blocks.
- `S::Vector{<:Sample}`: A vector of `Sample` objects to be split into blocks.

# Returns
- `Y`: vector of length `N` (number of agents) of vectors `Y[i]` of length `m` (number of samples).
Each entry of `Y` is the vector of measurement vectors that correspond to the agent `i`.
- `R_blocks`: vector of length `N` (number of agents) of vectors `R` of length `m` (number of samples). 
Each entry of `R`, `R[i]` is a vector of the blocks of `R` that correspond to the agent `i` 
- `r_blocks`: vector of length `N` (number of agents) of vectors `r` of length `m` (nummber of samples).
Each entry of `r`, `r[i]` is a vector of the block of `r` that corresponds to the agent `i`

"""
function splitDataSetIntoBlocks(p::QuadTeamProblem, S::Vector{<:Sample})
	split = S .|> s -> splitSampleIntoBlocks(p, s)
	Y = [s.Y for s in S]
	#vector of tuples to tuple of vectors 
	splitR = [split[i][1] for i in 1:length(S)] 
	splitr = [split[i][2] for i in 1:length(S)]
	#reorganize into samples per agent
	return [[y[i] for y in Y] for i in 1:p.N], [[R[i,:] for R in splitR] for i in 1:p.N], [[r[i] for r in splitr] for i in 1:p.N]
end

function uloss(u::Vector, R::Matrix, r::Vector)
	s = R\r
	return real(dot(u + s, R, u + s))
end

function urisk(Urange::Vector{<:Vector}, Rrange::Vector{<:Matrix}, rrange::Vector{<:Vector})
    return sum([uloss(u, R, r) for (u, R, r) in zip(Urange, Rrange, rrange)])/length(Urange)
end

function reformatR(N::Int, m::Int, R::Matrix{<:Vector})
    return [vcat([hcat([R[i,j][l] for i in 1:N]...) for j in 1:N]...) for l in 1:m]
end

function reformatr(N::Int, m::Int, r::Vector{<:Vector})
    return [vcat([r[i][l] for i in 1:N]...) for l in 1:m]
end

function reformatW(N::Int, m::Int, iterations::Int, w::Vector{<:Vector})
    return [[vcat([w[i][k][l] for i in 1:N]...) for l in 1:m] for k in 1:iterations]
end