using QuadraticTeamDecisionSolver, Distributions, LinearAlgebra

function sinrExperiment(
	p::QuadTeamProblem,
	solver!::Function,
	sinr::AbstractFloat;
	samples = 1000,
	iterations = 10,
	bandwidth = 0.5,
	regularizer = 0.0,
	eps = 0.0,
)
	t = teamMMSEproblem(
		sinr,
		zeros(Float64, p.N),
		ones(Float64, p.N),
		zeros(Float64, p.N),
		0.5 * ones(Float64, p.N),
		eps * ones(Float64, p.N),
	)
	Y, R, r = generateTeamMMSEsamples(p, t, samples)
	tR = reformatR(p.N, samples, R)
	tr = reformatr(p.N, samples, r)
	U = [[rand(ComplexF64, samples)] for _ ∈ 1:p.N]
	solver!(
		p,
		U,
		Y,
		R,
		r,
		iterations = iterations,
		h = bandwidth,
		λ = regularizer,
	)
	wtest = reformatU(p.N, samples, iterations, U)
	return [urisk(wtest[k], tR, tr) for k ∈ 1:iterations]
end

function bandwidthExperiment(
	p::QuadTeamProblem,
	solver!::Function,
	bandwidth::AbstractFloat;
	samples = 1000,
	iterations = 10,
	sinr = 1.5,
	regularizer = 0.0,
	eps = 0.0,
)
	t = teamMMSEproblem(
		sinr,
		zeros(Float64, p.N),
		ones(Float64, p.N),
		zeros(Float64, p.N),
		0.5 * ones(Float64, p.N),
		eps * ones(Float64, p.N),
	)
	Y, R, r = generateTeamMMSEsamples(p, t, samples)
	tR = reformatR(p.N, samples, R)
	tr = reformatr(p.N, samples, r)
	w = [[rand(ComplexF64, samples)] for _ ∈ 1:p.N]
	solver!(
		p,
		w,
		Y,
		R,
		r,
		iterations = iterations,
		h = bandwidth,
		λ = regularizer,
	)
	wtest = reformatU(p.N, samples, iterations, w)
	return [urisk(wtest[k], tR, tr) for k ∈ 1:iterations]
end

function samplesExperiment(
	p::QuadTeamProblem,
	samples::Int;
	bandwidth = 0.05,
	iterations = 10,
	sinr = 1.5,
	regularizer = 0.5,
	eps = 0.0,
	testSamples = 10000,
)
	t = teamMMSEproblem(
		sinr,
		zeros(Float64, p.N),
		ones(Float64, p.N),
		zeros(Float64, p.N),
		0.5 * ones(Float64, p.N),
		eps * ones(Float64, p.N),
	)
	#train
	Y, R, r = generateTeamMMSEsamples(p, t, samples)
	U = [[rand(ComplexF64, samples)] for _ ∈ 1:p.N]

	k(x, y) = exp((-norm(x - y)^2) / bandwidth)
	regression(Y, X) = kernelInterpolation(k, Y, X, λ = regularizer)
	regressor(w, X, x) = kernelFunction(k, w, X, x)

	empiricalJacobiSolver!(
		p,
		U,
		Y,
		R,
		r,
		regression,
		regressor,
		iterations = iterations,
	)

	W = [
		[kernelInterpolation(k, U[i][l], Y[i], λ = regularizer) for i in 1:p.N] for
		l in 1:iterations
	]

	#test
	Yt, Rt, rt = generateTeamMMSEsamples(p, t, testSamples)
	Ytt = reformatYm(p.N, testSamples, Yt)
	Rtt = reformatR(p.N, testSamples, Rt)
	rtt = reformatr(p.N, testSamples, rt)

	S = [Sample(Y, R, r, 1.0+0.0im) for (Y, R, r) in zip(Ytt, Rtt, rtt)]

	result = [
		risk(S, [x -> kernelFunction(k, W[i][l], Y[i], x) for i in 1:p.N]) for
		l in 1:iterations
	]

	return result
end
