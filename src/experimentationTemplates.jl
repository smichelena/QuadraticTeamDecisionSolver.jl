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
	t::teamMMSEproblem,
	samples::Int;
	bandwidth = 0.05,
	δ = 0.9,
	iterations = 10,
	regularizer = 0.5,
	testSamples = 10000,
)
	#train
	_, Y, R, r, _ = generateTeamMMSEsamples(t, samples)
	U = [[rand(ComplexF64, samples)] for _ ∈ 1:t.L]

	k(x, y) = δ * exp(-dot(x - y, x - y) / bandwidth) + (1 - δ) * dot(x, y)
	regression(Y, X) = kernelInterpolation(k, Y, X, λ = regularizer)
	regressor(w, X, x) = kernelFunction(k, w, X, x)

	jacobiPrecodingSolver!(
		t,
		U,
		Y,
		R,
		r,
		regression,
		regressor,
		iterations = iterations,
	)

	W = [
		[regression(U[i][l], Y[i]) for i in 1:t.L] for
		l in 1:iterations
	]

	#test
	_, Yt, Rt, rt, _ = generateTeamMMSEsamples(t, testSamples)
	Ytt = reformatYm(t.L, testSamples, Yt)
	Rtt = reformatR(t.L, testSamples, Rt)
	rtt = reformatr(t.L, testSamples, rt)

	S = [Sample(Y, R, r, 1.0 + 0.0im) for (Y, R, r) in zip(Ytt, Rtt, rtt)]

	return risk(S, [x -> regressor(W[end][i], Y[i], x) for i in 1:t.L])
end
