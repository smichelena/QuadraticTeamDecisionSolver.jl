"""
	exponentialKernel(x::Vector, y::Vector; h = 1)

Radial Basis Function Kernel 

# Arguments:
- `x::Vector`: A data sample.
- `y::Vector`: Also a data sample.
- `h=1`: Width of the kernel.

# Returns:
- Value of the kernel computed at `x` and `y` with window width `h`, 
that is, ``\\exp \\left( - \\frac{|x - y|^2}{h} \\right)`` where ``| \\ \\cdot \\ |`` is the norm 
on the original sample space.

"""
function exponentialKernel(x::Any, y::Any; h = 1)
	return exp(-real((norm(x - y)^2) / h))
end


"""
	matrixExponentialKernel(h::Vector{Float64}, M::Vector{<:Matrix}, λ::Vector{Float64}, x::Any, t::Any)

Matrix Valued Gaussian Mixture Kernel 

# Arguments:
- `h::Vector{Float64}`: Vector of kernel bandwidths ``\\sigma_i, i \\in \\{ 1, \\dots, L \\}``.
- `M::Vector{<:Matrix}`: Vector of mixture matrices ``M_i, i \\in \\{ 1, \\dots, L \\}``.
- `λ::Vector{Float64}`: Vector of convex combination coefficients ``\\lambda_i,i \\in \\{ 1, \\dots, L \\}^2, \\ \\sum_{i=1}^L\\lambda_i = 1``.
- `x::Vector`: A data sample ``x \\in \\mathcal{X}``.
- `x::Vector`: Also a data sample.

# Returns:
- Value of matrix kernel computed at `x` and `y` with specified input parameters.
```math
	K(x,t) = \\sum_{i=1}^L \\lambda_ie^{-\\sigma_i^{-1}d_{\\mathcal{X}}(x,y)^2}M_i
```

"""
function matrixExponentialKernel(
	h::Vector{Float64},
	M::Vector{<:Matrix},
	λ::Vector{Float64},
	x::Any,
	t::Any,
)
	return sum([
		a * exponentialKernel(x, t; h = l) * B for (a, l, B) in zip(λ, h, M)
	])
end


"""
	gramian(kernel::Function, X::Vector{<:Vector})

Compute the gramian of a kernel ``K`` over the samples ``\\mathbf{X}^n \\subset \\mathcal{X}``.

# Arguments:
- `kernel::Function`: A positive definite kernel function.
- `X::Vector{<:Vector}`: The samples over which the gramian is to be constructed.

"""
function gramian(kernel::Function, X::AbstractVector)
	return [kernel(x, y) for x in X, y in X]
end

"""
	kernelNorm(weights::Vector, kernelGramian::Matrix)

Compute the function norm: ``\\sum_{i=1}^n \\sum_{j=1}^n \\mu_i \\mu_j K( x^{(i)}, x^{(j)} )`` of a function in a reproducing kernel Hilbert space. Th

# Arguments:
- `weights::Vector`: The ``\\mu``'s in the above expression.
- `kernelGramian::Matrix`: The gramian of ``K`` over the samples ``\\mathbf{X}^n``

"""
function kernelNorm(weights::Vector, kernelGramian::Matrix)
	G = hvcat((size(kernelGramian, 1)), kernelGramian...) #flatten kernel matrix
	μ = vcat(weights...) #flatten weight vector
	return real(dot(μ, G * μ))
end

"""
	kernelFunction(
		kernel::Function,
		weights::Vector,
		X::Vector{<:Vector},
		x::Vector,
	)

Evaluates a kernel function of the form ``f(x) = \\sum_{i=1}^n \\mu_i K(x, x^{(i)})``

# Arguments:

- `kernel::Function` scalar or matrix valued Kernel function that defines the RKHS where ``f`` lives.
- `weights`: The ``\\mu``'s in the above expression.
- `x`: The point ``\\mathbf{x} \\in \\mathcal{X}`` at which ``f`` is to be evaluated.

"""
function kernelFunction(kernel::Function, weights::Vector, X::Vector, x::Any)
	return sum([kernel(x, t) * a for (t, a) in zip(X, weights)])
end

"""
	kernelRegression(
		kernel::Function,
		Y::Vector,
		X::Vector;
		λ = 0.5,
	)

Solves the kernel regression problem

```math
f^* = \\argmin_{f \\in \\mathcal{H}_k} \\sum_{i=1}^n \\lvert \\lvert f(x^{(i)}) - y^{(i)} \\rvert \\rvert_{\\mathcal{Y}}^2 + \\lambda \\lvert \\lvert f \\rvert \\rvert_{\\mathcal{H}_k}^2 
```
Where ``\\{x^{(i)}, y^{(i)} \\}_{i=1}^n \\subset \\mathcal{X}\\times\\mathcal{Y}`` are data samples, and ``\\mathcal{H}_k`` is a Reproducing Kernel Hilbert Space with kernel ``k``.

Moreover, this implementation can handle the case ``\\mathcal{Y} = \\mathbb{C}^d`` for ``d > 1``. That is, this implements vector valued kernel regression, as well as scalar valued kernel regression.

# Arguments:
- `kernel::Function`: The kernel function that corresponds to ``\\mathcal{H}_k``.
- `X::Vector`: Vector of samples in input space ``\\mathcal{X}``.
- `Y::Vector`: Vector of samples in output space ``\\mathcal{Y}``.
- `λ = 0.5`: Regularization constant.

Note that `Y::Vector` and `X::Vector` must have the same length `n`.

"""
function kernelRegression(
	k::Function,
	Y::AbstractVector,
	X::AbstractVector;
	λ = 0.5,
)
	#compute gramian and flatten it
	kernelGramian = gramian(k, X)
	n = size(kernelGramian[1, 1], 1)
	m = size(kernelGramian, 2)
	G = hvcat((m), kernelGramian...)

	#flatten observations vector
	Yc = vcat(Y...)

	#solve linear system
	μ = (G + λ * I) \ Yc

	#rearrange weights vector
	if ndims(Y[1]) > 1
		return [μ[i:i+n-1, :] for i in 1:n:m*n]
	else
		return [μ[i:i+n-1] for i in 1:n:m*n]
	end
end
