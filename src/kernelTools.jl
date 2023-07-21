"""
	exponentialKernel(x::Vector, y::Vector; h = 1)

Radial Basis Function Kernel 

# Arguments:
- `x::Vector`: A data sample.
- `x::Vector`: Also a data sample.
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
	gramian(kernel::Function, Y::Vector{<:Vector})

Compute the gramian of a kernel ``K`` over the samples ``\\mathbf{Y} \\subset \\mathcal{Y}``.

# Arguments:
- `kernel::Function`: A positive definite kernel function.
- `Y::Vector{<:Vector}`: The samples over which the gramian is to be constructed.

"""
function gramian(kernel::Function, X::Vector)
	return [kernel(x, y) for x in X, y in X]
end

"""
	kernelNorm(weights::Vector, kernelGramian::Matrix)

Compute the function norm: ``\\sum_{l=1}^m \\sum_{k=1}^m \\alpha_l \\alpha_k K( \\mathbf{y}_l, \\mathbf{y}_k )`` of a function in a reproducing kernel Hilbert space. Th

# Arguments:
- `weights::Vector`: The ``\\alpha``'s in the above expression.
- `kernelGramian::Matrix`: The gramian of ``K`` over the samples ``\\mathbf{Y}``

"""
function kernelNorm(weights::Vector, kernelGramian::Matrix)
	return real(dot(weights, kernelGramian * weights))
end

"""
	kernelFunction(
		kernel::Function,
		weights::Vector,
		Y::Vector{<:Vector},
		x::Vector,
	)

Evaluates a kernel function of the form ``f(\\mathbf{y}) = \\sum_{l=1}^m \\alpha_l K(\\mathbf{y}, \\mathbf{y}_{(l)})``

# Arguments:

- `kernel::Function` Kernel function that defines the RKHS where ``f`` lives.
- `weights`: The ``\\alpha``'s in the above expression.
- `y`: The point ``\\mathbf{y} \\in \\mathcal{Y}`` at which ``f`` is to be evaluated.

"""
function kernelFunction(
	kernel::Function,
	weights::Vector,
	Y::Vector,
	x::Any,
)
	return dot([kernel(x, y) for y in Y], weights)
end

"""
    densityConditionalMean(kernel::Function, Y::Vector, X::Vector, y::Any, h::Float64)

Computes an estimate of a conditonal expectation of the form ``\\mathbb{E}[\\mathbf{X}|\\mathbf{Y} = y]``

"""
function densityConditionalMean(kernel::Function, X::Vector, Y::Vector, y::Any, h::Float64)
	w = [kernel(y, yi; h=h) for yi ∈ Y]
	return dot(w, X) / sum(w)
end


"""
	kernelInterpolation(
		kernel::Function,
		Y::Vector,
		X::Vector;
		λ = 0.5,
	)

Solves the regularized interpolation problem:

```math
\\begin{align*}
		\\text{find: } &f^* \\in \\argmin_{f \\in \\mathcal{H}_k} \\frac{\\lambda}{2} \\lvert \\lvert f \\rvert \\rvert_{\\mathcal{H}_k}^2 \\\\
		\\text{such that: } &f(\\mathbf{y}^{(l)}) = \\mathbf{o}^{(l)}  \\qquad l \\in 1, \\dots, m
\\end{align*}
```
Where ``\\mathcal{H}_k`` is a Reproducing Kernel Hilbert Space with kernel ``k``.

# Arguments:
- `kernel::Function`: The kernel function that corresponds to ``\\mathcal{H}_k``.
- `X::Vector`: Vector of samples in input space ``\\mathcal{X}``.
- `Y::Vector`: Vector of samples in output space ``\\mathcal{Y}``.
- `λ = 0.5`: Regularization constant for ridge regression.

Note that `Y::Vector` and `O::Vector` must have the same length `m`.

"""
function kernelInterpolation(
    kernel::Function,
	Y::Vector,
	X::Vector;
	λ = 0.5,
)
	return (gramian(kernel, X) + λ * I) \ Y
end
