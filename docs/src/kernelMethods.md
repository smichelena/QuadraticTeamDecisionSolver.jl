# Kernel Tools

Some tools to set up a solver that uses kernel methods.

# Functions

```@docs
exponentialKernel(x::Vector, y::Vector; h = 1)
gramian(kernel::Function, Y::Vector{<:Vector})
kernelNorm(weights::Vector, kernelGramian::Matrix)
kernelFunction(
		kernel::Function,
		weights::Vector,
		Y::Vector{<:Vector},
		x::Vector,
	)
densityConditionalMean(kernel::Function, Y::Vector, X::Vector, y::Any, h::Float64)
kernelInterpolation(
		kernel::Function,
		Y::Vector,
		O::Vector;
		λ = 0.5,
	)
```