# Scalar and Vector Kernel Methods

The solvers implemented in this library reproducing kernel Hilbert space methods to approximate the conditional expectation functions that arise in our iterations.

For this reason, we provide a flexible and easy to use interface that can be employed to solve both scalar and vector valued regression problems.

We only provide two kernel definitions, one for the scalar case and one for the vector case, respectively. Nevertheless, the user can easily define their own kernels and use them. We make no requirement on the type and behaviour of a kernel, other that it must be a function.

For more details on the general theory of scalar reproducing kernel Hilbert space methods, please refer to [berlinetKernels](@cite).

For more details on the formulation of vector reproducing kernel Hilbert space methods, please refer to [vRKHS](@cite), for details on approximation properties and universality of these spaces refer to [carmeli2008vector](@cite).

## Functions

```@docs
exponentialKernel(x::Vector, y::Vector; h = 1)
matrixExponentialKernel(h::Vector{Float64}, M::Vector{<:Matrix}, λ::Vector{Float64}, x::Any, t::Any)
gramian(kernel::Function, Y::Vector{<:Vector})
kernelNorm(weights::Vector, kernelGramian::Matrix)
kernelFunction(
  kernel::Function,
  weights::Vector,
  X::Vector{<:Vector},
  x::Vector,
 )
kernelRegression(
  kernel::Function,
  Y::Vector,
  X::Vector;
  λ = 0.5,
 )
```
