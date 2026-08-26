# AdaptiveIteratedSIR.jl

This is a module which implements an iterated sampling importance resampling (i-SIR) Markov chain Monte Carlo algorithm. The implementation finds an appropriate number of particles automatically.

## Run the experiments in the paper

Before running the experiments, you need to clone this repository to your compute and launch Julia with four threads in the folder by calling:

```sh
julia --threads 4
```

Then, the following commands in Julia should install the required dependencies:

```julia
using Pkg
Pkg.activate("experiments")
Pkg.instantiate()
```

After those, you can run the experiments using the following commands:

```julia
include("experiments/mixture/cardoso_dim7.jl")
include("experiments/mixture/cardoso_dim10.jl")
include("experiments/logistic_laplace_approx/wdbc_run.jl")
include("experiments/logistic_laplace_approx/wdbc_analysis.jl")
```

## Install the package

If you want to use the package for your own model, you can install the package in Julia as follows:

```julia
using Pkg
Pkg.add(url="https://github.com/mvihola/AdaptiveIteratedSIR.jl")
```

## Quick start

The adaptive i-SIR is used for sampling from the 
distribution of the form:
$$p(x) \propto q(x)w(x),$$
where $q$ is the proposal distribution (which dominates $p$) and $w(x) \propto \frac{dp}{dq}(x)$ is the corresponding (unnormalised) importance weight.

The necessary ingredients in the implementation are as follows:

* `q!`: Function which simulates a realisation from the distribution: $q$: `q!(x, rng, scratch)` simulates a realisation of $x\sim q$ using random number generator `rng` and scratch space `scratch`.
* `log_w`: Function which returns logarithmic value of `w(x)`: `log_w(x, scratch)`
* `x0`: Initial state, with a function `similar` which creates a new (mutable) variable of type `x0` (e.g.  `Vector`)

The `scratch` can be any data structure that contains temporary variables for your model. By default, `scratch` is `nothing`.

Here's a toy example which samples $\chi^2$ distribution using $N(0,1)$ proposals.

```julia
using AdaptiveIteratedSIR, Distributions, Random
Random.seed!(12345)

# Function that mutates x at time k based on previous x_prev,
# using random number generator rng and scratch 
# This corresponds to random walk model with N(0,1) increments
function q!(x, rng, _)
    rand!(rng, Cauchy(), x)
end

function log_w(x, _)
    logpdf(Normal(), x[]) - logpdf(Cauchy(), x[])
end

# Vector of N-values to use in timing:
Ns = 4 .^ (0:4) .+ 1
# Number of iterations in timing:
n_timing = 1000
x0 = [0.0]
# Do the timing:
timing = isir_timing(q!, log_w, x0, Ns, n_timing)

n_sample = 10_000
N_max = Ns[end]

out = isir(q!, log_w, x0, n_sample, N_max; adaptive=true, cost=timing.cost, dcost=timing.dcost)
```

To see the histogram of the simulated samples:

```julia
using Plots
Xs = vcat(out.samples...)
histogram(Xs)
```
