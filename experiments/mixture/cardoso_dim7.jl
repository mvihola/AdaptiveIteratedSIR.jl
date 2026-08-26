dimension = 7

# How many iterations
n_timing = 20_000
n_run = 10_000
n_adaptive = 100

# Maximum power of 2
maxP = 13

# seconds per each N to calclate asymptotic variance estimates:
target_time = dimension

include("cardoso.jl")

Nticks = nticks(Ns, 2)
include("../tools/plot_code.jl")

save("mixtureExample_dim$(dimension).pdf", f; pt_per_unit=1)
save("mixtureExample_cost_adapt_trace_dim$(dimension).pdf", f2; pt_per_unit=1)
