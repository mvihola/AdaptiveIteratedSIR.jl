dimension = 10

# How many iterations
n_timing = 20_000
n_run = 100_000
n_adaptive = 100

# Maximum power of 2
maxP = 14

# seconds per each N to calclate asymptotic variance estimates:
target_time = dimension*2

include("cardoso.jl")

Nticks = nticks(Ns, 3)
include("../tools/plot_code.jl")

save("mixtureExample_dim$(dimension).pdf", f; pt_per_unit=1)
save("mixtureExample_cost_adapt_trace_dim$(dimension).pdf", f2; pt_per_unit=1)
