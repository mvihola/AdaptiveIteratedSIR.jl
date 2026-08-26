using Random, DataFrames, CSV, LogExpFunctions, LinearAlgebra, Random, JLD2, ProgressMeter, Statistics

# Iterations for timing:
n_timing = 10_000
# Iterations for sampling:
#n_sample = 1000n_timing
n_sample = 100_000
# How many repetitions
n_runs = 100

# Switch this off for cluster:
show_progress = false

Random.seed!(12345)

# Header
hdr = vcat(["id", "diagnosis"], string.("feature",1:30))
# Read data
df = DataFrame(CSV.File(joinpath(@__DIR__, "wdbc.data"), header=hdr))

# Response to binary:
df.diagnosis = df.diagnosis .== "M" # "M" malignant = 1, "B" bening = 0

# Fit GLM
#wdbc_glm = glm(term(:diagnosis) ~ sum(term.(Symbol.(names(df, Not(:id, :diagnosis))))), df, Bernoulli())
#wdbc_glm = glm(@formula(diagnosis ~ feature1 + feature2 + feature3 + feature5), df, Bernoulli())

# Response variable
response = df.diagnosis
n = length(response)

# Covariates
covariates = hcat(DataFrame(_intercept=ones(n)), df[:,3:end])

# Create scratch space for the model
scratch = (y=response, x=Matrix(covariates), z=ones(n), n=n, prior_sd = sqrt(1/5e-2))

# Common definitions for logistic models:
include("common.jl")
# ...and fitting Laplace approximations:
include("laplace.jl")
# ...and the i-SIR
using AdaptiveIteratedSIR
import AdaptiveIteratedSIR: _adapt_project
#include("../isir_general_state.jl")
#include("../timing.jl")

# Fitted parameters are the initial value for finding Laplace:
#x0 = coef(wdbc_glm)
x0 = zeros(ncol(covariates))

# Find Laplace approximation
LA = find_laplace(log_posterior, x0, scratch)
#LA = find_laplace(log_posterior, zeros(length(x0)), scratch)

# Define function that builds a new scratch space (for the model and Laplace approximation numerics):
newScratch = build_newScratch(deepcopy(LA), () -> deepcopy(scratch), (prob=0.1, draw! = draw_from_prior!))

# Initial value for MCMC:
x = similar(x0)
M_laplace!(x, Random.TaskLocalRNG(), newScratch())


# Ns for timing (and comparison):
#Ns = 2 .^ (2:7) .+ 1
n_threads = Threads.nthreads()
Ns = n_threads * 2 .^ (0:3) .+ 1
N_max = Ns[end]

# Trial runs (for timing)
println("Timing:")
#Ts, β, cost, dcost = timing(Ns, newScratch, x0, n_timing)
tim = isir_timing(M_laplace_safe!, logG_safe, x0, Ns, n_timing; newScratch=newScratch)
Ts = tim.Ts; β = tim.β; cost = tim.cost; dcost = tim.dcost

# Run adaptive i-SIR
println("Adaptive i-SIR:")
#out_adapt = isir(M_laplace_safe!, logG_safe, x, n_sample, N_max; adaptive=true, cost=cost, dcost=dcost, threads=true, newScratch=newScratch, show_progress=show_progress)

Λ = zeros(n_runs, n_sample)
out_adapt = nothing
@showprogress for i = 1:n_runs
    global out_adapt = isir(M_laplace_safe!, logG_safe, x, n_sample, N_max; adaptive=true, cost=cost, dcost=dcost, threads=true, newScratch=newScratch, show_progress=show_progress)
    λ = map(ξ -> begin
        _, λ, _  = _adapt_project.(ξ, Inf)
        λ
    end, out_adapt.ξ)
    Λ[i,:] = λ
end

N_final = round(Int, mean(Λ[:,end]))
#N = map(ξ -> begin 
#   _, _, N  = _adapt_project.(ξ, Inf)
#   N
#end, out_adapt.ξ)
#N_final = N[end]

target_time = out_adapt.elapsedTimes[end]*100

Ns_ = sort(unique(vcat(Ns, N_final)))

n_runs = Int.(ceil.(target_time ./ ((β[1] .+ β[2]*Ns_)/n_timing)))
outs = Vector{Any}(undef, length(Ns_))
println("i-SIR for fixed N:")
@showprogress for (i, N) in enumerate(Ns_)
   outs[i] = isir(M_laplace_safe!, logG_safe, x, n_runs[i], N; adaptive=false, threads=true, newScratch=newScratch, show_progress=show_progress)
end

#jldsave("wdbc_out_$(n_threads)threads.jld2"; out_adapt, outs, Ns_, N_final, β, Ts, Ns)

include("../tools/isir_empirical_variances.jl")

# Approximate loss:
cs = [approximate_loss(Ns_[i], outs[i].acceptance_rate, cost) for i=1:length(Ns_)]

# Approximate IACTs
approx_iact = [approximate_iact(outs[i].acceptance_rate) for i=1:length(outs)]

# Test functions:
first_element(x) = norm(LA.C.L\(x - LA.m)) 
second_element(x) = exp(log_posterior(x, scratch) + 95.0)

IREs_ = hcat(map(out -> ire_function(out, first_element), outs), 
             map(out -> ire_function(out, second_element), outs))'

IACTs_ = hcat(map(out -> iact_function(out, first_element), outs), 
              map(out -> iact_function(out, second_element), outs))'

acc_rates = [outs[i].acceptance_rate for i=eachindex(outs)]

include("../tools/estimate_what.jl")
est_what = estimate_what(outs, newScratch(), logG_safe)
what = est_what.what

jldsave("wdbc_out_$(n_threads)threads.jld2"; cs, approx_iact, IREs_, IACTs_, Ns_, N_final, β, Ts, Ns, acc_rates, out_adapt, Λ, what)
