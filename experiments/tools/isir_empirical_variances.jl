include("initialSequenceEstimator.jl")
#using MonteCarloMarkovKernels # -> estimateBM

function map_out(f, out, j, s_ind=1:length(out.samples))
    x = [out.samples[i][j] for i=s_ind]
    f(x)
end

function asymptotic_variances_worker(out; burn_q=0.1, ind=nothing, asvar_estimator=initial_sequence_estimator)
    n = length(out.samples)
    @assert 0 <= burn_q < 1
    n0 = Int(ceil(n*burn_q))
    @assert n - n0 >= 2
    s_ind = n0:n
    if isnothing(ind)
        d = length(out.samples[1])
        ind = 1:d
    end
    # Asymptotic variances
    as_vars = [map_out(asvar_estimator, out, j, s_ind) for j in ind]
    # Variances
    vs = [map_out(var, out, j, s_ind) for j in ind]
    iact = as_vars ./ vs
    as_vars, vs, iact
end


function asymptotic_variances(out; burn_q=0.1, ind=nothing)
    as_vars, _, _ = asymptotic_variances_worker(out; burn_q=burn_q, ind=ind)
    as_vars
end


function iact(out; burn_q=0.1, ind=nothing)
    _, _, iact = asymptotic_variances_worker(out; burn_q=burn_q, ind=ind)
    iact
end


# Standardised IREs
function inverse_relative_efficiencies(out; burn_q=0.1, ind=nothing)
    as_vars = asymptotic_variances(out; burn_q=burn_q, ind=ind) 
    if isnothing(ind)
        d = length(out.samples[1])
        ind = 1:d
    end
    vars = [map_out(Statistics.var, out, j) for j in ind]
    
    n = length(out.samples)
    per_sample_cost = out.elapsedTimes[end]/n 
    as_vars./vars * per_sample_cost
end



function approximate_iact(acceptance_rate)
    ϵ = 1-acceptance_rate
    (1+ϵ)/(1-ϵ)
end

function approximate_loss(N, acceptance_rate, cost)
    as_var = approximate_iact(acceptance_rate)
    cost(N)*as_var
end


function asymptotic_variance_function(out, f; burn_q=0.1, asvar_estimator=initial_sequence_estimator)
    n = length(out.samples)
    @assert 0 <= burn_q < 1
    n0 = Int(ceil(n*burn_q))
    @assert n - n0 >= 2
    s_ind = n0:n
    # Asymptotic variances
    x = map(f, out.samples[s_ind])
    as_var = asvar_estimator(x)
    v = var(x)
    iact = as_var/v
    as_var, v, iact
end

function iact_function(out, f; burn_q=0.1)
    as_var, v, iact = asymptotic_variance_function(out, f; burn_q=burn_q)
    iact
end

function ire_function(out, f; burn_q=0.1)
    as_var, v, iact = asymptotic_variance_function(out, f; burn_q=burn_q)
    n = length(out.samples)
    per_sample_cost = out.elapsedTimes[end]/n 
    as_var/v * per_sample_cost
end