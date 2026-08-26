using Optim, LogExpFunctions

# log(mean(exp(x))) but numerically safe:
logmeanexp(x) = logsumexp(x) - log(length(x))

function estimate_what(outs, scr, logG)
    n = length(outs)
    log_wu = Vector{Vector{Float64}}(undef, n)
    logc_hat = zeros(n)
    logw_hat = zeros(n)
    for i = 1:n
        log_wu[i] = [logG(x, scr) for x in outs[i].samples]
        logc_hat[i] = logmeanexp(log_wu[i])
        opt = optimize(x -> -logG(x, scr), outs[i].samples[end], BFGS())
        logw_hat[i] = -opt.minimum
    end
    what = exp(maximum(logw_hat) - logmeanexp(vcat(log_wu...)))
    (what = what, logc_hat = logc_hat, logw_hat = logw_hat, log_wu = log_wu)
end

