using Optim, LogExpFunctions

# log(mean(exp(x))) but numerically safe:
logmeanexp(x) = logsumexp(x) - log(length(x))

function estimate_what(outs, scr, logG)
    n = length(outs)
    logc_hat = zeros(n)
    logw_hat = zeros(n)
    for i = 1:n
        logc_hat[i] = outs[i].normalising_constant.log_v
        opt = optimize(x -> -logG(x, scr), outs[i].samples[end], BFGS())
        logw_hat[i] = -opt.minimum
    end
    what = exp(maximum(logw_hat) - logmeanexp(logc_hat))
    (what = what, logc_hat = logc_hat, logw_hat = logw_hat)
end

