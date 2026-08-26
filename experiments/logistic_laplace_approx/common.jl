# Prior N(0, sd^2 I) where sd is:

# Return (normalised) prior log density values:
function log_prior(θ, scr; normalised=false) 
    if normalised
        # Normalisation constant: -.5log(2*pi) - 
        C = -0.9189385332046727 - length(θ)*log(scr.prior_sd) 
    else 
        C = 0.0
    end
    C -.5*mapreduce(x -> x^2, +, θ)/scr.prior_sd^2
end

# Calculate log-likelihood of logistic regression
function log_lik(θ, scratch)
    mul!(scratch.z, scratch.x, θ)
    L = 0.0
    for i = 1:scratch.n
        if scratch.y[i] == 1
            L += loglogistic(scratch.z[i]) # log(logistic(x))
        else
             L += log1mlogistic(scratch.z[i]) # log(1 - logistic(x))
        end
    end
    L
end

# Log-posterior is just this:
log_posterior(θ, scratch) = log_prior(θ, scratch; normalised=true) + log_lik(θ, scratch)

# Draw from the prior:
function draw_from_prior!(x, rng, s) 
    randn!(rng, x)
    x .*= s.prior_sd
    nothing
end

function logG_safe(x_, scratch) 
    L_LA = log(1.0 - scratch.prior.prob) + logDensity_laplace(x_, scratch)
    L_pr = log(scratch.prior.prob) + log_prior(x_, scratch.targetScratch; normalised=true)
    L = logsumexp(L_LA, L_pr)
    log_posterior(x_, scratch.targetScratch) - L 
end

