# -*- coding: utf-8 -*-
from scipy.stats import beta

# Returns the probability that the true probability of the event is less than 
# the parameter p given s successes over t observations
def cdf(s, t, p):
    rv = beta(s+1, t-s+1)
    return rv.cdf(p)

# Returns the probability that the true probability of the event is greater  
# than the parameter p given s successes over t observations
def sf(s, t, p):
    rv = beta(s+1, t-s+1)
    return rv.sf(p)

# Given s successes over t observations, returns a tuple where the probability
# of the true estimated probability that lies within the bounds is 1-alpha.
# Specifically, this is an equal-tailed credible interval
def eq_tail_ci(s, t, alpha): 
    rv = beta(s+1, t-s+1)
    return (rv.ppf(alpha/2), rv.isf(alpha/2))

