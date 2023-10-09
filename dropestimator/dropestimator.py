# -*- coding: utf-8 -*-
from scipy.stats import beta

# Returns the probability that the true probability of the event is less than 
# the parameter p given s successes over t observations
def cdf(s, t, p):
    """
    Parameters
    ----------
    s : number
        Number of successes observed.
    t : number
        Total number of trials run. (Number of successes plus failures).
    p : number
        Estimate probability

    Returns
    -------
    number
        Returns the probability that the true probabilty is less than p given
        the model and observations
    """
    return beta(s+1, t-s+1).cdf(p)

# Returns the probability that the true probability of the event is greater  
# than the parameter p given s successes over t observations
def sf(s, t, p):
    return beta(s+1, t-s+1).sf(p)

# Given s successes over t observations, returns a tuple where the probability
# of the true estimated probability that lies within the bounds is 1-alpha.
# Specifically, this is an equal-tailed credible interval
def eq_tail_ci(s, t, alpha): 
    rv = beta(s+1, t-s+1)
    return (rv.ppf(alpha/2), rv.isf(alpha/2))

