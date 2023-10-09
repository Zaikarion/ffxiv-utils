# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:48:33 2023
"""

import scipy.stats as st
import numpy as np
from scipy.signal import fftconvolve

def discrete_pfm_iter(dist):
    i = 0
    while True:
        yield dist.pmf(i)
        i += 1
        
def pfm_iter_to_array(iter, n):
    return np.fromiter(iter, float, n+1)

def dist_convolve(a,b):
    n = len(a)
    if len(b) != n:
        raise Exception("dist_convolve arguments must be of the same length")
    else:
        return fftconvolve(a,b)[:n]

def array_sf(array, k = float('inf')):
    if len(array) < k:
        return 1-np.sum(array)
    else:
        return 1-np.sum(array[:k])
    
def single_slot(p,f):
    return pfm_iter_to_array(discrete_pfm_iter(st.geom(p=p,loc=-1)), f)    

def single_slot_probability(p, f):
    """
    Parameters
    ----------
    p : float in [0,1]
        Probability of success. For melds this is [0.17, 0.1, 0.07, 0.05].
    f : number
        Total number of materia lost. (Do not include success).

    Returns
    -------
    number
        Returns the probability that someone gets more failures than you (that
        is, you are more lucky than this proportion of the population)
    """
    return array_sf(single_slot(p, f))

def single_piece(a,b,c,d = -1,s=-1):
    s = max(s, a + b + c + max(0, d))
    ret = dist_convolve(dist_convolve(single_slot(0.17, s), single_slot(0.1, s)), single_slot(0.07, s))
    if d >= 0:
        ret = dist_convolve(ret, single_slot(0.05,s))
    return ret

def single_piece_probability(a,b,c,d = -1):
    """
    Parameters
    ----------
    a : number
        Total number of materia lost for first overmeld slot. (Do not include success).
    b : number
        Total number of materia lost for second overmeld slot. (Do not include success).
    c : number
        Total number of materia lost for third overmeld slot. (Do not include success).
    d : number
        Total number of materia lost for fourth overmeld slot. Leave blank if
        there isn't a fourth overmeld slot. (Do not include success).

    Returns
    -------
    number
        Returns the probability that someone gets more failures than you (that
        is, you are more lucky than this proportion of the population)
    """
    return array_sf(single_piece(a,b,c,d))

def multiple_piece_probability(a,b,c,d):
    """
    Parameters (BROKEN)
    ----------
    a : list of number
        Total number of materia lost for first overmeld slot. (Do not include success).
    b : list of number
        Total number of materia lost for second overmeld slot. (Do not include success).
    c : list of number
        Total number of materia lost for third overmeld slot. (Do not include success).
    d : list of number
        Total number of materia lost for fourth overmeld slot. Can be shorter
        than the other lists to represent pieces without a 4th overmeld.

    Returns
    -------
    number
        Returns the probability that someone gets more failures than you (that
        is, you are more lucky than this proportion of the population)
    """
    l = len(a)
    if len(b) != l or len(c) != l:
        raise Exception("multiple_piece_probability arguments must be of the same length")
    if len(d) < l:
        d += [-1] * (l - len(d))
    for i in range(l):
        if i == 0:
            ret = single_piece(a[i],b[i],c[i],d[i])
        else:
            ret = dist_convolve(single_piece(a[i],b[i],c[i],d[i]), ret)
    return array_sf(ret)
