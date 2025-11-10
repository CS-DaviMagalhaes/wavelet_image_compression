import math
import numpy as np


# Haar wavelet coefficients
sq = 1 / math.sqrt(2)
a = [sq, sq]       # low-pass coefficients (a_k)
b = [sq, -sq]      # high-pass coefficients (Beta_k)

# equation 1
# Gives us the approximation (smoothed version of the signal)
def low_pass(signal, a):
    s1 = []
    for i in range(0, len(signal)-1, 2):
        val = a[0]*signal[i] + a[1]*signal[i+1]
        s1.append(val)
    
    return s1

# equation 2
# Gives us the details (difference of corners in the signal)
def high_pass(signal, b):
    d1 = []
    for i in range(0, len(signal)-1, 2):
        val = b[0]*signal[i] + b[1]*signal[i+1]
        d1.append(val)
    
    return d1

def dwt_2d(img):
    # apply 1D to each row
    rows_low, rows_high = [], []
    for row in img:
        rows_low.append(low_pass(row, a))
        rows_high.append(high_pass(row, b))
    rows_low, rows_high = np.array(rows_low), np.array(rows_high)

    # apply 1D to each column of the matrix we obtained above
    A, H, V, D = [], [], [], []
    for col_low, col_high in zip(rows_low.T, rows_high.T):
        A.append(low_pass(col_low, a))
        H.append(high_pass(col_low, b))
        V.append(low_pass(col_high, a))
        D.append(high_pass(col_high, b))

    A = np.array(A).T # Approximation (LL)
    H = np.array(H).T # Horizontal detail (HL)
    V = np.array(V).T # Vertical detail (LH)
    D = np.array(D).T # Diagonal detail (HH)
    return A, H, V, D # Coefficients returned in the order of scanning (LL, HL, LH, HH)

# WDR Method (first sorting pass only)
def wdr_method(dwt_coeffs): 
    ics = set() # Insignificant Coefficients Set
    scs = set() # Significant Coefficients Set 
    tps = set() # Temporary Set of new significant coefficients 

    max_coef = 0.0
    coeff_list = []  # flattened list to keep coefficients with order

    # Initially all coefficients are insignificant
    for subband in dwt_coeffs: # Order LL, HL, LH, HH
        for row in subband: 
            for coef in row:
                coeff_list.append(coef)
                ics.add(len(coeff_list) - 1)  # store index instead of value
                max_coef = max(max_coef, abs(coef)) # Calculate the maximum coefficient
    
    t0 = 2**math.floor(math.log2(max_coef)) # Initial threshold

    # Sorting pass: Put significant coefficients of ics in tps
    # (A coefficient x is significant if abs(x) >= Threshold)
    signs = [] # 1 for positive and 0 for negative
    tps_indices = [] # store indices of new significant coefficients
    for i, coef in enumerate(coeff_list):
        if abs(coef) >= t0: 
            tps.add(i)
            tps_indices.append(i)
            signs.append(1 if coef >= 0 else 0)

    # encode (difference reduction)
    # For each significant coefficient, store the difference between
    # its index and the previous significant one.
    S_reduced = []
    prev = 0
    for idx in tps_indices:
        diff = idx - prev
        S_reduced.append(diff)
        prev = idx
    
    # Binary encoding (Concatenate binary and the sign)
    # For example 2 -> 10, since its positive we add the sign at the end -> 100
    # Then we apply binary reduction (remove most significant 1 bit)
    S_encoded = []
    for diff, sign in zip(S_reduced, signs): 
        b = bin(diff)[2:]  # binary string
        b_reduced = b[1:] if len(b) > 1 else ''  # remove MSB, if only 1 bit then blank
        b_final = b_reduced + str(sign)  # attach sign bit
        S_encoded.append(b_final)

    # We can join the encodings as a bitstream
    bitstream = ''.join(S_encoded)
    return bitstream






    



