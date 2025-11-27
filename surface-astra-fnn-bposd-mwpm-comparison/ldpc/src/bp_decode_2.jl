#module bp_decode_2
#This is a new version of the necessary mod2sparse functions for decoding which makes use
# of the sparsearrays library in julia.

#Need to get first and last non-zero element in each row/col of a sparse matrix

#SBitStream values: channel probabilities, check_to_bit values, bit_to_check values,

#Need to store bit_to_check and check_to_bit for each entry in the sparse matrix
# Possible options:
#   1. Two parallel sparse matrices, one for bit_to_check and one for check_to_bit.
#   2. Two arrays parallel with m.nzval of bit_to_check messages and check_to_bit messages.
# A: Currently going with option 2, called bits_to_checks and checks_to_bits respectively.


using BitSAD
using SparseArrays

export decoder, bp_decode_prob_ratios

#DONE:

#these are redundant. Access bit_to_check for an entry at the index of that entry in nzval.
#and same for check_to_bit, as these are 3 parallel arrays.
function mod2sparse_bit_to_check(bits_to_checks, i)
    return bits_to_checks[i]
end

function mod2sparse_check_to_bit(checks_to_bits, i)
    return checks_to_bits[i]
end

function mod2sparse_at_start_row(m, i, e)
    return (mod2sparse_first_in_row(m, i) == e)
end

#unfortunately requires m as well as e, not as elegant as next_in_col
function mod2sparse_next_in_row(m, e)
    for i in (e+1):length(m.rowval)
        if (m.rowval[i] == e)
            return i
        end
    end
end

#can only guarantee a prev exists with a unique mod2sparse_at_start_row function
function mod2sparse_prev_in_row(m, e)
    for i in e:-1:1
        if (m.rowval[i] == e)
            return i
        end
    end
end

#TODO:


#structs, SBitStreams


#MOSTLY DONE:

#Note: m.colptr is always of size n+1 for a matrix of n non-zero elements.
#Q: How should this function behave in the absence of a non-zero element in the column?
#A: For now, assume this is never the case. TODO: ACTUALLY ANSWER THIS LATER
#returns the index of the value in nzval associated with the first non-zero element of column i in sparse matrix m.
function mod2sparse_first_in_col(m, i)
    return m.colptr[i]
end

function mod2sparse_last_in_col(m, i) #No index out of bounds error because colptr is 1 too large.
    return (m.colptr[i+1] - 1)
end

#returns true if e is the last nz element in the column of m, false otherwise.
#TODO: make this such that m does not need to be passed as a parameter.
#I am unsure if I need separate at_end row and col functions.
function mod2sparse_at_end_col(m, j, e)
    return (m.colptr[j+1] == e)
end

#this sucks. TODO: is there a way to not use m?
function mod2sparse_at_end_row(m, i, e)
    return (mod2sparse_last_in_row(m, i) == e)
end

#returns the nzval index associated with the next nz value in a matrix.
#TODO: get rid of this function call, only kept for now to mirror original structure.
function mod2sparse_next_in_col(e)
    return e+1
end

#SHOULD TRY TO ACCESS BY COLUMN, NOT ROW.
#Q: How should this function behave in the absence of a non-zero element in the column?
#A: For now, assume this is never the case. TODO: ACTUALLY ANSWER THIS LATER
#returns the index of the value in nzval associated with the first non-zero element of row i in sparse matrix m.
function mod2sparse_first_in_row(m, i) #m is assumed to be a SparseMatrixCSC struct
    print("row is ")
    print(i)
    for e in 1:length(m.rowval) #(front to back)
        if (m.rowval[e] == i)
            print("first is ")
            print(e)
            return e
        end
    end
end

function mod2sparse_last_in_row(m, i)
    for e in length(m.rowval):-1:1 #(back to front)
        if (m.rowval[e] == i)
            return e
        end
    end
end

function mod2sparse_at_start_col(m, j, e)
    return (m.colptr[j] <= e)
end

function mod2sparse_prev_in_col(e)
    return e-1
end

function mod2sparse_mulvec(H, received_codeword, synd)
    #rows is [1], cols is [2]
    M = size(H)[1]
    N = size(H)[2]

    for i in 1:M
        synd[i] = 0
    end

    for j in 1:N
        if received_codeword[j] == 1
            e = mod2sparse_first_in_col(H, j)
            while !(mod2sparse_at_end_col(H, j, e))
                synd[H.rowval[e]] ^= 1
                e = mod2sparse_next_in_col(e)
            end
        end
    end
end

function bp_decode_prob_ratios(dec) #product-sum
    for j in 1:dec.N
        # e is first non-zero entry in column j of H.
        e = mod2sparse_first_in_col(dec.H, j)
        while !(mod2sparse_at_end_col(dec.H, j, e))
            #print(e)
            dec.bits_to_checks[e] = SBitstream(float(dec.channel_probs[j]) / (1.0 - float(dec.channel_probs[j]))) #first
            e = mod2sparse_next_in_col(e)
        end
    end

    dec.converge = 0

    #TODO: change this to go by columns rather than by rows, Julia is column-major.
    for iteration in 1:(dec.max_iter+1)
        for i in 1:dec.M
            e = mod2sparse_first_in_row(dec.H, i)
            temp = SBitstream((-1.0) ^(float(dec.synd[i])))
            while !(mod2sparse_at_end_row(dec.H, i, e))
                print(" first loop ")
                checks_to_bits[e] = temp #first
                temp = temp * SBitstream(2.0/(1.0 + float(bits_to_checks[e])) - 1)
                e = mod2sparse_next_in_row(dec.H, e)
            end
            e = mod2sparse_last_in_row(dec.H, i)
            temp = SBitstream(1.0)
            while !(mod2sparse_at_start_row(dec.H, i, e))
                print("second loop, e is ")
                print(e)
                checks_to_bits[e] = checks_to_bits[e] * temp
                checks_to_bits[e] = SBitstream((1.0 - float(checks_to_bits[e])) / (1.0 + float(checks_to_bits[e])))
                temp = temp * SBitstream(2.0 / (1.0 + float(bits_to_checks[e])) - 1.0)
                e = mod2sparse_prev_in_row(dec.H, e)
            end
        end
        #bit to check messages
        for j in 1:dec.N
            e = mod2sparse_first_in_col(dec.H, j)
            temp = SBitstream(float(dec.channel_probs[j]) / (1.0 - float(dec.channel_probs[j])))
            while !(mod2sparse_at_end_col(dec.H, j, e))
                print(2)
                bits_to_checks[e] = temp
                temp = temp * checks_to_bits[e]
                #Maybe an isnan(temp) check here? idk when that would be true though. bp_decoder.pyx line 287
                e = mod2sparse_next_in_col(e)
            end
            #dec.log_prob_ratios[j] = temp
            if float(temp) >= 1 #????? How? Should this not be <= 1?
                dec.bp_decoding[j] = 1
            else
                dec.bp_decoding[j] = 0
            end

            e = mod2sparse_last_in_col(dec.H, j)
            temp = SBitstream(1.0)
            while !(mod2sparse_at_start_col(dec.H, j, e))
                print(1)
                bits_to_checks[e] = bits_to_checks[e] * temp
                temp = temp * checks_to_bits[e]
                #maybe nan check on temp again?
                e = mod2sparse_prev_in_col(e)
            end

            mod2sparse_mulvec(dec.H, dec.bp_decoding, dec.bp_decoding_synd) #no SBitstreams!!

            equal = 1
            for check in 1:dec.M
                if dec.synd[check] != dec.bp_decoding_synd[check]
                    equal = 0
                    break
                end
            end
            if equal == 1
                dec.converge = 1
                return 1
            end
        end
    end
    return 0
end

#IN PROGRESS:
#=
function bp_decode_log_prob_ratios(dec) #product-sum
    for j in 1:dec.N
        # e is first non-zero entry in column j of H.
        e = mod2sparse_first_in_col(dec.H, j)
        while !(mod2sparse_at_end_col(m, j, e))
            dec.bits_to_checks[e] = log((1 - dec.channel_probs[j]) / (dec.channel_probs[j]))
            e = mod2sparse_next_in_col(e)
        end

    dec.converge = 0
    #TODO: change this to go by columns rather than by rows, Julia is column-major.
    for iteration in 1:(dec.max_iter+1)
        for i in 1:dec.M
            e = mod2sparse_first_in_row(dec.H, i)
            temp = 1.0
            while !(mod2sparse_at_end_row(dec.H, i, e)
                checks_to_bits[e] = temp #first
                temp *= tanh(bits_to_checks[e] / 2)
                e = mod2sparse_next_in_row(dec.H, e)
            end
            e = mod2sparse_last_in_row(m, i)
            temp = 1.0
            while !(mod2sparse_at_start_row(dec.H, i, e)
                checks_to_bits[e] *= temp
                checks_to_bits[e] = (((-1)^dec.synd[i]) * log((1 + checks_to_bits[e]) / (1 - checks_to_bits[e])))
                temp *= tanh(bits_to_checks[e] / 2)
                e = mod2sparse_prev_in_row(dec.H, e)
            end

        #bit to check messages
        for j in 1:dec.N
            e = mod2sparse_first_in_col(dec.H, j)
            temp = log((1-dec.channel_probs[j]) / (dec.channel_probs[j]))
            while !(mod2sparse_at_end_col(dec.H, j, e)
                bits_to_checks[e] = temp
                temp += checks_to_bits[e]
                e = mod2sparse_next_in_col(e)
            end
            dec.log_prob_ratios[j] = temp
            if temp <= 0
                dec.bp_decoding[j] = 1
            else
                dec.bp_decoding[j] = 0

            e = mod2sparse_last_in_col(dec.H, j)
            temp = 0.0
            while !(mod2sparse_at_start_col)
                bits_to_checks[e] += temp
                temp += checks_to_bits[e]
                e = mod2sparse_prev_in_col(e)
            end

            mod2sparse_mulvec(dec.H, dec.bp_decoding, dec.bp_decoding_synd)

            equal = 1
            for check in 1:dec.M
                if dec.synd[check] != dec.bp_decoding_synd[check]
                    equal = 0
                    break
                end
            end
            if equal == 1
                dec.converge = 1
                return 1
            end

            return 0
end
=#
mutable struct decoder
    channel_probs #SBitstream[]
    H
    max_iter
    synd
    log_prob_ratios #SBitstream[]
    bp_decoding
    bp_decoding_synd
    N
    M
    converge
    bits_to_checks #SBitstream[]
    checks_to_bits #SBitstream[]
end

#end