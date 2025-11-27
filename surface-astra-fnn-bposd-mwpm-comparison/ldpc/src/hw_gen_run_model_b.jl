using BitSAD
#using .bp_decode_2
include("bp_decode_2.jl")

# pcm = #not a param for dec

#=  1 2 3 4 5 6
1 1 1
2   1 1
3       1 1
4         1 1
=#

H = zeros(Int8, 4, 6)
H[1,1] = 1
H[1,2] = 1
H[2,2] = 1
H[2,3] = 1
H[3,4] = 1
H[3,5] = 1
H[4,5] = 1
H[4,6] = 1

H = sparse([1, 1, 2, 2, 3, 3, 4, 4], [1, 2, 2, 3, 4, 5, 5, 6], [1, 1, 1, 1, 1, 1, 1, 1])


################ above is temp pcm stuff ##################

max_iter = 10
synd = "0010" #inputvector as string
log_prob_ratios = [] #should this be an SBitstream?
bp_decoding = []
bp_decoding_synd = []
M = 4
N = 6
x1 = SBitstream(0.05)
ch = []
converge = 0
zero = SBitstream(0.0)
bits_to_checks = []
checks_to_bits = []
for i in 1:(N*M)
    push!(bits_to_checks, zero)
    push!(checks_to_bits, zero)
    push!(ch, x1)
    push!(bp_decoding, 0)
    push!(bp_decoding_synd, 0)
    push!(log_prob_ratios, zero)
end
sampleDec = decoder(ch, H, max_iter, synd, log_prob_ratios, bp_decoding, bp_decoding_synd, N, M, converge, bits_to_checks, checks_to_bits)

a = bp_decode_prob_ratios(sampleDec)

#f_verilog, f_circuit = generatehw(bp_decode_prob_ratios, sampleDec)
#io = open("hwfile.vl", "w")
#write(io, f_verilog)
#close(io)
