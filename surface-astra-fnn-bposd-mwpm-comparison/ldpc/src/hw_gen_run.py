from py_jl_conversion_decorators import write_hw_gen_jl
from py_jl_conversion_decorators import L

#The probabilities should be sbitstreams
# ^^ just the channel probabilities? Or the log probs?
# need to L.append import bitsad right?

#remaining questions:
#   what is the input parity_check_matrix?
#   A: it is some_code@error % 2. I do not know where the error decorator is

#TODO:
#   Answer remaining questions and apply answers
#   Debug code
#   Clean code/modify to work on different sets of parameters.
@write_hw_gen_jl
def hw_gen_run():
    print("running hw generation sample decode")
    L.append("using BitSAD")
    L.append("pcm = ") #For now this is used to hardcode stuff. This needs to become a parameter.
    #todo
    L.append("for i in 1:N")
    L.append("ch[i] = 0.05")
    L.append("end")
    #above should be the ch part. May need to initialize ch beforehand.
    L.append("H = ") #??
    L.append("max_iter = 100") #not stoch
    L.append("synd = ") #binary
    L.append("log_prob_ratios = ") #should initialize to stoch
    L.append("bp_decoding = ") #binary, length N. I think I can leave these uninit?
    L.append("bp_decoding_synd = ") #binary, length N. ^^?
    #done
    L.append("M = size(pcm)") #dimensions so not stoch. TODO: make this not hardcoded
    L.append("N = size(pcm, 2)") #dimensions so not stoch. TODO: make this not hardcoded
    L.append("converge = 0") #boolean so not stoch. Just init
    L.append("sampleDec = decoder(ch, H, max_iter, synd, log_prob_ratios, bp_decoding, bp_decoding_synd, N, M, converge)")
    L.append("bp_decode_log_prob_ratios(sampleDec)")
