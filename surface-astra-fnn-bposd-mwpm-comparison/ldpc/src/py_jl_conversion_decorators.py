import functools

L = []

def write_hw_gen_jl(func):
    @functools.wraps(func)
    def hw_gen_jl_wrapper(*args, **kwargs):
        a = func(*args, **kwargs)
        L.append("f_verilog, f_circuit = generatehw(NAMEOFFUNC)")
        L.append("io = open(\"hwfile.vl\", \"w\")")
        L.append("write(io, f_verilog)")
        L.append("close(io)")

        jl_file = open("jl_code.jl", 'a')
        jl_file.writelines(L)  # where L is an array of all the lines of julia to be written
        jl_file.close()

        return a
    return hw_gen_jl_wrapper()


# I need a globally accessible list L which holds lines of code. It must be modified
#   by every function which is called at any point. So anytime a function call is
#   made within an existing function call, the inner function must write to the
#   same L which the outer function writes to. One approach is to pass L as an
#   argument every time a function is called, and return it from every inner function
#   (overwriting the previously existing L each time this is done). This requires
#   modifications to every function definition, call, and return. Another solution
#   is just a global variable L. This requires no modifications beyond appending
#   lines to L. Ideally, I can make a variable limited to the scope of the decorator
#   which is accessible by any function called from within the decorator.