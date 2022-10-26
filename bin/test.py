from Compiler.library import print_ln, sint
from Compiler.compilerLib import Compiler

compiler = Compiler()

@compiler.register_function('helloworld')
def hello_world():
    i_1 = sint.get_input_from(0)
    i_2 = sint.get_input_from(1)
    r = i_1 + i_2
    r.reveal_to(0)
    r.reveal_to(1)

if __name__ == "__main__":
    compiler.compile_func()

