from attack import *
import os

def throw_exception(text):
    print(text)
    os.system('pause')
    exit()

A_transition_array = np.array([1, 0])
B_transition_array = np.array([1, 0, 1])
C_transition_array = np.array([1, 1, 0, 0])
D_transition_array = np.array([1, 1, 1, 0, 1, 0, 0])

projection_matrix = np.array([[1, 0, 0, 0], 
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])

output_function = sp.lambdify(sp.symbols('at bt ct dt'), sp.parse_expr(input('Enter the output function (use only symbols "at", "bt", "ct", "dt" for corresponding LFSR elements): ')))
n = int(input('Enter the keystream length: '))
key = []
for bit in input('Enter the key (16 bits): '):
    if (int(bit) == 0 or int(bit) == 1):
        key.append(int(bit))
    else:
        throw_exception('Use only "0" and "1"!')
if (len(key) != 16):
    throw_exception('The key must be 16 bits!')

combiner = createGenerator(A_transition_array,
                           B_transition_array,
                           C_transition_array,
                           D_transition_array,
                           projection_matrix,
                           output_function, 
                           key)

keystream = generateKeyStream(combiner, n)

sym_combiner = createSymGenerator(A_transition_array,
                                  B_transition_array,
                                  C_transition_array,
                                  D_transition_array,
                                  projection_matrix,
                                  output_function)
print('Generated sytem of equations:')
system = generateEquations(sym_combiner, keystream, n)
matrix = generateMatrix(system)

try:
    solution = solveSystem(matrix, keystream, system[1])
    print('Calculated key: ' + ''.join([str(bit % 2) for bit in solution]))
    os.system('pause')
except:
    throw_exception('System of equations is unsolvable')