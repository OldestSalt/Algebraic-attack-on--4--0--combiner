from generator import *
import os, psutil

A_transition_array = np.array([1, 0])
B_transition_array = np.array([1, 0, 1])
C_transition_array = np.array([1, 1, 0, 0])
D_transition_array = np.array([1, 1, 1, 0, 1, 0, 0])

A = LFSR(np.array([1, 1]), A_transition_array)
B = LFSR(np.array([0, 0, 0]), B_transition_array)
C = LFSR(np.array([0, 1, 1, 0]), C_transition_array)
D = LFSR(np.array([0, 1, 1, 0, 1, 0, 1]), D_transition_array)
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
combiner = LMcombiner([A, B, C, D], projection_matrix, lambda at, bt, ct, dt: bt * ct + at + ct + dt)

# A_transition_array = np.array([1, 1])
# B_transition_array = np.array([1, 1, 0])

# A = LFSR(np.array([0, 1]), A_transition_array)
# B = LFSR(np.array([1, 1, 0]), B_transition_array)
# projection_matrix = np.array([[0, 0], 
#                               [1, 0],
#                               [0, 1],
#                               [0, 0],
#                               [0, 0]])
# combiner = LMcombiner([A, B], projection_matrix, lambda at, bt: bt * at + bt + at)

keystream = [combiner.calculate_output()]
for i in range(50):
    keystream.append(combiner.next())

key_symbols = sp.symbols("a:16")
A = LFSR(np.array(key_symbols[:2]), A_transition_array)
B = LFSR(np.array(key_symbols[2:5]), B_transition_array)
C = LFSR(np.array(key_symbols[5:9]), C_transition_array)
D = LFSR(np.array(key_symbols[9:]), D_transition_array)
combiner = SymLMcombiner([A, B, C, D], projection_matrix, lambda at, bt, ct, dt: bt * ct + at + ct + dt)

# key_symbols = sp.symbols("a:5")
# A = LFSR(np.array(key_symbols[:2]), A_transition_array)
# B = LFSR(np.array(key_symbols[2:]), B_transition_array)
# combiner = SymLMcombiner([A, B], projection_matrix, lambda at, bt: bt * at + bt + at)

sys = [sp.Eq(combiner.calculate_output(), keystream[0])]

symbols = combiner.calculate_output().free_symbols
for i in range(1, 50):
    poly = combiner.next()
    symbols |= poly.free_symbols
    sys.append(sp.Eq(poly, keystream[i]))
symbols = list(symbols)

sys_matrices = sp.linear_eq_to_matrix(sys, symbols)

F_2 = sp.GF(2)
a = sp.polys.matrices.domainmatrix.DomainMatrix.from_Matrix(sys_matrices[0]).convert_to(F_2)
b = sp.polys.matrices.domainmatrix.DomainMatrix.from_Matrix(sys_matrices[1]).convert_to(F_2)
solution = a.lu_solve(b).to_Matrix()

sym_values = {symbols[i]: solution[i] for i in range(len(symbols))}
sym_dict = {v: k for k, v in combiner.monomials.items()} | {sym.name: sym for sym in key_symbols}
new_sys = [sp.Eq(sym_dict[sym.name], value) for sym, value in sym_values.items()]
new_sys2 = []

for equation in new_sys:
    if (equation.rhs):
        for sym in equation.lhs.free_symbols:
            if (sp.Eq(sym, 1) not in new_sys2):
                new_sys2.append(sp.Eq(sym, 1))
    else:
        new_sys2.append(equation)
sol = sp.nonlinsolve(new_sys2, key_symbols)

process = psutil.Process()