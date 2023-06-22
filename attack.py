from generator import *
from itertools import compress
import sympy.polys.matrices.domainmatrix as dm

def createSymGenerator(A_transition_array, B_transition_array, C_transition_array, D_transition_array, projection_matrix, output_function):
    key_symbols = sp.symbols("a:16")
    A = LFSR(np.array(key_symbols[:2]), A_transition_array)
    B = LFSR(np.array(key_symbols[2:5]), B_transition_array)
    C = LFSR(np.array(key_symbols[5:9]), C_transition_array)
    D = LFSR(np.array(key_symbols[9:]), D_transition_array)
    return SymLMcombiner([A, B, C, D], projection_matrix, output_function)

def createGenerator(A_transition_array, B_transition_array, C_transition_array, D_transition_array, projection_matrix, output_function, key_array):
    A = LFSR(np.array(key_array[:2]), A_transition_array)
    B = LFSR(np.array(key_array[2:5]), B_transition_array)
    C = LFSR(np.array(key_array[5:9]), C_transition_array)
    D = LFSR(np.array(key_array[9:]), D_transition_array)
    return LMcombiner([A, B, C, D], projection_matrix, output_function)

def generateKeyStream(combiner, n):
    keystream = []
    for i in range(n):
        keystream.append(combiner.calculate_output())
        combiner.next()
    return keystream

def generateEquations(combiner, keystream, n):
    splitted_polys = []
    monomials = set()

    for i in range(n):
        poly = combiner.calculate_output()
        sp.pprint(sp.Eq(poly.expr, keystream[i]), use_unicode=False)
        splitted_poly = [math.prod(compress(poly.gens, monomial)) for monomial in poly.monoms()]
        monomials |= set(splitted_poly)
        splitted_polys.append(splitted_poly)
        combiner.next()
    
    monomials_dict = {v: k for k, v in enumerate(monomials)}
    return (splitted_polys, monomials_dict)

def generateMatrix(polys):
    matrix = np.zeros((len(polys[0]), len(polys[1])), dtype=np.int16)
    for i in range(len(polys[0])):
        for monom in polys[0][i]:
            matrix[i, polys[1][monom]] = 1
    return matrix

def solveSystem(matrix, keystream, monomials):
    F_2 = sp.GF(2)
    a = dm.DomainMatrix.from_Matrix(sp.Matrix(matrix)).convert_to(F_2)
    b = dm.DomainMatrix.from_Matrix(sp.Matrix(keystream)).convert_to(F_2)
    monomials_solution = a.lu_solve(b).to_Matrix()

    monomial_system = []
    for monom, value in zip(monomials.keys(), monomials_solution):
        if (value and not monom.is_symbol):
            monomial_system.extend([sp.Poly(arg + 1) for arg in monom.args])
        else:
            monomial_system.append(sp.Poly(monom + value))
    # monomial_system = [sp.Poly(monom - value) for monom, value in zip(monomials.keys(), monomials_solution)]
    return sp.solve_poly_system(monomial_system, sp.symbols("a:16"), domain=F_2)[0]