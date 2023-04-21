from generator import *

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

fi = open("input.txt", "r")
fo = open("output.txt", "w")
char = fi.read(1)

while char:
    if (int(char) > 1):
         raise Exception
    fo.write(str((int(char) + combiner.calculate_output()) % 2))
    #print(combiner.calculate_output())
    char = fi.read(1)
    combiner.next()
fi.close()
fo.close()
