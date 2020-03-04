epsilon = 0.00001

def calculate_difference(file_1, file_2, label=''):
    with open(file_1, 'r') as f:
        input1 = f.readline()
    with open(file_2, 'r') as f:
        input2 = f.readline()
    input1_flat = input1.strip().split(' ')
    input1_flat = [float(x) for x in input1_flat]

    input2_flat = input2.strip().split(' ')
    input2_flat = [float(x) for x in input2_flat]

    diff = 0
    for i in range(len(input1_flat)):
        diff += abs(input1_flat[i] - input2_flat[i])
    avg_diff = diff/float(len(input1_flat))
    if avg_diff < epsilon:
      print('%s: OK' % label)
    else:
      print('%s diff too significant: %.6f' % (label, avg_diff))

file1 = 'output.txt'
file2 = 'test_output.txt'
calculate_difference(file1, file2, 'output')


file1 = 'test_input1_grad.txt'
file2 = 'input1_grad.txt'
calculate_difference(file1, file2, 'input 1 grad')

file1 = 'test_input2_grad.txt'
file2 = 'input2_grad.txt'
calculate_difference(file1, file2, 'input 2 grad')