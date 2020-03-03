
with open('output.txt', 'r') as f:
    input1 = f.readline()
with open('test_output.txt', 'r') as f:
    input2 = f.readline()
input1_flat = input1.strip().split(' ')
input1_flat = [float(x) for x in input1_flat]

input2_flat = input2.strip().split(' ')
input2_flat = [float(x) for x in input2_flat]

diff = 0
for i in range(len(input1_flat)):
    diff += abs(input1_flat[i] - input2_flat[i])

print(diff)