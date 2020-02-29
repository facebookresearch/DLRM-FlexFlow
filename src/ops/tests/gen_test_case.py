import numpy as np

np.random.seed(0)
m = 4
k = 5
n = 3
d = 2

input1 = np.random.uniform(0, 1, (d,k,m))
input2 = np.random.uniform(0, 1, (d,k,n))
# input1 = np.ones((d,k,m))
# input2 = np.ones((d,k,n))
input1_buffer = []
for batch in input1:
    for row in batch:
        for each in row:
            input1_buffer.append(each)
input2_buffer = []
for batch in input2:
    for row in batch:
        for each in row:
            input2_buffer.append(each)
input1 = np.array(input1)
input2 = np.array(input2)
input1 = input1.transpose((0,2,1))
print()
output_buffer = []
ret = np.matmul(input1, input2)
for batch in ret:
    for row in batch:
        for each in row:
            output_buffer.append(each)

input1_buffer = [str(x) for x in input1_buffer]
input2_buffer = [str(x) for x in input2_buffer]
output_buffer = [str(x) for x in output_buffer]
with open('test_input1.txt', 'w+') as f:
    f.write(' '.join(input1_buffer))


with open('test_input2.txt', 'w+') as f:
    f.write(' '.join(input2_buffer))

with open('test_output.txt', 'w+') as f:
    f.write(' '.join(output_buffer))

with open('test_meta.txt', 'w+') as f:
    f.write(' '.join([str(m), str(k), str(n), str(d)]))