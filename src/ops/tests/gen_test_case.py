import numpy as np

np.random.seed(0)
m = 3
k = 4
n = 5
d = 5

input1 = np.random.uniform(0, 1, (d,k,m))
input2 = np.random.uniform(0, 1, (d,k,n))
output_grad = np.random.uniform(0, 1, (d,m,n))

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
output_grad_buffer = []
for batch in output_grad:
    for row in batch:
        for each in row:
            output_grad_buffer.append(each)


ret = np.matmul(input1.transpose((0,2,1)), input2)
output_buffer = []
for batch in ret:
    for row in batch:
        for each in row:
            output_buffer.append(each)

input1_grad = np.matmul(input2, output_grad.transpose((0, 2, 1)))
input2_grad = np.matmul(input1, output_grad)
input1_grad_buffer = []
input2_grad_buffer = []
for batch in input1_grad:
    for row in batch:
        for each in row:
            input1_grad_buffer.append(each)
for batch in input2_grad:
    for row in batch:
        for each in row:
            input2_grad_buffer.append(each)



input1_buffer = [str(x) for x in input1_buffer]
input2_buffer = [str(x) for x in input2_buffer]
output_buffer = ["%.5f"%x for x in output_buffer]
output_grad_buffer = [str(x) for x in output_grad_buffer]
input1_grad_buffer = ["%.6f"%x for x in input1_grad_buffer]
input2_grad_buffer = ["%.6f"%x for x in input2_grad_buffer]
with open('test_input1.txt', 'w+') as f:
    f.write(' '.join(input1_buffer))
with open('test_input2.txt', 'w+') as f:
    f.write(' '.join(input2_buffer))
with open('test_output.txt', 'w+') as f:
    f.write(' '.join(output_buffer))
with open('test_output_grad.txt', 'w+') as f:
    f.write(' '.join(output_grad_buffer))
with open('test_input1_grad.txt', 'w+') as f:
    f.write(' '.join(input1_grad_buffer))
with open('test_input2_grad.txt', 'w+') as f:
    f.write(' '.join(input2_grad_buffer))

with open('test_meta.txt', 'w+') as f:
    f.write(' '.join([str(m), str(k), str(n), str(d)]))