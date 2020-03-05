import subprocess
import numpy as np
import unittest

def dump_tensor_3d_to_file(tensor, file_name):
    buffer = []
    for entry in tensor.flatten():
      buffer.append(entry)
    buffer = ["%.16f"%x for x in buffer]
    with open(file_name, 'w+') as f:
      f.write(' '.join(buffer))

def batch_matmul_3d_reference(input1, input2, trans1, trans2):
    '''
    Input layout:
    input1 (d,k,m)
    input2 (d,k,n)
    output (d,m,n)
    '''
    input1 = input1.transpose((0,2,1)) if trans1 else input1
    input2 = input2.transpose((0,2,1)) if trans2 else input2
    return np.matmul(input1, input2)

def batch_transpose_3d_reference(input):
    '''
    This operation transposes the inner 2 dimensions (flip inner two)
    and assume all outter dimensions are sample dimension
    '''
    return input.transpose((0,2,1))

def gen_FF_result(test_target, num_gpu):
    command = 'cd ~/DLRM_FlexFlow/src/ops/tests/ && ./run_FF_test_target.sh %s %s' % (test_target, str(num_gpu))
    test_process = subprocess.Popen([command], shell=True)
    test_process.wait()

def is_equal_tensor_from_file(file_1, file_2, label='', epsilon=0.00001):
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
      return True
    else:
      print('diff too significant: %.6f' % avg_diff)
      return False

class BatchMatmulTest(unittest.TestCase):
    '''
    BMM default layout
    input1 (d,m,k)
    input2 (d,k,n)
    output (d,m,n)

    Target shape in caffe2
    input1 (d,k,m)
    input2 (d,k,n)
    output (d,m,n)
    so we set trans1=True and trans2=False
    '''
    test_target = 'batch_matmul_test'
    @staticmethod
    def dump_meta(m,k,n,d):
      with open('test_meta.txt', 'w+') as f:
        f.write(' '.join([str(m), str(k), str(n), str(d)]))

    def batch_matmul_test_pipeline(self, num_gpu, d, m, n, k, epsilon=0.00001):
        # generate python reference and input payload
        input1_tensor = np.random.uniform(0, 1, (d,k,m))
        dump_tensor_3d_to_file(input1_tensor, "test_input1.txt")
        input2_tensor = np.random.uniform(0, 1, (d,k,n))
        dump_tensor_3d_to_file(input2_tensor, "test_input2.txt")
        output_gradient_tensor = np.random.uniform(0, 1, (d,m,n))
        dump_tensor_3d_to_file(output_gradient_tensor, "test_output_grad.txt")

        output_tensor = batch_matmul_3d_reference(input1_tensor, input2_tensor, trans1=True, trans2=False)
        input1_grad_tensor = batch_matmul_3d_reference(input2_tensor, output_gradient_tensor, trans1=False, trans2=True)
        input2_grad_tensor = batch_matmul_3d_reference(input1_tensor, output_gradient_tensor, trans1=False, trans2=False)
        dump_tensor_3d_to_file(output_tensor, "test_output.txt")
        dump_tensor_3d_to_file(input1_grad_tensor, "test_input1_grad.txt")
        dump_tensor_3d_to_file(input2_grad_tensor, "test_input2_grad.txt")
        BatchMatmulTest.dump_meta(m,k,n,d)

        # generate FF results
        gen_FF_result(BatchMatmulTest.test_target, num_gpu)
        file1 = 'output.txt'
        file2 = 'test_output.txt'
        ret1 = is_equal_tensor_from_file(file1, file2, 'output', epsilon=epsilon)
        file1 = 'test_input1_grad.txt'
        file2 = 'input1_grad.txt'
        ret2 = is_equal_tensor_from_file(file1, file2, 'input1_grad', epsilon=epsilon)
        file1 = 'test_input2_grad.txt'
        file2 = 'input2_grad.txt'
        ret3 = is_equal_tensor_from_file(file1, file2, 'input2_grad', epsilon=epsilon)
        return (ret1 and ret2 and ret3)

    def test_single_gpu_single_batch(self):
        # generate test payload
        d,m,n,k = 1,2,3,4
        num_gpu = 1
        ret = self.batch_matmul_test_pipeline(num_gpu, d, m, n, k)
        assert ret == True
    
    def test_single_gpu_multi_batches(self):
        # generate test payload
        d,m,n,k = 5,2,3,4
        num_gpu = 1
        ret = self.batch_matmul_test_pipeline(num_gpu, d, m, n, k)
        assert ret == True

    def test_multi_gpus_multi_batches(self):
        # generate test payload
        d,m,n,k = 5,2,3,4
        num_gpu = 2
        ret = self.batch_matmul_test_pipeline(num_gpu, d, m, n, k)
        assert ret == True

    def test_unit_size_matrix(self):
        # generate test payload
        d,m,n,k = 1,1,1,1
        num_gpu = 1
        ret = self.batch_matmul_test_pipeline(num_gpu, d, m, n, k)
        assert ret == True

    def test_multi_gpus_ads_team_target_model_shape(self):
        # generate test payload
        d,m,n,k = 145,265,15,64
        num_gpu = 2
        ret = self.batch_matmul_test_pipeline(num_gpu, d, m, n, k, epsilon=0.0001)
        assert ret == True

    def test_single_gpu_ads_team_target_model_shape(self):
        # generate test payload
        d,m,n,k = 145,265,15,64
        num_gpu = 1
        ret = self.batch_matmul_test_pipeline(num_gpu, d, m, n, k, epsilon=0.0001)
        assert ret == True

class TransposeTest(unittest.TestCase):
    '''
    Transpose shape (d,m,k)
    '''
    test_target = 'transpose_test'
    @staticmethod
    def dump_meta(m,k,d):
      with open('test_meta.txt', 'w+') as f:
        f.write(' '.join([str(m), str(k), str(d)]))

    def test_single_gpu_single_batch(self):
        # generate test payload
        d,m,k = 1,2,3
        num_gpu = 1
        ret = self.transpose_test_pipeline(num_gpu, d, m, k)
        assert ret == True

    def test_single_gpu_multi_batches(self):
        d,m,k = 9,2,3
        num_gpu = 1
        ret = self.transpose_test_pipeline(num_gpu, d, m, k)
        assert ret == True
    
    def test_unit_batch_matrix(self):
        d,m,k = 1,1,1
        num_gpu = 1
        ret = self.transpose_test_pipeline(num_gpu, d, m, k)
        assert ret == True
      
    def test_multi_gpus_ads_team_target_shape(self):
        d,m,k = 145, 265, 64
        num_gpu = 2
        ret = self.transpose_test_pipeline(num_gpu, d, m, k)
        assert ret == True

    def test_single_gpu_ads_team_target_shape(self):
        d,m,k = 145, 265, 64
        num_gpu = 1
        ret = self.transpose_test_pipeline(num_gpu, d, m, k)
        assert ret == True

    def test_multi_gpus_small_problem(self):
        d,m,k = 2,3,4
        num_gpu = 2
        ret = self.transpose_test_pipeline(num_gpu, d, m, k)
        assert ret == True
    
    def uneven_split_multi_gpus_multi_batch(self):
        d,m,k = 3,4,5
        num_gpu = 2
        ret = self.transpose_test_pipeline(num_gpu, d, m, k)
        assert ret == True        

    # if number_gpu * number_node > batch_size will throw exception
    # def test_multi_gpus_single_batch(self):
    #     d,m,k = 1,2,3
    #     num_gpu = 2
    #     ret = self.transpose_test_pipeline(num_gpu, d, m, k)
    #     assert ret == False

    def transpose_test_pipeline(self, num_gpu, d, m, k, epsilon=0.00001):
        # generate python reference and input payload
        input_tensor = np.random.uniform(0, 1, (d,m,k))
        dump_tensor_3d_to_file(input_tensor, "test_input1.txt")
        output_gradient_tensor = np.random.uniform(0, 1, (d,k,m))
        dump_tensor_3d_to_file(output_gradient_tensor, "test_output_grad.txt")
        output_tensor = batch_transpose_3d_reference(input_tensor)
        input_grad_tensor = batch_transpose_3d_reference(output_gradient_tensor)
        dump_tensor_3d_to_file(output_tensor, "test_output.txt")
        dump_tensor_3d_to_file(input_grad_tensor, "test_input1_grad.txt")
        TransposeTest.dump_meta(m,k,d)

        # generate FF results
        gen_FF_result(TransposeTest.test_target, num_gpu)
        file1 = 'output.txt'
        file2 = 'test_output.txt'
        ret1 = is_equal_tensor_from_file(file1, file2, 'output', epsilon=epsilon)
        file1 = 'test_input1_grad.txt'
        file2 = 'input1_grad.txt'
        ret2 = is_equal_tensor_from_file(file1, file2, 'input_grad', epsilon=epsilon)
        return (ret1 and ret2)

if __name__ == '__main__':
    unittest.main()



    