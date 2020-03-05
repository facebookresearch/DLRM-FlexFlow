import subprocess, time, unittest
import numpy as np
global is_file_locked
time
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
    and assumes tensor outter dimension is sample dimension
    '''
    return input.transpose((0,2,1))

def gen_FF_result(test_target, num_gpu):
    command = 'cd ~/DLRM_FlexFlow/src/ops/tests/ && ./test_run_FF_target.sh %s %s' % (test_target, str(num_gpu))
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
    np.testing.assert_allclose(input1_flat, input2_flat, rtol=epsilon, atol=0)

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
    TEST_TARGET = 'batch_matmul_test'
    @staticmethod
    def dump_meta(m,k,n,d):
          with open('test_meta.txt', 'w+') as f:
            f.write(' '.join([str(m), str(k), str(n), str(d)]))

    @staticmethod
    def batch_matmul_test_pipeline(num_gpu, d, m, n, k, epsilon=0.00001):
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
        gen_FF_result(BatchMatmulTest.TEST_TARGET, num_gpu)
        file1 = 'output.txt'
        file2 = 'test_output.txt'
        ret1 = is_equal_tensor_from_file(file1, file2, 'output', epsilon=epsilon)
        file1 = 'test_input1_grad.txt'
        file2 = 'input1_grad.txt'
        ret2 = is_equal_tensor_from_file(file1, file2, 'input1_grad', epsilon=epsilon)
        file1 = 'test_input2_grad.txt'
        file2 = 'input2_grad.txt'
        ret3 = is_equal_tensor_from_file(file1, file2, 'input2_grad', epsilon=epsilon)

    def test_single_gpu_single_batch(self):
        # generate test payload
        d,m,n,k = 1,2,3,4
        num_gpu = 1
        BatchMatmulTest.batch_matmul_test_pipeline(num_gpu, d, m, n, k)
    
    def test_single_gpu_multi_batches(self):
        # generate test payload
        d,m,n,k = 5,2,3,4
        num_gpu = 1
        BatchMatmulTest.batch_matmul_test_pipeline(num_gpu, d, m, n, k)

    def test_multi_gpus_multi_batches(self):
        # generate test payload
        d,m,n,k = 5,2,3,4
        num_gpu = 2
        BatchMatmulTest.batch_matmul_test_pipeline(num_gpu, d, m, n, k)

    def test_8_gpus_small_problem(self):
        # generate test payload
        # need to make sure each gpu have assigned some workload, the assignment uses
        # round robin fashion to assign gpus, if we ask for 14 batches to distribute on 8 gpus
        # each gpu will get 2 batches, so the last gpu will have no data to allocate
        d,m,n,k = 15,2,3,4
        num_gpu = 8
        BatchMatmulTest.batch_matmul_test_pipeline(num_gpu, d, m, n, k)

    # def uneven_distribute_test(self):
    #     # for this configuration we can't distribute payload to each GPU because
    #     # ceil(9 / 8) = 2, for each gpu we assign 2 batches, such we only assign payloads to 5 gpus, 3 gpus won't get 
    #     # any payload, in this scenario FF throws a  `acc.accessor.is_dense_arbitrary(rect)' failed error
    #     # this error is too deep for user to debug, we need to handle this case in FF 
    #     # and throw proper exception - so this test should expect a exception
    #     d,m,n,k = 9,2,3,4
    #     num_gpu = 8
    #     BatchMatmulTest.batch_matmul_test_pipeline(num_gpu, d, m, n, k)

    def test_unit_size_matrix(self):
        # generate test payload
        d,m,n,k = 1,1,1,1
        num_gpu = 1
        BatchMatmulTest.batch_matmul_test_pipeline(num_gpu, d, m, n, k)
    
    def test_unit_size_matrix(self):
        # generate test payload
        d,m,n,k = 2,1,1,1
        num_gpu = 2
        BatchMatmulTest.batch_matmul_test_pipeline(num_gpu, d, m, n, k)

    def test_multi_gpus_ads_team_target_model_shape(self):
        # generate test payload
        d,m,n,k = 145,265,15,64
        num_gpu = 8
        ret = BatchMatmulTest.batch_matmul_test_pipeline(num_gpu, d, m, n, k, epsilon=0.0001)

    def test_single_gpu_ads_team_target_model_shape(self):
        # generate test payload
        d,m,n,k = 145,265,15,64
        num_gpu = 1
        ret = BatchMatmulTest.batch_matmul_test_pipeline(num_gpu, d, m, n, k, epsilon=0.0001)

class TransposeTest(unittest.TestCase):
    '''
    Transpose shape (d,m,k)
    '''
    TEST_TARGET = 'transpose_test'
    @staticmethod
    def dump_meta(m,k,d):
        while is_file_locked:
          time.sleep(0.1)
        is_file_locked = True
        with open('test_meta.txt', 'w+') as f:
          f.write(' '.join([str(m), str(k), str(d)]))
        if_file_locked = False

    def test_single_gpu_single_batch(self):
        # generate test payload
        d,m,k = 1,2,3
        num_gpu = 1
        TransposeTest.transpose_test_pipeline(num_gpu, d, m, k)

    def test_single_gpu_multi_batches(self):
        d,m,k = 9,2,3
        num_gpu = 1
        TransposeTest.transpose_test_pipeline(num_gpu, d, m, k)
    
    def test_unit_batch_matrix(self):
        d,m,k = 1,1,1
        num_gpu = 1
        TransposeTest.transpose_test_pipeline(num_gpu, d, m, k)
      
    def test_multi_gpus_ads_team_target_shape(self):
        d,m,k = 145, 265, 64
        num_gpu = 8
        TransposeTest.transpose_test_pipeline(num_gpu, d, m, k)

    def test_single_gpu_ads_team_target_shape(self):
        d,m,k = 145, 265, 64
        num_gpu = 1
        TransposeTest.transpose_test_pipeline(num_gpu, d, m, k)

    def test_multi_gpus_small_problem(self):
        d,m,k = 2,3,4
        num_gpu = 2
        TransposeTest.transpose_test_pipeline(num_gpu, d, m, k)
    
    def uneven_split_multi_gpus_multi_batch(self):
        d,m,k = 3,4,5
        num_gpu = 2
        TransposeTest.transpose_test_pipeline(num_gpu, d, m, k)

    # # if number_gpu * number_node > batch_size will throw exception
    # # need to handle this exception in FF and add this unit test later on (to expect an exception)
    # def test_multi_gpus_single_batch(self):
    #     d,m,k = 1,2,3
    #     num_gpu = 2
    #     ret = self.transpose_test_pipeline(num_gpu, d, m, k)

    @staticmethod
    def transpose_test_pipeline(num_gpu, d, m, k, epsilon=0.00001):
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
        gen_FF_result(TransposeTest.TEST_TARGET, num_gpu)
        file1 = 'output.txt'
        file2 = 'test_output.txt'
        is_equal_tensor_from_file(file1, file2, 'output', epsilon=epsilon)
        file1 = 'test_input1_grad.txt'
        file2 = 'input1_grad.txt'
        is_equal_tensor_from_file(file1, file2, 'input_grad', epsilon=epsilon)

if __name__ == '__main__':
    unittest.main()