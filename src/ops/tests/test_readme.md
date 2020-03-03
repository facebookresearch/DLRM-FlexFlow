
# Batch Matmul Operator test
1. run `python gen_test_case.py 3 4 5 6` function with 4 arguments to set m,k,n,d for random test case, for `example python gen_test_case.py 3 5 7 9` will generate input1 in shape (9,5,3)
2. run `cd ~/DLRM_FlexFlow/src/ops/tests/ && ./run.sh 2 && cd ~/DLRM_FlexFlow` to generate Flexflow operator results, it will dump to a file (assume you have strategy file ready)
3. run `python validate_bmm.py` to validate the results, it will compare the output, input1_grad and input2_grad with python reference and Flexflow operator and output the average difference