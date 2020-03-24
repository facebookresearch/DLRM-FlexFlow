// #ifndef _FF_UTILS_H_
// #define _FF_UTILS_H_
// #include "model.h"
// #include <sstream>
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #define MAX_DATASET_PATH_LEN 1023
// #define  PRECISION 16

// struct ArgsConfig {
//   char dataset_path[MAX_DATASET_PATH_LEN];
//   char data_type[30];
//   int num_dim;
// };

// void initialize_tensor_from_file(const std::string file_path, 
//   Tensor label, 
//   const FFModel& ff, 
//   std::string data_type="float", 
//   int num_dim=3);

// void initialize_tensor_gradient_from_file(const std::string file_path, Tensor label, const FFModel& ff);

// void initialize_tensor_3d_from_file_task(const Task *task,
//                     const std::vector<PhysicalRegion> &regions,
//                     Context ctx,
//                     Runtime* runtime);

// void dump_region_to_file(FFModel &ff, LogicalRegion &region, std::string file_path);

// void dump_tensor_task(const Task* task,
//                       const std::vector<PhysicalRegion>& regions,
//                       Context ctx, Runtime* runtime);

// #endif