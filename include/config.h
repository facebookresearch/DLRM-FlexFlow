/* Copyright 2018 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _FLEXFLOW_CONFIG_H_
#define _FLEXFLOW_CONFIG_H_
#include <cstring>
#include "legion.h"

// ========================================================
// Define Runtime Constants
// ========================================================
#define MAX_NUM_INPUTS 512
#define MAX_NUM_LOCALS 3
#define MAX_NUM_WORKERS 1024
#define MAX_DIM 10
#define MAX_FILENAME 4096
#define MAX_OPNAME 4096
// DataLoader
#define MAX_SAMPLES_PER_LOAD 64
#define MAX_FILE_LENGTH 128
// Pre-assigned const flags
#define MAP_TO_FB_MEMORY 0xABCD0000
#define MAP_TO_ZC_MEMORY 0xABCE0000

using namespace Legion;

struct ParallelConfig {
  enum DeviceType {
    GPU = 0,
    CPU = 1,
  };
  DeviceType device_type;
  int nDims, dim[MAX_DIM];
  int device_ids[MAX_NUM_WORKERS];
};

bool load_strategies_from_file(const std::string& filename,
                               std::map<MappingTagID, ParallelConfig>& strategies);

bool save_strategies_to_file(const std::string& filename,
                             const std::map<MappingTagID, ParallelConfig>& strategies);

class FFConfig {
public:
  enum PreservedIDs{
    InvalidID = 0,
    DataParallelism_1D = 1,
    DataParallelism_2D = 2,
    DataParallelism_3D = 3,
    DataParallelism_4D = 4,
    DataParallelism_5D = 5,
    DataParallelism_6D = 6,
  };

  FFConfig();
  //bool load_strategy_file(std::string filename);
  //bool save_strategy_file(std::string filename);
  void parse_args(char** argv, int argc);
  static MappingTagID get_hash_id(const std::string& pcname);
  bool find_parallel_config(int ndims,
                            const std::string& pcname,
                            ParallelConfig& config);
public:
  int epochs, batchSize, iterations, printFreq;
  int inputHeight, inputWidth;
  int numNodes, loadersPerNode, workersPerNode;
  float learningRate, weightDecay;
  size_t workSpaceSize;
  Context lg_ctx;
  Runtime* lg_hlr;
  FieldSpace field_space;
  bool syntheticInput, profiling, debug;
  std::string datasetPath, strategyFile;
  // We use MappingTagID has the key since we will pass the tag to the mapper
  std::map<MappingTagID, ParallelConfig> strategies;
};

struct ParaConfigCompare {
  bool operator()(const ParallelConfig& a, const ParallelConfig& b) const {
    if (a.nDims != b.nDims)
      return a.nDims < b.nDims;
    for (int i = 0; i < a.nDims; i++)
      if (a.dim[i] != b.dim[i])
        return a.dim[i] < b.dim[i];
    return false;
  }
};
#endif//_FLEXFLOW_CONFIG_H_
