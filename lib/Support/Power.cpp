
#include <iostream>
#include <fstream>

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/FIRRTL/Power.h"
#include "mlir/IR/Threading.h"

#include "circt/Dialect/Perf/PerfDialect.h"
#include "circt/Dialect/Perf/PerfHelpers.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "circt/Dialect/Perf/PerfPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;

// =================================
// ===== Default virtual class =====
// =================================

// constructor
PowerClusterer::PowerClusterer(std::string args, int max_num_clusters)
  : max_num_clusters(max_num_clusters)
{
  parseArgs(args);
}

PowerClusterer::~PowerClusterer() {}

// parse clustering arguments
void PowerClusterer::parseArgs(std::string args) {

}

void PowerClusterer::runOnCircuit(firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  std::vector<power_cluster_t> &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {

}

// ===============================
// ===== Max power clusterer =====
// ===============================

std::string MaxPowerClusterer::ID = "MaxPowerClusterer";

MaxPowerClusterer::MaxPowerClusterer(std::string args, int max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

MaxPowerClusterer::~MaxPowerClusterer() {}

void MaxPowerClusterer::runOnCircuit(firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  std::vector<power_cluster_t> &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {
  double idle_power = *idle_power_ptr;

  int num_clusters = (int)clusters.size();
  log_stream << "Number of clusters to start: " << num_clusters << ", must minimize to " << max_num_clusters << "\n";
  int i = 0;
  for (; i < max_num_clusters && i < num_clusters; ++i) {
      // find maximum power cluster
      int max_power_i = i;
      double max_power = clusters[i].power;
      for (int j = i + 1; j < num_clusters; ++j) {
          if (clusters[j].power > max_power) {
              max_power_i = j;
              max_power = clusters[j].power;
          }
      }

      // swap clusters
      log_stream << "Entry " << i << " found at " << max_power_i << " with power " << clusters[max_power_i].power << " (" << clusters[max_power_i].name << ")\n";
      if (max_power_i != i) {
          power_cluster_t tmp = clusters[i];
          clusters[i] = clusters[max_power_i];
          clusters[max_power_i] = tmp;
      }
  }

  // trim off less significant events as idle
  for (; i < num_clusters; ++i) {
      idle_power += clusters[i].power;
  }
  clusters.resize(max_num_clusters);

  *idle_power_ptr = idle_power;
}


