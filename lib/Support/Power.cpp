
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

PowerClusterer *PowerClusterer::create(std::string clustering_alg, std::string args, int max_num_clusters) {
  if (clustering_alg == MaxPowerClusterer::ID) {
    return new MaxPowerClusterer(args, max_num_clusters);
  }
  else if (clustering_alg == GlueLogicClusterer::ID) {
    return new GlueLogicClusterer(args, max_num_clusters);
  }
  else if (clustering_alg == SimmaniClusterer::ID) {
    return new SimmaniClusterer(args, max_num_clusters);
  }
  else if (clustering_alg == MinimizeConnectionsClusterer::ID) {
    return new MinimizeConnectionsClusterer(args, max_num_clusters);
  }
  else if (clustering_alg == SwitchingProbClusterer::ID) {
    return new SwitchingProbClusterer(args, max_num_clusters);
  }
  else {
    return nullptr;
  }
}

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

          // todo ignore cluster when a vector
          // reg_target->setAttr(getBusWidthAttrName(), getUintAttr(context, bus_width));

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

// ==========================================
// ===== Simmani clusterer (prior work) =====
// ==========================================

std::string SimmaniClusterer::ID = "SimmaniClusterer";

SimmaniClusterer::SimmaniClusterer(std::string args, int max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

SimmaniClusterer::~SimmaniClusterer() {}

void SimmaniClusterer::runOnCircuit(firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  std::vector<power_cluster_t> &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {
  double idle_power = *idle_power_ptr;

  *idle_power_ptr = idle_power;
}

// ================================
// ===== Glue logic clusterer =====
// ================================

std::string GlueLogicClusterer::ID = "GlueLogicClusterer";

GlueLogicClusterer::GlueLogicClusterer(std::string args, int max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

GlueLogicClusterer::~GlueLogicClusterer() {}

void GlueLogicClusterer::runOnCircuit(firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  std::vector<power_cluster_t> &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {
  double idle_power = *idle_power_ptr;

  *idle_power_ptr = idle_power;
}

// ==============================================
// ===== Minimize inter-cluster connections =====
// ==============================================

std::string MinimizeConnectionsClusterer::ID = "MinimizeConnectionsClusterer";

MinimizeConnectionsClusterer::MinimizeConnectionsClusterer(std::string args, int max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

MinimizeConnectionsClusterer::~MinimizeConnectionsClusterer() {}

void MinimizeConnectionsClusterer::runOnCircuit(firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  std::vector<power_cluster_t> &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {
  double idle_power = *idle_power_ptr;

  *idle_power_ptr = idle_power;
}

// ================================================================
// ===== Cluster with highest probability of switching output =====
// ================================================================

std::string SwitchingProbClusterer::ID = "SwitchingProbClusterer";

SwitchingProbClusterer::SwitchingProbClusterer(std::string args, int max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

SwitchingProbClusterer::~SwitchingProbClusterer() {}

void SwitchingProbClusterer::runOnCircuit(firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  std::vector<power_cluster_t> &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {
  double idle_power = *idle_power_ptr;

  *idle_power_ptr = idle_power;
}

/*
// ====================================
// ===== Power clusterer template =====
// ====================================

std::string <Power>Clusterer::ID = "<Power>Clusterer";

<Power>Clusterer::<Power>Clusterer(std::string args, int max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

<Power>Clusterer::~<Power>Clusterer() {}

void <Power>Clusterer::runOnCircuit(firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  std::vector<power_cluster_t> &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {
  double idle_power = *idle_power_ptr;

  *idle_power_ptr = idle_power;
}
*/
