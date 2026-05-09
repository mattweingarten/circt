
#include <iostream>
#include <fstream>
#include <limits>
#include <stdint.h>

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

// ======================================
// ===== Circuit-specific functions =====
// ======================================

uint32_t compute_fan_out(mlir::Operation *op) {
  uint32_t fan_out = 0;
  for (auto indexedResult : llvm::enumerate(op->getResults())) {
    auto result = indexedResult.value();
    for (Operation *userOp : result.getUsers()) {
      ++fan_out;
    }
  }

  return fan_out;
}

void tag_register(
  MLIRContext *context,
  mlir::Operation *op,
  std::string hdl_path,
  double power,
  uint32_t num_instances,
  uint32_t bus_width,
  uint32_t cluster_id
) {
  uint32_t fan_out = compute_fan_out(op);
  op->setAttr(getFanOutAttrName(), getUintAttr(context, fan_out));
  op->setAttr(getHdlPathAttrName(), getStringAttr(context, hdl_path));
  op->setAttr(getPowerAttrName(), getFloatAttr(context, power));
  op->setAttr(getNumInstancesAttrName(), getUintAttr(context, num_instances));
  op->setAttr(getBusWidthAttrName(), getUintAttr(context, bus_width));
  op->setAttr(getClusterIdAttrName(), getUintAttr(context, cluster_id));
}

void untag_register(MLIRContext *context, mlir::Operation *op) {
  op->removeAttr(getFanOutAttrName());
  op->removeAttr(getHdlPathAttrName());
  op->removeAttr(getPowerAttrName());
  op->removeAttr(getNumInstancesAttrName());
  op->removeAttr(getBusWidthAttrName());
  op->removeAttr(getClusterIdAttrName());
}

// ========================================
// ===== Switching-specific functions =====
// ========================================

#define APPLY(OPTYPE, func) \
  bool apply_expr(OPTYPE op, uint32_t inputs) { \
    return func; \
  }

APPLY(Operation*, 0)
APPLY(AndPrimOp, inputs == 0b11)
APPLY(OrPrimOp, inputs ? true : false)
APPLY(XorPrimOp, (inputs & 0b01) ^ (inputs & 0b10))
APPLY(ConstantOp, 0)

            /*ConstantOp, SpecialConstantOp, AggregateConstantOp, InvalidValueOp,
            SubfieldOp, SubindexOp, SubaccessOp, IsTagOp, SubtagOp,
            BundleCreateOp, VectorCreateOp, FEnumCreateOp, MultibitMuxOp,
            TagExtractOp, OpenSubfieldOp, OpenSubindexOp, ObjectSubfieldOp,
            // Arithmetic and Logical Binary Primitives.
            AddPrimOp, SubPrimOp, MulPrimOp, DivPrimOp, RemPrimOp, AndPrimOp,
            OrPrimOp, XorPrimOp,
            // Elementwise operations,
            ElementwiseOrPrimOp, ElementwiseAndPrimOp, ElementwiseXorPrimOp,
            // Comparisons.
            LEQPrimOp, LTPrimOp, GEQPrimOp, GTPrimOp, EQPrimOp, NEQPrimOp,
            // Misc Binary Primitives.
            CatPrimOp, DShlPrimOp, DShlwPrimOp, DShrPrimOp,
            // Unary operators.
            AsSIntPrimOp, AsUIntPrimOp, AsAsyncResetPrimOp, AsClockPrimOp,
            CvtPrimOp, NegPrimOp, NotPrimOp, AndRPrimOp, OrRPrimOp, XorRPrimOp,
            // Intrinsic Expressions.
            IsXIntrinsicOp, PlusArgsValueIntrinsicOp, PlusArgsTestIntrinsicOp,
            SizeOfIntrinsicOp, ClockGateIntrinsicOp, ClockInverterIntrinsicOp,
            ClockDividerIntrinsicOp, LTLAndIntrinsicOp, LTLOrIntrinsicOp,
            LTLDelayIntrinsicOp, LTLConcatIntrinsicOp, LTLNotIntrinsicOp,
            LTLImplicationIntrinsicOp, LTLEventuallyIntrinsicOp,
            LTLClockIntrinsicOp, LTLDisableIntrinsicOp, Mux2CellIntrinsicOp,
            Mux4CellIntrinsicOp, HasBeenResetIntrinsicOp,
            // Miscellaneous.
            BitsPrimOp, HeadPrimOp, MuxPrimOp, PadPrimOp, ShlPrimOp, ShrPrimOp,
            TailPrimOp, VerbatimExprOp, HWStructCastOp, BitCastOp, RefSendOp,
            RefResolveOp, RefSubOp, RWProbeOp, XMRRefOp, XMRDerefOp,
            // Casts to deal with weird stuff
            UninferredResetCastOp, ConstCastOp, RefCastOp,
            // Property expressions.
            StringConstantOp, FIntegerConstantOp, BoolConstantOp,
            DoubleConstantOp, ListCreateOp, UnresolvedPathOp, PathOp>*/
bool apply_op(Operation *op, uint32_t inputs) {
  return TypeSwitch<Operation *, bool>(op)
        // Basic Expressions
        .template Case<
            AndPrimOp, OrPrimOp, XorPrimOp, ConstantOp>(
            [&](auto expr) -> bool {
              return apply_expr(expr, inputs);
            })
        .Default([&](auto expr) -> bool {
          return false;
        });
}

/*
void compute_fanout(power_cluster_t cluster) {

  // for each register in the cluster
  for (reg_node_t node : cluster.nodes) {

    Operation *op = REG_NODE_T_GET_OP(node);

  }

}
*/

void walk_register(power_cluster_t cluster, std::ofstream &log_stream) {

  //llvm::outs() << "Walking register in cluster " << cluster.name << "\n";

  Operation *op = REG_NODE_T_GET_OP(cluster.nodes[cluster.indicator_idx]);

  //llvm::outs() << "Has " << op->getNumResults() << " results:\n";
  for (auto indexedResult : llvm::enumerate(op->getResults())) {
    auto result = indexedResult.value();
    //llvm::outs() << "  - Result " << indexedResult.index();
    //if (result.use_empty()) {
    //  llvm::outs() << " has no uses\n";
    //  continue;
    //}
    //if (result.hasOneUse())
    //  llvm::outs() << " has a single use: ";
    //else
    //  llvm::outs() << " has multiple uses:\n";
    int i = 0;
    for (Operation *userOp : result.getUsers()) {
      llvm::outs() << "    (" << i << ") - " << userOp->getName() << "\n";
      ++i;
    }
  }

}

void compute_switching_factor(Operation *op, uint32_t num_inputs, uint32_t num_outputs, std::vector<double> &sw_factor_inputs, std::vector<double> &sw_factor_outputs) {

  uint32_t num_states = 1 << num_inputs;

  for (uint32_t i = 0; i < num_outputs; ++i) {

    double switching_factor = 0.0;

    // base output table
    std::vector<bool> base_output = std::vector<bool>(num_states);

    // compute base state (all inputs are zero)
    //DEBUG("Base outputs: ");
    for (uint32_t j = 0; j < num_states; ++j) {
      bool val = apply_op(op, j);
      base_output[j] = val;
      //DEBUGF("  %08x => %01x\n", j, val);
    }

    // for each switch (j = 0 => no switches)
    // switching_factor += 0.0 * PI(1 - alpha(in_j))
    for (uint32_t j = 1; j < num_states; ++j) {

      // compute switching probability for this switch based on switching factor of each input bit
      // TODO make list of these and reuse for each output
      double switching_probability = 1.0;
      uint32_t switch_mask = j;
      for (uint32_t k = 0; k < num_inputs; k++, switch_mask >>= 1) {
        // get probability of k-th input switching
        //double input_factor = DEFAULT_SWITCHING;
        double input_factor = 0.0;
        if (sw_factor_inputs[k] >= 0.0) {
          input_factor = sw_factor_inputs[k];
        }

        // determine if the current switch (j) includes the k-th input
        if (switch_mask & 0b1) {
          //DEBUG("  Input " << k << " has switching probability of " << input_factor);
          switching_probability *= input_factor;
        }
        else {
          //DEBUG("  Input " << k << " has non-switching probability of " << (1 - input_factor));
          switching_probability *= (1 - input_factor);
        }
      }
      //DEBUGF("Switch %x: \n", j);
      //DEBUG("  Probability of switch = " << switching_probability);

      // apply switch to each input combination
      uint32_t num_switches = 0;
      for (uint32_t k = 0; k < num_states; ++k) {
        // apply switch
        uint32_t new_input = j ^ k;
        // determine if output switches
        if (apply_op(op, new_input) != base_output[k]) {
          ++num_switches;
        }
      }

      // compute probability that the output switches given the input switch combination
      double switching_contribution = (double)num_switches;
      //std::cout << "  Number of switches in output = " << num_switches << std::endl;
      switching_contribution /= (double)num_states;
      //std::cout << "  Output switching probability = " << switching_contribution << std::endl;
      switching_contribution *= switching_probability;
      //std::cout << "  Per-switch output switching probability = " << switching_contribution << std::endl;
      switching_factor += switching_contribution;
      //std::cout << "  Post-switch switching factor = " << switching_factor << std::endl;

    }

    sw_factor_outputs[i] = switching_factor;

  }

}

// ======================================
// ===== Cluster-specific functions =====
// ======================================

static uint32_t merge_clusters(power_clusters_t &clusters, uint32_t id0, uint32_t id1, uint32_t num_clusters) {

  //if (idx0 == idx1) {
  //  return idx0;
  //}
  //else if (idx0 > idx1) {
  //  // we want idx0 < idx1, so swap
  //  idx0 = idx0 ^ idx1; // a ^ b
  //  idx1 = idx0 ^ idx1; // (a ^ b) ^ b = a
  //  idx0 = idx0 ^ idx1; // (a ^ b) ^ a = b
  //}

  power_cluster_t c0 = clusters[id0];
  power_cluster_t c1 = clusters[id1];
  power_cluster_t flattened;

  // add registers in clusters[idx1] to clusters[idx0]
  flattened.nodes.reserve(c0.nodes.size() + c1.nodes.size());
  flattened.nodes.insert(flattened.nodes.end(), c0.nodes.begin(), c0.nodes.end());
  flattened.nodes.insert(flattened.nodes.end(), c1.nodes.begin(), c1.nodes.end());

  // select indicator with a higher fanout
  if (c0.indicator_fanout >= c1.indicator_fanout) {
    flattened.id = id0;
    flattened.name = c0.name;
    flattened.indicator_idx = c0.indicator_idx;
    flattened.indicator_fanout = c0.indicator_fanout;
    flattened.indicator_path = c0.indicator_path;

    clusters[id0] = flattened;
    clusters.erase(id1);
    return id0;
  }
  else {
    flattened.id = id1;
    flattened.name = c1.name;
    flattened.indicator_idx = (uint32_t)c0.nodes.size() + c1.indicator_idx;
    flattened.indicator_fanout = c1.indicator_fanout;
    flattened.indicator_path = c1.indicator_path;

    clusters[id1] = flattened;
    clusters.erase(id0);
    return id1;
  }

  // shift over merged cluster
  //for (uint32_t i = idx1; i < num_clusters; ++i) {
  //  clusters[i] = clusters[i + 1];
  //}
  //clusters.resize(clusters.size()-1);
  //clusters[idx0] = flattened;

  //return idx0;
}

// factory method
PowerClusterer *PowerClusterer::create(std::string clustering_alg, std::string args, uint32_t max_num_clusters) {
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

/*
* Add clusters together that: have more intermediate connections than new inputs/outputs when combined. Continue merging clusters until hit interface boundary of 32 input 32 output as registers. Can then experiment with a decomposition on that level
    * Start clustering in circt and map
    * Module-oblivious optimizations to: reduce gate count, optimize power by reducing switching internal to cluster
    * Find commonalities among these N-bit input to N-bit output regions so can copy the implementations
    * Favor signals likely to go high (OR/XOR gates instead of AND as AND gates can be masks)
    * Stop clustering at voltage islands
*/

// constructor
PowerClusterer::PowerClusterer(std::string args, uint32_t max_num_clusters)
  : max_num_clusters(max_num_clusters)
{
  parseArgs(args);
}

PowerClusterer::~PowerClusterer() {}

// parse clustering arguments
void PowerClusterer::parseArgs(std::string args) {
  this->args = args;
}

void PowerClusterer::runOnCircuit(
  MLIRContext *context,
  firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  power_clusters_t &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {

}

// ===============================
// ===== Max power clusterer =====
// ===============================

std::string MaxPowerClusterer::ID = "MaxPowerClusterer";

MaxPowerClusterer::MaxPowerClusterer(std::string args, uint32_t max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

MaxPowerClusterer::~MaxPowerClusterer() {}

void MaxPowerClusterer::runOnCircuit(
  MLIRContext *context,
  firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  power_clusters_t &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {
  double idle_power = *idle_power_ptr;

  uint32_t num_clusters = (uint32_t)clusters.size();
  log_stream << "Number of clusters to start: " << num_clusters << ", must minimize to " << max_num_clusters << "\n";

  // remove vector clusters
  for (auto map : clusters) {
    int id = map.first;
    power_cluster_t entry = map.second;

    // ignore cluster when a vector
    APInt bus_width = get_reg_node_attr<IntegerAttr, APInt>(entry.nodes[entry.indicator_idx], getBusWidthAttrName());
    if (bus_width.ugt(1)) { // bus_width > 1
      clusters.erase(id);
      --num_clusters;
    }
  }

  // remove min-power clusters
  uint32_t kept_id;
  while (num_clusters > max_num_clusters) {
    uint32_t min_power_id = 0;
    double min_power = std::numeric_limits<double>::max();

    // find minimum power
    for (auto map : clusters) {
      if (map.second.power < min_power) {
        min_power_id = map.first;
        min_power = map.second.power;
      }
      else {
        kept_id = map.first;
      }
    }

    // remove
    if (!min_power_id) {
      break;
    }
    idle_power += min_power;
    clusters.erase(min_power_id);
    --num_clusters;
  }

  // temporary test
  walk_register(clusters[kept_id], log_stream);

  *idle_power_ptr = idle_power;
}

// ==========================================
// ===== Simmani clusterer (prior work) =====
// ==========================================

std::string SimmaniClusterer::ID = "SimmaniClusterer";

SimmaniClusterer::SimmaniClusterer(std::string args, uint32_t max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

SimmaniClusterer::~SimmaniClusterer() {}

void SimmaniClusterer::runOnCircuit(
  MLIRContext *context,
  firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  power_clusters_t &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {
  double idle_power = *idle_power_ptr;

  // open Simmani clusters
  std::ifstream simmani_clusters(this->args);

  // TODO take function out of PowerTrace.cpp to map path to register

  *idle_power_ptr = idle_power;
}

// ================================
// ===== Glue logic clusterer =====
// ================================

std::string GlueLogicClusterer::ID = "GlueLogicClusterer";

GlueLogicClusterer::GlueLogicClusterer(std::string args, uint32_t max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

GlueLogicClusterer::~GlueLogicClusterer() {}

void GlueLogicClusterer::runOnCircuit(
  MLIRContext *context,
  firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  power_clusters_t &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {
  double idle_power = *idle_power_ptr;

  // group vectors with control logic
  //for

  // select maximum power clusters
  MaxPowerClusterer sub_clusterer("", max_num_clusters);
  sub_clusterer.runOnCircuit(context, circuit, inst_graph, clusters, &idle_power, log_stream);

  *idle_power_ptr = idle_power;
}

// ==============================================
// ===== Minimize inter-cluster connections =====
// ==============================================

std::string MinimizeConnectionsClusterer::ID = "MinimizeConnectionsClusterer";

MinimizeConnectionsClusterer::MinimizeConnectionsClusterer(std::string args, uint32_t max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

MinimizeConnectionsClusterer::~MinimizeConnectionsClusterer() {}

void MinimizeConnectionsClusterer::runOnCircuit(
  MLIRContext *context,
  firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  power_clusters_t &clusters,
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

SwitchingProbClusterer::SwitchingProbClusterer(std::string args, uint32_t max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

SwitchingProbClusterer::~SwitchingProbClusterer() {}

void SwitchingProbClusterer::runOnCircuit(
  MLIRContext *context,
  firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  power_clusters_t &clusters,
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

<Power>Clusterer::<Power>Clusterer(std::string args, uint32_t max_num_clusters)
  : PowerClusterer(args, max_num_clusters) {

}

<Power>Clusterer::~<Power>Clusterer() {}

void <Power>Clusterer::runOnCircuit(firrtl::CircuitOp circuit,
  circt::igraph::InstanceGraph *inst_graph,
  power_clusters_t &clusters,
  double *idle_power_ptr,
  std::ofstream &log_stream
) {
  double idle_power = *idle_power_ptr;

  *idle_power_ptr = idle_power;
}
*/
