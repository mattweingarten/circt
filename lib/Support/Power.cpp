
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
    //for (Operation *userOp : result.getUsers()) {
    for (auto _ : result.getUsers()) {
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
  uint32_t cluster_id,
  uint32_t *indicator_fan_out
) {
  uint32_t fan_out = compute_fan_out(op);
  if (indicator_fan_out) {
    *indicator_fan_out = fan_out;
  }
  op->setAttr(getFanOutAttrName(), getUintAttr(context, fan_out));
  op->setAttr(getRtlPathAttrName(), getStringAttr(context, hdl_path));
  op->setAttr(getPowerAttrName(), getFloatAttr(context, power));
  op->setAttr(getNumInstancesAttrName(), getUintAttr(context, num_instances));
  op->setAttr(getBusWidthAttrName(), getUintAttr(context, bus_width));
  op->setAttr(getClusterIdAttrName(), getUintAttr(context, cluster_id));
}

void untag_register(MLIRContext *context, mlir::Operation *op) {
  op->removeAttr(getFanOutAttrName());
  op->removeAttr(getRtlPathAttrName());
  op->removeAttr(getPowerAttrName());
  op->removeAttr(getNumInstancesAttrName());
  op->removeAttr(getBusWidthAttrName());
  op->removeAttr(getClusterIdAttrName());
}

inline uint32_t get_register_cluster_id(mlir::Operation *op) {
  if (op->hasAttr(getClusterIdAttrName())) {
    return (uint32_t)(llvm::dyn_cast_or_null<IntegerAttr>(
      op->getAttr(getClusterIdAttrName())
    ).getValue().getZExtValue());
  } else {
    return 0;
  }
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
void compute_fan_out(power_cluster_t cluster) {

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
      llvm::outs() << "    (" << i << ") - " << userOp->getName().getStringRef().data()  << "\n";
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

// get the width of a signal in the MLIR
// some signals get flattened from Uint<n> to one-bit signals
//   so when they are marked with bus_width = 1, the Uint
//   may not be a Uint<1>
// so we need to confirm that anything tagged with bus_width = 1
//   is indeed a Uint<1>
uint32_t get_register_bus_width(Operation *op/*, std::ofstream &log_stream*/) {

  uint32_t bus_width = 0;
  if (op->hasAttr(getBusWidthAttrName())) {
    bus_width = (uint32_t)llvm::dyn_cast_or_null<IntegerAttr>(
      op->getAttr(getBusWidthAttrName())
    ).getValue().getZExtValue();
  }
  else {
    return 0;
  }

  // bus width = 1
  //uint32_t bus_width = get_register_bus_width(op);
  //log_stream << "  Bus width for operation " << op->getName().getStringRef().data() << " is " << bus_width << std::endl;
  if (bus_width == 1) {
    for (auto indexedResult : llvm::enumerate(op->getResults())) {
      auto value = indexedResult.value();

      auto uintType = llvm::dyn_cast<circt::firrtl::UIntType>(value.getType());
      if (uintType) {
        return uintType.getWidthOrSentinel();
      }
      else {
        return 0;
      }
    }
  }

  return bus_width;
}

Operation *find_one_bit_register(Operation *op, std::ofstream &log_stream) {
  if (op) {
    log_stream << "  find_one_bit_register for operation ";
    mlir::OperationName op_name = op->getName();
    if (op_name.getContext()) {
      log_stream << "with context" << std::flush;
      log_stream << op->getName().getStringRef().data() << std::endl;
    }
    else {
      log_stream << "    No name" << std::endl;
    }
  }
  else {
    log_stream << "  op is null!" << std::endl;
  }
  for (auto indexedOperand : op->getOperands()) {
    // get defining operation for each operand
    //log_stream << "  index is " << indexedOperand.getOperandNumber() << std::endl;
    //auto value = indexedOperand.value();
    log_stream << "  value" << std::flush;
    //auto defining_op = value.getDefiningOp();
    auto defining_op = indexedOperand.getDefiningOp();
    if (defining_op) {
      log_stream << " defined by " << defining_op->getName().getStringRef().data() << std::endl;
    }
    else {
      log_stream << "  no defining operation" << std::endl;
      continue;
    }

    /*
    // classify register
    uint32_t register_bits = TypeSwitch<Operation *, uint32_t>(defining_op)
      .template Case<RegOp>([&](auto reg_op) -> uint32_t {

      })
      .template Case<RegResetOp>([&](auto reg_reset_op) -> uint32_t {

      })
      .Default([&](auto expr) -> bool {
        return false;
      });

    if (is register) {
      if (is one bit register) {
        return defining_op;
      }
      else if (is multi bit register) {
        return nullptr;
      }
    }
    */

    // determine bus width
    // if unknown, traverse down the combinational path
    // if known, return if a single bit
    // if known
    uint32_t bus_width = get_register_bus_width(defining_op);
    log_stream << "  Bus width for operation " << defining_op->getName().getStringRef().data() << " is " << bus_width << std::endl;
    if (bus_width == 1) {
      return defining_op;
      break;
    }
    else if (bus_width > 1) {
      // another vector, so skip this path
      continue;
    }

    // traverse one more level
    Operation *ancestor_defining_op = find_one_bit_register(defining_op, log_stream);
    if (ancestor_defining_op) {
      return ancestor_defining_op;
    }
  }
  return nullptr;
}

static uint32_t merge_clusters_ids(MLIRContext *context, power_clusters_t &clusters, uint32_t id0, uint32_t id1, uint32_t *num_clusters) {

  if (id0 == 0 || id1 == 0) {
    return 0;
  }

  power_cluster_t c0 = clusters[id0];
  power_cluster_t c1 = clusters[id1];
  power_cluster_t flattened;

  // add registers in clusters[idx1] to clusters[idx0]
  flattened.nodes.reserve(c0.nodes.size() + c1.nodes.size());
  flattened.nodes.insert(flattened.nodes.end(), c0.nodes.begin(), c0.nodes.end());
  flattened.nodes.insert(flattened.nodes.end(), c1.nodes.begin(), c1.nodes.end());

  // select indicator with a higher fan_out
  bool select_c0 =
    (
      c0.indicator_bus_width < c1.indicator_bus_width
    ) || (
      c0.indicator_fan_out >= c1.indicator_fan_out
      && c0.indicator_bus_width == c1.indicator_bus_width
    );

  // fill in flattened metadata
  flattened.id = id0;
  if (select_c0) {
    flattened.name = c0.name;
    flattened.indicator_idx = c0.indicator_idx;
    flattened.indicator_fan_out = c0.indicator_fan_out;
    flattened.indicator_path = c0.indicator_path;
  }
  else {
    flattened.name = c1.name;
    flattened.indicator_idx = (uint32_t)c0.nodes.size() + c1.indicator_idx;
    flattened.indicator_fan_out = c1.indicator_fan_out;
    flattened.indicator_path = c1.indicator_path;
  }

  // update cluster ID metadata
  for (reg_node_t node : flattened.nodes) {
    REG_NODE_T_GET_OP(node)->setAttr(getClusterIdAttrName(), getUintAttr(context, id0));
  }

  // remove separate node
  clusters[id0] = flattened;
  clusters.erase(id1);
  *num_clusters = *num_clusters - 1;
  return id0;
}

static uint32_t merge_clusters_ops(MLIRContext *context, power_clusters_t &clusters, Operation *reg0, Operation *reg1, uint32_t *num_clusters) {
  return merge_clusters_ids(
    context,
    clusters,
    get_register_cluster_id(reg0),
    get_register_cluster_id(reg1),
    num_clusters
  );
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

  // get keys
  std::vector<uint32_t> keys;
  keys.reserve(num_clusters);
  for (auto entry : clusters) {
    keys.push_back(entry.first);
  }

  // remove RegOps until bug with reset reference is fixed
  for (uint32_t id : keys) {
    auto entry_it = clusters.find(id);
    if (entry_it == clusters.end()) {
      continue;
    }
    power_cluster_t entry = entry_it->second;

    reg_node_t node = entry.nodes[entry.indicator_idx];
    if (!REG_NODE_T_IS_RESET(node)) {
      Operation *op = REG_NODE_T_GET_OP(node);
      uint32_t bus_width = get_register_bus_width(op);
      idle_power +=
          entry.power * bus_width * VECTOR_DEFAULT_SWITCHING_FACTOR;
      clusters.erase(id);
      --num_clusters;
    }
  }

  // remove vector clusters
  for (uint32_t id : keys) {
    // lookup entry in list of clusters
    log_stream << "Looking up id " << id << std::endl;
    auto entry_it = clusters.find(id);
    if (entry_it == clusters.end()) {
      continue;
    }
    power_cluster_t entry = entry_it->second;
    log_stream << "Found entry with name " << entry.name << std::endl;

    // ignore cluster when a vector
    reg_node_t node = entry.nodes[entry.indicator_idx];
    Operation *op = REG_NODE_T_GET_OP(node);
    uint32_t bus_width = get_register_bus_width(op);
    log_stream << "Bus width for " << entry.name << " is " << bus_width << std::endl;
    if (bus_width > 1) {

      log_stream << "  Found vector cluster starting with " << op->getName().getStringRef().data() << ": " << entry.name << std::endl;

      // traverse results until find logic with one-bit
      Operation *single_bit_merge = nullptr;
      for (auto indexedResult : llvm::enumerate(op->getResults())) {
        auto result = indexedResult.value();

        // for each user of the vector result
        for (Operation *userOp : result.getUsers()) {

          // walk back to find a one-bit register
          // for each operand in the user of the result
          log_stream << "  Looking at operation " << userOp->getName().getStringRef().data() << std::endl;
          single_bit_merge = find_one_bit_register(userOp, log_stream);
          if (single_bit_merge) {
            break;
          }
        }
      }

      // if found something to merge, merge
      if (single_bit_merge) {
        log_stream << "    Merging" << std::endl;
        uint32_t new_id = merge_clusters_ops(context, clusters, op, single_bit_merge, &num_clusters);
        if (new_id) {
          log_stream << "    Merged into cluster with id " << new_id << std::endl;
          continue;
        } else {
          log_stream << "    Unable to merge" << std::endl;
        }
      }

      // otherwise keep as idle
      log_stream << "    Could not find ancestor one-bit register" << std::endl;
      idle_power +=
        entry.power * bus_width * VECTOR_DEFAULT_SWITCHING_FACTOR;
      clusters.erase(id);
      --num_clusters;
    }
  }

  log_stream << "Number of clusters to start after removing vectors: " << num_clusters << " (" << (uint32_t)clusters.size() << "), must minimize to " << max_num_clusters << "\n";
  // remove min-power clusters
  uint32_t kept_id;
  num_clusters = (uint32_t)clusters.size();
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

    log_stream << "  After removing " << min_power_id << ": " << num_clusters << " (" << (uint32_t)clusters.size() << "), must minimize to " << max_num_clusters << "\n";
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
