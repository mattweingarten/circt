
#ifndef CIRCT_DIALECT_FIRRTL_Power_H
#define CIRCT_DIALECT_FIRRTL_Power_H

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Threading.h"

#include "circt/Dialect/Perf/PerfDialect.h"
#include "circt/Dialect/Perf/PerfHelpers.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "circt/Dialect/Perf/PerfPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <stdint.h>

using namespace mlir;
using namespace circt;
using namespace firrtl;

// =============================
// ===== Utility functions =====
// =============================

inline StringRef getPowerAnnotationAttrName() { return "annotations"; }
inline StringRef getPowerAttrName() { return "power"; }
inline StringRef getRtlPathAttrName() { return "rtl_path"; }
inline StringRef getNumInstancesAttrName() { return "num_instances"; }
inline StringRef getBusWidthAttrName() { return "bus_width"; }
inline StringRef getClusterIdAttrName() { return "cluster_id"; }
inline StringRef getFanOutAttrName() { return "fan_out"; }

// create string attribuet
inline StringAttr getStringAttr(MLIRContext *context, std::string s) {
  return StringAttr::get(context, s);
}

// create float attribute
inline FloatAttr getFloatAttr(MLIRContext *context, double f) {
  return FloatAttr::get(mlir::FloatType::getF64(context), f);
}

// create uint32_t attribute
inline IntegerAttr getUintAttr(MLIRContext *context, uint32_t d) {
  return IntegerAttr::get(
    mlir::IntegerType::get(
      context, 32,
      mlir::IntegerType::Signless),
    d);
}

// traverse the instance graph one level
inline firrtl::InstanceOp find_instance(circt::igraph::InstanceGraphNode *node, std::string inst_name, circt::igraph::InstanceGraphNode **new_node) {
  // get instances from module in node
  for (auto it = node->begin(); it != node->end(); it++) {
    auto inst_node = (*it)->getTarget();
    auto inst = (*it)->getInstance().getOperation();
    auto inst_ptr = dyn_cast<firrtl::InstanceOp>(inst);

    if (std::string(inst_ptr.getName().data()) == inst_name) {
      *new_node = inst_node;
      return inst_ptr;
    }
  }
  *new_node = nullptr;
  return nullptr;
}

// strip register and array index from name
inline std::string strip_misc_name(std::string name, bool *is_bus, bool *is_replicated) {
    // strip bus tag
    int size = name.length();
    int idx = name.rfind("[0]/q");
    std::string ret = name;
    if (idx == size - 5) {
        ret = name.substr(0, idx);
        *is_bus = true;
    }
    else {
        // ret = ret;
        *is_bus = false;
    }

    // strip register tag
    size = ret.length();
    idx = ret.rfind("/q");
    if (idx == size - 2) {
        ret = name.substr(0, idx);
    }

    // determine if one of many instances
    size = ret.length();
    idx = ret.find("[0]");
    *is_replicated = idx != -1;

    return ret;
}

// tag a register operation with fine-grained information in attributes
void tag_register(
  MLIRContext *context,
  mlir::Operation *op,
  std::string hdl_path,
  double power,
  uint32_t num_instances,
  uint32_t bus_width,
  uint32_t cluster_id);

// remove fine-grained information
void untag_register(MLIRContext *context, mlir::Operation *op);

// compute the fan-out of an operation
uint32_t compute_fan_out(mlir::Operation *op);

// =========================
// ===== Utility types =====
// =========================

// power node
typedef std::tuple<bool, firrtl::RegOp, firrtl::RegResetOp, mlir::Operation*> reg_node_t;
#define REG_NODE_T(is_reset, reg_op, reg_reset_op, op) reg_node_t{is_reset, reg_op, reg_reset_op, op}
#define REG_NODE_T_IS_RESET(node) std::get<0>(node)
#define REG_NODE_T_GET_REG_OP(node) std::get<1>(node)
#define REG_NODE_T_GET_REG_RESET_OP(node) std::get<2>(node)
#define REG_NODE_T_GET_OP(node) std::get<3>(node)

template <typename AttrType, typename ValueType>
inline ValueType get_reg_node_attr(reg_node_t node, StringRef attr_name) {
  return llvm::dyn_cast_or_null<AttrType>(
    REG_NODE_T_GET_OP(node)->getAttr(attr_name)
  ).getValue();
}

// power cluster
typedef struct {

  //firrtl::FModuleOp module;
  uint32_t id;
  double power;
  std::string name;
  std::vector<reg_node_t> nodes;
  uint32_t indicator_idx;
  uint32_t indicator_fanout;
  circt::firrtl::AnnoPathValue indicator_path;

} power_cluster_t;
typedef std::unordered_map<uint32_t, power_cluster_t> power_clusters_t;

// ===========================
// ===== Clusterer types =====
// ===========================

// base class providing an API to various clustering algorithms
class PowerClusterer {

protected:

  uint32_t max_num_clusters;
  std::string args;

  virtual void parseArgs(std::string args);

public:

  PowerClusterer(std::string args, uint32_t max_num_clusters);
  PowerClusterer();
  virtual ~PowerClusterer();

  virtual void runOnCircuit(MLIRContext *context,
    firrtl::CircuitOp circuit,
    circt::igraph::InstanceGraph *inst_graph,
    power_clusters_t &clusters,
    double *idle_power_ptr,
    std::ofstream &log_stream
    );

  static PowerClusterer *create(std::string clustering_alg, std::string args, uint32_t max_num_clusters);

};

// Select the maximum power signals
class MaxPowerClusterer : public PowerClusterer {

public:

  static std::string ID;

  MaxPowerClusterer(std::string args, uint32_t max_num_clusters);
  ~MaxPowerClusterer();

  void runOnCircuit(MLIRContext *context,
    firrtl::CircuitOp circuit,
    circt::igraph::InstanceGraph *inst_graph,
    power_clusters_t &clusters,
    double *idle_power_ptr,
    std::ofstream &log_stream
    ) override;

};

// Select the highest power glue logic between the top-level modules. Each cluster is a set of modules.
class GlueLogicClusterer : public PowerClusterer {

public:

  static std::string ID;

  GlueLogicClusterer(std::string args, uint32_t max_num_clusters);
  ~GlueLogicClusterer();

  void runOnCircuit(MLIRContext *context,
    firrtl::CircuitOp circuit,
    circt::igraph::InstanceGraph *inst_graph,
    power_clusters_t &clusters,
    double *idle_power_ptr,
    std::ofstream &log_stream
    ) override;

};

// Minimize inter-cluster connections
class MinimizeConnectionsClusterer : public PowerClusterer {

public:

  static std::string ID;

  MinimizeConnectionsClusterer(std::string args, uint32_t max_num_clusters);
  ~MinimizeConnectionsClusterer();

  void runOnCircuit(MLIRContext *context,
    firrtl::CircuitOp circuit,
    circt::igraph::InstanceGraph *inst_graph,
    power_clusters_t &clusters,
    double *idle_power_ptr,
    std::ofstream &log_stream
    ) override;

};

// Cluster based on highest probability of switching through the logic function.
class SwitchingProbClusterer : public PowerClusterer {

public:

  static std::string ID;

  SwitchingProbClusterer(std::string args, uint32_t max_num_clusters);
  ~SwitchingProbClusterer();

  void runOnCircuit(MLIRContext *context,
    firrtl::CircuitOp circuit,
    circt::igraph::InstanceGraph *inst_graph,
    power_clusters_t &clusters,
    double *idle_power_ptr,
    std::ofstream &log_stream
    ) override;

};

// Use the clustering provided by Simmani, prior work in power modeling
class SimmaniClusterer : public PowerClusterer {

protected:

  //void parseArgs(std::string args) override;

public:

  static std::string ID;

  SimmaniClusterer(std::string args, uint32_t max_num_clusters);
  ~SimmaniClusterer();

  void runOnCircuit(MLIRContext *context,
    firrtl::CircuitOp circuit,
    circt::igraph::InstanceGraph *inst_graph,
    power_clusters_t &clusters,
    double *idle_power_ptr,
    std::ofstream &log_stream
    ) override;

};

#endif // CIRCT_DIALECT_FIRRTL_Power_H
