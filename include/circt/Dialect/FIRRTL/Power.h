
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

using namespace mlir;
using namespace circt;
using namespace firrtl;

inline StringRef getPowerAnnotationAttrName() { return "annotations"; }
inline StringRef getPowerAttrName() { return "power"; }
inline StringRef getNumInstancesAttrName() { return "num_instances"; }
inline StringRef getBusWidthAttrName() { return "bus_width"; }

// create float attribute
inline FloatAttr getFloatAttr(MLIRContext *context, double f) { return FloatAttr::get(mlir::FloatType::getF64(context), f); }
inline IntegerAttr getUintAttr(MLIRContext *context, uint32_t d) {
  return IntegerAttr::get(
    mlir::IntegerType::get(
      context, 32,
      mlir::IntegerType::Signless),
    d);
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


// power node
typedef std::tuple<bool, firrtl::RegOp, firrtl::RegResetOp> reg_node_t;
#define REG_NODE_T(is_reset, reg_op, reg_reset_op) std::tuple<bool, firrtl::RegOp, firrtl::RegResetOp>{is_reset, reg_op, reg_reset_op}
#define REG_NODE_T_IS_RESET(n) std::get<0>(n)
#define REG_NODE_T_GET_REG_OP(n) std::get<1>(n)
#define REG_NODE_T_GET_REG_RESET_OP(n) std::get<2>(n)

// power cluster
typedef struct {

  firrtl::FModuleOp module;
  double power;
  std::string name;
  std::vector<reg_node_t> nodes;
  int indicator_idx;

} power_cluster_t;

// base class providing an API to various clustering
class PowerClusterer {

private:

  void parseArgs(std::string args);

protected:

  int max_num_clusters;

public:

  PowerClusterer(std::string args, int max_num_clusters);
  PowerClusterer();
  virtual ~PowerClusterer();

  virtual void runOnCircuit(firrtl::CircuitOp circuit,
    circt::igraph::InstanceGraph *inst_graph,
    std::vector<power_cluster_t> &clusters,
    double *idle_power_ptr,
    std::ofstream &log_stream
    );

};

class MaxPowerClusterer : public PowerClusterer {

public:

  static std::string ID;

  MaxPowerClusterer(std::string args, int max_num_clusters);
  ~MaxPowerClusterer();

  void runOnCircuit(firrtl::CircuitOp circuit,
    circt::igraph::InstanceGraph *inst_graph,
    std::vector<power_cluster_t> &clusters,
    double *idle_power_ptr,
    std::ofstream &log_stream
    ) override;

};

#endif // CIRCT_DIALECT_FIRRTL_Power_H
