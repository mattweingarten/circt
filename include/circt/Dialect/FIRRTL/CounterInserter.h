#ifndef CIRCT_DIALECT_FIRRTL_COUNTERINSERTER_H
#define CIRCT_DIALECT_FIRRTL_COUNTERINSERTER_H

#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/Operation.h"

#include <map>
#include <memory>
#include <string>
#include <z3++.h>

class CounterInserter {
public:
  mlir::Operation *module;
  circt::firrtl::InstanceGraph &instanceGraph;

  CounterInserter(mlir::Operation *module,
                  circt::firrtl::InstanceGraph &instanceGraph)
      : module(module), instanceGraph(instanceGraph) {}

  static std::unique_ptr<CounterInserter>
  create(mlir::Operation *module, circt::firrtl::InstanceGraph &instanceGraph);

  void insertPerfCounters(
      z3::expr counterExpr, const std::string &counterName,
      const std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap);
};

#endif // CIRCT_DIALECT_FIRRTL_COUNTERINSERTER_H