#ifndef CIRCT_DIALECT_FIRRTL_COUNTERINSERTER_H
#define CIRCT_DIALECT_FIRRTL_COUNTERINSERTER_H

#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/SmallVector.h"

#include <map>
#include <memory>
#include <string>
#include <z3++.h>

class CounterInserter {
public:
  struct AddedOutputPortInfo {
    unsigned newPortIdx;
    circt::firrtl::InstanceOp rewrittenPathInstance;
  };
  struct ValueOpPairLess {
    bool operator()(const std::pair<mlir::Value, mlir::Operation *> &a,
                    const std::pair<mlir::Value, mlir::Operation *> &b) const {
      if (a.first.getAsOpaquePointer() != b.first.getAsOpaquePointer())
        return a.first.getAsOpaquePointer() < b.first.getAsOpaquePointer();
      return a.second < b.second;
    }
  };

  std::map<std::pair<mlir::Value, mlir::Operation *>, mlir::Value,
           ValueOpPairLess>
      portInsertCache;
  std::map<mlir::Operation *, mlir::Operation *> rewrittenInstanceMap;
  mlir::Operation *module;
  circt::firrtl::InstanceGraph &instanceGraph;
  static unsigned numCountersInserted;
  static unsigned globalPortId;
  std::map<std::string, mlir::Value> insertCache;
  std::map<std::pair<mlir::Operation *, std::string>, unsigned>
      exportedPortCache;

  CounterInserter(mlir::Operation *module,
                  circt::firrtl::InstanceGraph &instanceGraph)
      : module(module), instanceGraph(instanceGraph) {}

  static std::unique_ptr<CounterInserter>
  create(mlir::Operation *module, circt::firrtl::InstanceGraph &instanceGraph);

  void insertPerfCounters(
      z3::expr counterExpr, const std::string &counterName,
      std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap,
      const std::string descr = "", bool onlyDirect = true);
  circt::firrtl::InstanceOp
  resolveCurrentInstance(circt::firrtl::InstanceOp inst);
  mlir::Value makeValueVisibleInModule(
      const circt::firrtl::AnnoPathValue &apv, mlir::Value value,
      circt::firrtl::FModuleLike targetModule, llvm::StringRef debugName,
      std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap);

private:
  bool tryInsertDirectTargetCounter(
      const std::string &counterName, const z3::expr &counterExpr,
      const std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap,
      const std::string descr);

  std::optional<AddedOutputPortInfo> addOutputPortToModuleAndInstances(
      circt::firrtl::FModuleLike module, mlir::Value exportedValue,
      llvm::StringRef debugName, circt::firrtl::InstanceOp pathInstance,
      std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap, circt::firrtl::FModuleLike targetModule);

  llvm::SmallVector<circt::firrtl::FModuleLike, 8>
  getModulePath(const circt::firrtl::AnnoPathValue &apv);

  circt::firrtl::FModuleLike findLowestCommonModule(
      circt::firrtl::CircuitOp circuit,
      llvm::ArrayRef<const circt::firrtl::AnnoPathValue *> paths);
  void
  updateAnnoMap(std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap,
                circt::firrtl::InstanceOp oldInst,
                circt::firrtl::InstanceOp newInst);

  void updatePortInsertCacheValues(circt::firrtl::InstanceOp oldInst,
                                   circt::firrtl::InstanceOp newInst);
};

#endif // CIRCT_DIALECT_FIRRTL_COUNTERINSERTER_H