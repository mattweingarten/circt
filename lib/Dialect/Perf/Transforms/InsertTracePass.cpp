//===- InsertTracePass.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/Perf/PerfHelpers.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "circt/Dialect/Perf/PerfPasses.h"
#include "circt/Support/InstanceGraphInterface.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "perf-insert-trace"

namespace circt {
namespace perf {
#define GEN_PASS_DEF_INSERTTRACE
#include "circt/Dialect/Perf/PerfPasses.h.inc"
} // namespace perf
} // namespace circt

using namespace circt;
using namespace circt::perf;

namespace {
struct InsertTracePass
    : public circt::perf::impl::InsertTraceBase<InsertTracePass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<circt::perf::PerfDialect>();
    registry.insert<circt::firrtl::FIRRTLDialect>();
  }

private:
  void runOnOperation() override;

  mlir::LogicalResult
  insertTraceForTarget(circt::firrtl::CircuitOp circuit,
                       mlir::SymbolTable &symbolTable,
                       circt::firrtl::CircuitTargetCache &targetCache,
                       llvm::StringRef rawTarget);
};
} // namespace

mlir::LogicalResult InsertTracePass::insertTraceForTarget(
    circt::firrtl::CircuitOp circuit, mlir::SymbolTable &symbolTable,
    circt::firrtl::CircuitTargetCache &targetCache, llvm::StringRef rawTarget) {
  rawTarget = rawTarget.trim();

  if (rawTarget.empty())
    return mlir::success();

  if (!rawTarget.starts_with("~")) {
    return circuit.emitError()
           << "[PERF] perf-insert-trace target must use FIRRTL annotation "
              "target syntax"
           << rawTarget;
  }

  auto pathValue =
      circt::firrtl::resolvePath(rawTarget, circuit, symbolTable, targetCache);

  if (!pathValue) {
    return circuit.emitError()
           << "[PERF] failed to resolve FIRRTL annotation target: "
           << rawTarget;
  }

  if (pathValue->isOpOfType<circt::firrtl::CircuitOp>() ||
      pathValue->isOpOfType<circt::firrtl::FModuleOp>() ||
      pathValue->isOpOfType<circt::firrtl::FExtModuleOp>()) {
    return circuit.emitError()
           << "[PERF] perf-insert-trace target must resolve to a signal-like "
              "operation, not a circuit or module target: "
           << rawTarget;
  }

  if (llvm::isa<circt::firrtl::PortAnnoTarget>(pathValue->ref)) {
    return circuit.emitError()
           << "[PERF] perf-insert-trace does not currently support port "
              "targets: "
           << rawTarget;
  }

  llvm::StringRef label = rawTarget;

  if (auto opRef =
          llvm::dyn_cast<circt::firrtl::OpAnnoTarget>(pathValue->ref)) {
    if (auto nameAttr =
            opRef.getOp()->getAttrOfType<mlir::StringAttr>("name")) {
      if (!nameAttr.getValue().empty())
        label = nameAttr.getValue();
    }
  }

  LLVM_DEBUG(
      llvm::dbgs() << "[PERF] Inserting PerfTraceOp for annotation target: "
                   << rawTarget << " label=" << label << "\n");

  if (!circt::perf::FIRRTLPerfInserter::insertTraceOp(*pathValue,
                                                      /*label=*/label)) {
    return circuit.emitError()
           << "[PERF] failed to insert PerfTraceOp for target: " << rawTarget;
  }
  ++numTracesInserted;

  if (numTracesInserted >= maxTraceCount) {
    return circuit.emitError()
           << "[PERF] perf-insert-trace would insert more than the configured "
              "maximum number of traces. max-trace-count="
           << maxTraceCount;
  }

  return mlir::success();
}

void InsertTracePass::runOnOperation() {
  auto circuit = getOperation();

  mlir::SymbolTable symbolTable(circuit);
  circt::firrtl::CircuitTargetCache targetCache;

  for (const std::string &targetString : this->targets) {
    if (mlir::failed(insertTraceForTarget(circuit, symbolTable, targetCache,
                                          targetString))) {
      signalPassFailure();
      return;
    }

    targetCache.invalidate();
  }
}

std::unique_ptr<mlir::Pass> circt::perf::createInsertTracePass() {
  auto pass = std::make_unique<InsertTracePass>();
  return pass;
}