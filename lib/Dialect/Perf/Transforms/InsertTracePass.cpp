//===- InsertTracePass.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/Perf/PerfHelpers.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "circt/Dialect/Perf/PerfPasses.h"
#include "circt/Support/InstanceGraphInterface.h"

#include "mlir/Pass/Pass.h"
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
    // registry.insert<circt::seq::SeqDialect>();
  }

private:
  void runOnOperation() override;
  void createTargets(StringRef moduleName,
                     llvm::SmallVector<llvm::StringRef, 4> &entries);
  void insertTraceOps(circt::firrtl::FModuleOp moduleOp,
                      llvm::StringRef moduleName,
                      llvm::ArrayRef<llvm::StringRef> signals);
};

} // namespace

void InsertTracePass::insertTraceOps(circt::firrtl::FModuleOp moduleOp,
                                     llvm::StringRef moduleName,
                                     llvm::ArrayRef<llvm::StringRef> signals) {
  moduleOp.walk([&](circt::firrtl::FNamableOp op) {
    llvm::StringRef signalName = op.getName();

    if (!llvm::is_contained(signals, signalName))
      return;

    LLVM_DEBUG(llvm::dbgs() << "[PERF] Inserting PerfTraceOp for signal: "
                            << signalName << "\n");

    mlir::Operation *rawOp = op.getOperation();

    if (rawOp->getNumResults() == 0) {
      llvm::errs() << "[PERF] Named op has no result: " << signalName << "\n";
      signalPassFailure();
      return;
    }

    if (!circt::perf::FIRRTLPerfInserter::insertTraceOp(rawOp->getResult(0),
                                                        moduleOp, signalName)) {
      llvm::errs() << "[PERF] Failed to insert PerfTraceOp for signal "
                   << signalName << " in module " << moduleName << "\n";
      signalPassFailure();
    }
  });
}

void InsertTracePass::createTargets(
    StringRef moduleName, llvm::SmallVector<llvm::StringRef, 4> &entries) {
  for (const std::string &t : this->targets) {
    llvm::StringRef entry(t);
    auto split = entry.split(':');

    llvm::StringRef targetModule = split.first.trim();
    llvm::StringRef signalName = split.second.trim();

    // Must be exactly module:signal
    if (targetModule.empty() || signalName.empty())
      continue;

    if (targetModule != moduleName)
      continue;

    LLVM_DEBUG(llvm::dbgs()
               << "[PERF] Matched target for module '" << moduleName
               << "': signal '" << signalName << "'\n");

    entries.push_back(signalName);
  }
}

void InsertTracePass::runOnOperation() {
  firrtl::FModuleOp module = getOperation();
  StringRef moduleName = module.getName();
  llvm::SmallVector<llvm::StringRef, 4> entries;
  createTargets(moduleName, entries);
  insertTraceOps(module, moduleName, entries);
}

std::unique_ptr<mlir::Pass> circt::perf::createInsertTracePass() {
  auto pass = std::make_unique<InsertTracePass>();
  return pass;
}