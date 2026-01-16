//===- InsertCounterPass.cpp ----------------------------------------------===//
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

#define DEBUG_TYPE "perf-insert-counter"

namespace circt {
namespace perf {
#define GEN_PASS_DEF_INSERTCOUNTER
#include "circt/Dialect/Perf/PerfPasses.h.inc"
} // namespace perf
} // namespace circt

using namespace circt;
using namespace circt::perf;

namespace {

struct InsertCounterPass
    : public circt::perf::impl::InsertCounterBase<InsertCounterPass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<circt::perf::PerfDialect>();
    registry.insert<circt::firrtl::FIRRTLDialect>();
    // registry.insert<circt::seq::SeqDialect>();
  }

private:
  void runOnOperation() override;
  void createTargets(StringRef moduleName,
                     llvm::SmallVector<llvm::StringRef, 4> &entries);
  void insertCounters(circt::firrtl::FModuleOp moduleOp,
                      llvm::StringRef moduleName,
                      llvm::ArrayRef<llvm::StringRef> signals);
};

} // namespace

void InsertCounterPass::insertCounters(
    circt::firrtl::FModuleOp moduleOp, llvm::StringRef moduleName,
    llvm::ArrayRef<llvm::StringRef> signals) {

  moduleOp.walk([&](circt::firrtl::NodeOp op) {
    llvm::StringRef nodeName = op.getName();

    if (!llvm::is_contained(signals, nodeName))
      return;

    LLVM_DEBUG(llvm::dbgs() << "[PERF] Inserting PerfCounterOp for signal: "
                            << nodeName << "\n");

    mlir::Value clk = findClock(moduleOp);
    if (!clk) {
      llvm::errs() << "[PERF] No clock found for " << op << "\n";
      signalPassFailure();
      return;
    }

    mlir::OpBuilder b(op.getContext());
    b.setInsertionPointAfter(op.getOperation());

    b.create<circt::perf::PerfCounterOp>(
        op.getLoc(),
        /*cond=*/op.getResult(),
        /*clk=*/clk,
        /*reset=*/mlir::Value(), // optional reset (single Value)
        /*name=*/b.getStringAttr(nodeName),
        /*desc=*/mlir::StringAttr()); // or b.getStringAttr("")
  });
}

void InsertCounterPass::createTargets(
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
void InsertCounterPass::runOnOperation() {
  firrtl::FModuleOp module = getOperation();
  StringRef moduleName = module.getName();
  llvm::SmallVector<llvm::StringRef, 4> entries;
  createTargets(moduleName, entries);
  insertCounters(module, moduleName, entries);
}

std::unique_ptr<mlir::Pass> circt::perf::createInsertCounterPass() {
  auto pass = std::make_unique<InsertCounterPass>();
  return pass;
}