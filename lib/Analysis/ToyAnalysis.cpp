//===- ToyAnalysis.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/ToyAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/CallGraph.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace debug;
using namespace mlir;

namespace {
struct ToyAnalysisBuilder {
  ToyAnalysisBuilder(Operation *rootOp) : rootOp(rootOp) {}
  Operation *rootOp;
  DenseSet<Operation *> toyOps;

  void run();
};
} // namespace

void ToyAnalysisBuilder::run() {

  llvm::errs() << "RootOp: \n";
  rootOp->dumpPretty();

  rootOp->walk([&](Operation *op) {
    if (isa<::circt::hw::InstanceOp>(op)) {
      circt::hw::InstanceOp instance = dyn_cast<circt::hw::InstanceOp>(op);
      llvm::errs() << "Hello we found you: ";
      op->dumpPretty();
      toyOps.insert(op);
      // op->getContext();
      /// Resolve a symbol to a Module.
      // FModuleLike getModule(StringAttr name);
    } else if (isa<::circt::hw::OutputOp>(op)) {
      llvm::errs() << "Hello we found you: ";
      op->dumpPretty();
      toyOps.insert(op);
    }
    if (isa<::circt::hw::HWModuleOp>(op)) {
      llvm::errs() << "Hello we found you: ";
      circt::hw::HWModuleOp mod = dyn_cast<circt::hw::HWModuleOp>(op);
      circt::hw::ModulePortInfo iports(mod.getPortList());
      for (auto [info, arg] :
           llvm::zip(iports.getInputs(), mod.getBodyBlock()->getArguments())) {

        llvm::errs() << info.getName().str() << "\n";
      }
      // op->dumpPretty();
      // op->getName().dump();
      llvm::errs() << "\n";
      toyOps.insert(op);
    }
  });
}

ToyAnalysis::ToyAnalysis(Operation *op) {
  ToyAnalysisBuilder builder(op);
  builder.run();
  toyOps = std::move(builder.toyOps);
}
