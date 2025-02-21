//===- ToyAnalysis.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWPasses.h"

#include "circt/Analysis/ToyAnalysis.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/CallGraph.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace debug;
using namespace mlir;

namespace {
struct ToyAnalysisBuilder {
  ToyAnalysisBuilder(Operation *rootOp, mlir::AnalysisManager &am)
      : rootOp(rootOp) {}
  Operation *rootOp;
  DenseSet<Operation *> toyOps;
  DenseMap<Operation *, Operation *> instanceToModule;

  void run(mlir::AnalysisManager &am);
};
} // namespace

void ToyAnalysisBuilder::run(mlir::AnalysisManager &am) {
  auto &instanceGraph = am.getAnalysis<hw::InstanceGraph>();

  llvm::errs() << "RootOp: \n";
  rootOp->dumpPretty();

  auto ctxt = rootOp->getContext();

  llvm::errs() << "Finding interesting points: \n";
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

  llvm::errs() << "Building whole program callgraph: \n";
  rootOp->walk([&](Operation *op) {
    if (isa<::circt::hw::InstanceOp>(op)) {
      circt::hw::InstanceOp instance = dyn_cast<circt::hw::InstanceOp>(op);
      llvm::errs() << "Hello we found you: ";
      // instance
      auto attrName = instance.getReferencedModuleNameAttr();
      llvm::errs() << attrName << "\n";
      circt::igraph::InstanceGraphNode *node =
          instanceGraph.lookupOrNull(attrName);
      if (node) {
        auto module = dyn_cast_or_null<circt::hw::HWModuleOp>(
            node->getModule().getOperation());
        if (module) {
          llvm::errs() << "Found module code: " << attrName << "\n";
          Operation *module_op = (Operation *)module;
          instanceToModule.insert({op, module_op});
          // module_op->dumpPretty();
        }
      }
    }
  });
}

ToyAnalysis::ToyAnalysis(Operation *op, mlir::AnalysisManager &am) {
  ToyAnalysisBuilder builder(op, am);
  builder.run(am);
  toyOps = std::move(builder.toyOps);
}
