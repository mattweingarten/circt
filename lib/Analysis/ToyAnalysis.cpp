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
#include "circt/Support/PetriNet.h"

using namespace circt;
using namespace debug;
using namespace mlir;

namespace {

struct ValueState {
  int value; // dummy value for now;
};

// Maps analysis resuls of InputAndReturnParameters
struct ParameterMap {
  enum class Type { ReturnParameter, InputParameter };
  Type paramType;
  Operation *in;
  Operation *out;
  Operation *call;
  ValueState ac;
  ParameterMap(Type paramType, Operation *in, Operation *out, Operation *call,
               ValueState ac)
      : paramType(paramType), in(in), out(out), call(call), ac(ac) {}
};

struct CallerStateContext {
  std::vector<ValueState> params;
  CallerStateContext(Operation *op) {
    assert(
        isa<::circt::hw::InstanceOp>(op) &&
        "Panic! Trying to create a CallerStateContext for an operation that is "
        "not a hwInstanceOp!");
    auto instance = dyn_cast<::circt::hw::InstanceOp>(op);
    llvm::errs() << "Creating CallerStateContext for instance: ";
    op->dump();
  }
  CallerStateContext() {};
};

struct ToyAnalysisBuilder {
  ToyAnalysisBuilder(Operation *rootOp, mlir::AnalysisManager &am)
      : rootOp(rootOp) {}
  Operation *rootOp;
  DenseSet<Operation *> toyOps;
  DenseMap<Operation *, Operation *> instanceToModule;
  void runBuildGraph(mlir::AnalysisManager &am);
  // This can be replaced with more in-depth implementation for getting top
  // level module.
  Operation *getTopLevelModuleOp(AnalysisManager &am);
  void runOnModule(Operation *op, mlir::AnalysisManager &am,
                   CallerStateContext &pc);
  void run(mlir::AnalysisManager &am);
  bool handleInstanceOp(Operation *op, CallerStateContext &cc,
                        mlir::AnalysisManager &am);
  bool handleOutputOp(Operation *op, CallerStateContext &cc,
                      mlir::AnalysisManager &am);
  // bool handleProcessOp(Operation *op, CallerStateContext &cc,
  //                      mlir::AnalysisManager &am);
};
} // namespace

Operation *ToyAnalysisBuilder::getTopLevelModuleOp(AnalysisManager &am) {
  igraph::InstanceGraph &instanceGraph = am.getAnalysis<hw::InstanceGraph>();
  igraph::InstanceGraphNode *topLevelNode = instanceGraph.getTopLevelNode();

  // Just testing the petri net
  pnet::PetriNet petriNet;

  assert(topLevelNode && "Panic! Could not find top level node");
  assert(topLevelNode && topLevelNode->getNumUses() == 0 &&
         "Panic! Top level node has more than one use");
  size_t count = 0;
  igraph::InstanceGraphNode *target;
  for (auto it = topLevelNode->begin(); it != topLevelNode->end(); ++it) {
    target = (*it)->getTarget();
    count++;
  }
  assert(count == 1 && "Panic! Top level node has more than one child");
  auto module = target->getModule().getOperation();
  assert(module && "Could not find module code for top level node");
  assert(module->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>() &&
         "Panic! Top level module is not isolated from above");
  assert(isa<circt::hw::HWModuleOp>(module) &&
         "Panic! Top level module is not a HWModuleOp");
  auto hw_module = dyn_cast_or_null<circt::hw::HWModuleOp>(module);
  return module;
}

void ToyAnalysisBuilder::runOnModule(Operation *op, mlir::AnalysisManager &am,
                                     CallerStateContext &cc) {
  assert(isa<circt::hw::HWModuleOp>(op) &&
         "Panic! Running runOnModule on an operation that is not a HWModule!");
  hw::HWModuleOp moduleOp = dyn_cast<circt::hw::HWModuleOp>(op);
  llvm::errs() << "Running interprocedrual analsyis on Module: "
               << moduleOp.getName() << "\n";
  moduleOp.walk([&](Operation *op) {
    if (isa<::circt::hw::InstanceOp>(op)) {
      handleInstanceOp(op, cc, am);
    } else if (isa<::circt::hw::OutputOp>(op)) {
      handleOutputOp(op, cc, am);
    }
    // else if (isa<::circt::comb::CombProcessOp>(op)) {
    //   handleProcessOp(op, cc, am);
    // }
  });
}
bool ToyAnalysisBuilder::handleOutputOp(Operation *op, CallerStateContext &cc,
                                        mlir::AnalysisManager &am) {
  assert(isa<::circt::hw::OutputOp>(op) &&
         "Panic! Trying to handle an operation that is not a hwOutputOp!");
  llvm::errs() << "Handling OutputOp: ";
  op->dump();
  auto output = dyn_cast<::circt::hw::OutputOp>(op);
  return false;
}
bool ToyAnalysisBuilder::handleInstanceOp(Operation *op, CallerStateContext &cc,
                                          mlir::AnalysisManager &am) {
  assert(isa<::circt::hw::InstanceOp>(op) &&
         "Panic! Trying to handle an operation that is not a hwInstanceOp!");
  llvm::errs() << "Handling InstanceOp: ";
  op->dump();
  CallerStateContext new_cc(op);
  auto instance = dyn_cast<::circt::hw::InstanceOp>(op);
  auto moduleOp = instanceToModule[op];
  assert(moduleOp && "Panic! Could not find module for instance");
  assert(isa<circt::hw::HWModuleOp>(moduleOp) &&
         "Panic! Module for instance is not a HWModule");
  runOnModule(moduleOp, am, new_cc);
  return false;
}

void ToyAnalysisBuilder::runBuildGraph(mlir::AnalysisManager &am) {
  auto &instanceGraph = am.getAnalysis<hw::InstanceGraph>();

  llvm::errs() << "Building whole program callgraph: \n";
  rootOp->walk([&](Operation *op) {
    if (isa<::circt::hw::InstanceOp>(op)) {
      circt::hw::InstanceOp instance = dyn_cast<circt::hw::InstanceOp>(op);
      auto attrName = instance.getReferencedModuleNameAttr();
      llvm::errs() << attrName << "\n";
      circt::igraph::InstanceGraphNode *node =
          instanceGraph.lookup(attrName);
      if (node) {
        auto module = dyn_cast_or_null<circt::hw::HWModuleOp>(
            node->getModule().getOperation());
        if (module) {
          llvm::errs() << "Found module code for op ";
          llvm::errs() << " " << attrName << "\n";
          Operation *module_op = (Operation *)module;
          instanceToModule.insert({op, module_op});
        }
      }
    }
  });
}

void ToyAnalysisBuilder::run(mlir::AnalysisManager &am) {
  // Step 1: Get top level module
  Operation *top_level_module = getTopLevelModuleOp(am);
  assert(top_level_module && "Panic! Could not find top level module");
  llvm::errs() << "Top level module: \n";
  top_level_module->dump();
  // Top-level context, should be empty?
  CallerStateContext cc;
  runOnModule(top_level_module, am, cc);

  // Step 2: recursive runOnModule with Empty Parameter contex
}

// void ToyAnalysisBuilder::run(mlir::AnalysisManager &am) {

//   // We ideally want to run from the top module, I am not sure yet how to
//   // determine if something is the topmodule.
//   // --> For this we use the HWModuleGraph/HWInstanceGraph (i.e the
//   equivalent
//   // of a callgraph)
//   llvm::errs() << "RootOp: " << rootOp->getName() << "\n";
//   // rootOp->dump();

//   auto ctxt = rootOp->getContext();

//   llvm::errs() << "RootOp " << rootOp->getName() << "\n";
//   llvm::errs() << "RootOp is module?: "
//                << (isa<circt::hw::HWModuleOp>(rootOp) ? "true" : "false")
//                << "\n";
//   rootOp->walk(
//       [&](Operation *op) { // At some point we want to start from root of
//                            // whatever we mean by "making progresss!"
//         if (isa<::circt::hw::InstanceOp>(op)) {
//           circt::hw::InstanceOp instance =
//           dyn_cast<circt::hw::InstanceOp>(op); llvm::errs() << "InstanceOp:
//           "; op->dump(); toyOps.insert(op); auto it =
//           instanceToModule.find(op); if (it == instanceToModule.end()) {
//             assert(false && "Panic! Could not resolve instanceOp to
//             module");
//           }

//           for (const auto &users : op->getUsers()) {
//             llvm::errs() << "User: ";
//             users->dump();
//             llvm::errs() << "\n";
//           }

//           assert(isa<::circt::hw::HWModuleOp>(it->second) &&
//                  "Panic! Instance to module map not pointing to module");
//           circt::hw::HWModuleOp mod =
//               dyn_cast<circt::hw::HWModuleOp>(it->second);

//           // In this case we wan to enter the HW instance.

//           for (const Value &args : op->getOperands()) {
//             llvm::errs() << "Operand: ";
//             args.dump();
//             Operation *def_op = args.getDefiningOp();
//             if (def_op) {
//               llvm::errs() << "DefiningOp: ";
//               def_op->dump();
//               llvm::errs() << "\n";
//             }
//           }

//           llvm::errs() << "Module: ";
//           llvm::errs() << op->getName();
//           llvm::errs() << "\n";
//         } else if (isa<::circt::hw::OutputOp>(op)) {
//           llvm::errs() << "Output: " << op->getParentOp()->getName() << "
//           -->
//           "; op->dump(); toyOps.insert(op);
//         }
//       });
// }

ToyAnalysis::ToyAnalysis(Operation *op, mlir::AnalysisManager &am) {
  ToyAnalysisBuilder builder(op, am);
  builder.runBuildGraph(am);
  builder.run(am);
  toyOps = std::move(builder.toyOps);
}
