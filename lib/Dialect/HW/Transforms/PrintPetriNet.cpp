//===- PrintPetriNet.cpp - Print Petri Net ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Print Petri Net of Module.
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/PetriNet.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

#include "circt/Support/PetriNet.h"

#include <string>

// namespace circt {
// namespace hw {
// #define GEN_PASS_DEF_PRINTPETRINET
// #include "circt/Dialect/HW/Passes.h.inc"
// } // namespace hw
// } // namespace circt

using namespace circt;
using namespace hw;

namespace {
struct PrintPetriNetPass : public circt::hw::PrintPetriNetBase<PrintPetriNetPass> {
  PrintPetriNetPass(std::string moduleName, std::string progressSignal,
                    int levels) {
    moduleName = moduleName;
    progressSignal = progressSignal;
  }
  void runOnOperation() override;

private:
};
} // namespace

void PrintPetriNetPass::runOnOperation() {
  assert(progressSignal != "" &&
         "progress signal must be specified for PrintPetriNetPass");

  // llvm::errs() << "Modulename: " << moduleName << "\n";
  // llvm::errs() << "ProgressSignal: " << progressSignal << "\n";

  // Step 1: initialize graph

  getOperation().walk([&](hw::HWModuleOp module) {
    // Step 2: create graph for each module
    // llvm::errs() << "module: " << module.getName() << "\n";
    if (module.getName() == moduleName) {
      // llvm::errs() << "PrintPetriNetPass running on module: " << moduleName
      //              << "\n";
      auto petriNet = pnet::PetriNet::buildGraphFromModule(module, progressSignal);
      petriNet->writeGraph(llvm::errs(), levels);
      // Todo: Support WriteGraph Interface
      // llvm::WriteGraph(llvm::errs(), buildGraph(), /*ShortNames=*/false);
    }
  });
}

std::unique_ptr<mlir::Pass>
circt::hw::createPrintPetriNetPass(std::string moduleName,
                                   std::string progressSignal, int levels) {
  return std::make_unique<PrintPetriNetPass>(moduleName, progressSignal,
                                             levels);
}
