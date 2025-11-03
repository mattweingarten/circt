//===- PrintR2RGraph.cpp - Print Petri Net ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//


#include "PassDetails.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "llvm/Support/Debug.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/R2RGraph.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"


#define DEBUG_TYPE "r2r-graph"


using namespace circt;
using namespace hw;



namespace {
    struct PrintR2RGraphPass : public circt::hw::PrintR2RGraphBase<PrintR2RGraphPass> {
        PrintR2RGraphPass(raw_ostream &os) : os(os) {}
        void runOnOperation() override {
            auto am = getAnalysisManager();
            getOperation().walk([&](hw::HWModuleOp mod) {
                r2r::R2RGraph &r2r = am.getChildAnalysis<r2r::R2RGraph>(mod);
                
                LLVM_DEBUG(llvm::outs()
                << "[R2RGraph] Built for module: "
                << mod.getName() << "\n";);
                
                r2r.writeGraph(os);
            });
            markAllAnalysesPreserved();
        }        
        raw_ostream &os;
    };
} // end anonymous namespace


std::unique_ptr<mlir::Pass> circt::hw::createPrintR2RGraphPass() {
    return std::make_unique<PrintR2RGraphPass>(llvm::errs());
}