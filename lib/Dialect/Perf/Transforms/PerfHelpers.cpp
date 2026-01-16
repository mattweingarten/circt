//===- PerfHelpers.cpp - Perf helper methods --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various helper methods shared by Perf transformation passes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Perf/PerfOps.h"

namespace circt {
namespace perf {

bool isClockLikeType(mlir::Type t) {
  return t.isa<circt::firrtl::ClockType>() || t.isa<circt::seq::ClockType>();
}

mlir::Value
findClock(circt::firrtl::FModuleOp fmod) {
  mlir::Value foundClk;

  auto ports = fmod.getPorts();
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    const auto &port = ports[i];

    if (port.isOutput())
      continue;

    mlir::Value arg = fmod.getArgument(static_cast<unsigned>(i));
    if (!arg)
      continue;

    if (!isClockLikeType(arg.getType()))
      continue;

    if (!foundClk) {
      foundClk = arg;
      continue;
    }
  }
  return foundClk;
}

mlir::Value findClockOperandOrModuleClock(mlir::Operation *op) {
  // 1) If the op has an operand that is a clock, use it.
  for (mlir::Value v : op->getOperands())
    if (isClockLikeType(v.getType()))
      return v;

  // // 2) Otherwise, fall back to the containing FIRRTL module's "clock" port
  // (if
  // // any).
  // if (auto fmod = op->getParentOfType<circt::firrtl::FModuleOp>()) {
  //   for (auto port : fmod.getPorts()) {
  //     if (port.getName() == "clock") {
  //       mlir::Value arg = fmod.getArgument(port.getArgNum());
  //       if (arg && isClockLikeType(arg.getType()))
  //         return arg;
  //     }
  //   }
  // }

  // 3) Generic MLIR module fallback: pick the first argument that is
  // clock-typed.
  if (auto m = op->getParentOfType<mlir::ModuleOp>()) {
    for (mlir::BlockArgument a : m.getBodyRegion().front().getArguments())
      if (isClockLikeType(a.getType()))
        return a;
  }

  return {};
}

// static std::string getSSAName(mlir::Value v) {
//   std::string s;
//   llvm::raw_string_ostream os(s);
//   v.print(os);
//   return s;
// }

// Hacky way to get the original name back, just use SSA results name.
std::string getSSAName(mlir::Value v,
                       ::circt::igraph::ModuleOpInterface module) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mlir::AsmState asmState(module);
  v.printAsOperand(os, asmState);
  return s.erase(0, 1); // remove leading %
}

std::string getSSAName(mlir::Value v, circt::firrtl::FModuleOp module) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mlir::AsmState asmState(module);
  v.printAsOperand(os, asmState);
  return s.erase(0, 1); // remove leading %
}

// std::string getSSAName(mlir::Value v, circt::firrtl::FModuleOp module) {
//   std::string s;
//   llvm::raw_string_ostream os(s);
//   mlir::AsmState asmState(module);
//   v.printAsOperand(os, asmState);
//   return s.erase(0, 1); // remove leading %
// }

} // namespace perf

} // namespace circt
