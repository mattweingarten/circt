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

#include "circt/Dialect/Perf/PerfHelpers.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Perf/PerfOps.h"

using namespace circt;
using namespace circt::perf;

bool FIRRTLPerfInserter::isClockLikeType(mlir::Type t) {
  return llvm::isa<circt::firrtl::ClockType>(t) ||
         llvm::isa<circt::seq::ClockType>(t);
}

bool FIRRTLPerfInserter::isResetType(mlir::Type t) {
  return llvm::isa<circt::firrtl::ResetType>(t);
}

// TODO: Find reset and find clock are a bit hacky and dumb, but works for
// now...
mlir::Value FIRRTLPerfInserter::findClock(circt::firrtl::FModuleOp fmod) {
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

mlir::Value FIRRTLPerfInserter::findReset(circt::firrtl::FModuleOp fmod) {
  mlir::Value foundReset;

  auto ports = fmod.getPorts();
  for (size_t i = 0, e = ports.size(); i != e; ++i) {
    const auto &port = ports[i];

    if (port.isOutput())
      continue;

    mlir::Value arg = fmod.getArgument(static_cast<unsigned>(i));
    if (!arg)
      continue;

    if (!isResetType(arg.getType()))
      continue;

    if (!foundReset) {
      foundReset = arg;
      continue;
    }
  }

  return foundReset;
}

mlir::Value
FIRRTLPerfInserter::findClockOperandOrModuleClock(mlir::Operation *op) {
  for (mlir::Value v : op->getOperands())
    if (isClockLikeType(v.getType()))
      return v;

  if (auto m = op->getParentOfType<mlir::ModuleOp>()) {
    for (mlir::BlockArgument a : m.getBodyRegion().front().getArguments())
      if (isClockLikeType(a.getType()))
        return a;
  }

  return {};
}

std::string
FIRRTLPerfInserter::getSSAName(mlir::Value v,
                               ::circt::igraph::ModuleOpInterface module) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mlir::AsmState asmState(module);
  v.printAsOperand(os, asmState);
  return s.erase(0, 1);
}

std::string FIRRTLPerfInserter::getSSAName(mlir::Value v,
                                           circt::firrtl::FModuleOp module) {
  std::string s;
  llvm::raw_string_ostream os(s);
  mlir::AsmState asmState(module);
  v.printAsOperand(os, asmState);
  return s.erase(0, 1);
}

mlir::FailureOr<mlir::Value> FIRRTLPerfInserter::materializeVisiblePerfSignal(
    mlir::Value v, circt::firrtl::FModuleOp fmod, mlir::OpBuilder &builder,
    std::string &signalName) {
  mlir::Operation *defOp = v.getDefiningOp();

  if (defOp && (!defOp->getParentOfType<circt::firrtl::FModuleOp>() ||
                defOp->getParentOfType<circt::firrtl::FModuleOp>() != fmod)) {
    llvm::errs() << "[PERF] Value is not defined inside requested module\n";
    return mlir::failure();
  }

  signalName = getSSAName(v, fmod);

  bool alreadyVisible = defOp && (llvm::isa<circt::firrtl::WireOp>(defOp) ||
                                  llvm::isa<circt::firrtl::RegOp>(defOp) ||
                                  llvm::isa<circt::firrtl::RegResetOp>(defOp));

  if (alreadyVisible) {
    builder.setInsertionPointAfter(defOp);
    return v;
  }

  std::string visibleName = signalName + "_perf";

  // Insert the wire after the defining op if there is one. For block args,
  // insert at the beginning of the module body.
  if (defOp)
    builder.setInsertionPointAfter(defOp);
  else
    builder.setInsertionPointToStart(fmod.getBodyBlock());

  auto wire = builder.create<circt::firrtl::WireOp>(
      v.getLoc(), v.getType(), builder.getStringAttr(visibleName));

  // The connect must come after the wire and must be in a location where both
  // the original value and the new wire are visible.
  builder.setInsertionPointAfter(wire.getOperation());

  builder.create<circt::firrtl::ConnectOp>(v.getLoc(), wire.getResult(), v);

  signalName = visibleName;
  builder.setInsertionPointAfter(wire.getOperation());

  return wire.getResult();
}

bool FIRRTLPerfInserter::insertPerfCounterOp(mlir::Value v,
                                             circt::firrtl::FModuleOp fmod,
                                             llvm::StringRef label,
                                             llvm::StringRef description) {
  if (!v) {
    llvm::errs() << "[PERF] Cannot insert counter for null value\n";
    return false;
  }

  auto uintType = llvm::dyn_cast<circt::firrtl::UIntType>(v.getType());
  if (!uintType || uintType.getWidthOrSentinel() != 1) {
    llvm::errs() << "[PERF] Cannot insert counter for non-UInt<1> value: ";
    v.print(llvm::errs());
    llvm::errs() << " : ";
    v.getType().print(llvm::errs());
    llvm::errs() << "\n";
    return false;
  }
  if (label == "")
    description = label;
  mlir::OpBuilder builder(fmod.getContext());

  std::string visibleSignalName;
  auto countCondOrFailure =
      materializeVisiblePerfSignal(v, fmod, builder, visibleSignalName);

  if (mlir::failed(countCondOrFailure))
    return false;

  mlir::Value countCond = *countCondOrFailure;

  mlir::Value clk = findClock(fmod);
  if (!clk) {
    llvm::errs() << "[PERF] No clock found in module " << fmod.getName()
                 << " for value ";
    countCond.print(llvm::errs());
    llvm::errs() << "\n";
    return false;
  }

  mlir::Value reset = findReset(fmod);
  if (!clk) {
    llvm::errs() << "[PERF] No reset found in module " << fmod.getName()
                 << " for value ";
    countCond.print(llvm::errs());
    llvm::errs() << "\n";
    return false;
  }

  builder.create<circt::perf::PerfCounterOp>(
      countCond.getLoc(),
      /*cond=*/countCond,
      /*clk=*/clk,
      /*reset=*/reset,
      /*name=*/builder.getStringAttr(label),
      /*desc=*/description.empty() ? mlir::StringAttr()
                                   : builder.getStringAttr(description));

  return true;
}

bool FIRRTLPerfInserter::insertTraceOp(mlir::Value v,
                                       circt::firrtl::FModuleOp fmod,
                                       llvm::StringRef label,
                                       llvm::StringRef description) {
  if (!v) {
    llvm::errs() << "[PERF] Cannot insert trace for null value\n";
    return false;
  }

  auto uintType = llvm::dyn_cast<circt::firrtl::UIntType>(v.getType());
  if (!uintType || uintType.getWidthOrSentinel() != 1) {
    llvm::errs() << "[PERF] Cannot insert trace for non-UInt<1> value: ";
    v.print(llvm::errs());
    llvm::errs() << " : ";
    v.getType().print(llvm::errs());
    llvm::errs() << "\n";
    return false;
  }

  if (label == "")
    description = label;

  mlir::OpBuilder builder(fmod.getContext());

  std::string visibleSignalName;
  auto traceValueOrFailure =
      materializeVisiblePerfSignal(v, fmod, builder, visibleSignalName);

  if (mlir::failed(traceValueOrFailure))
    return false;

  mlir::Value traceValue = *traceValueOrFailure;

  mlir::Value clk = findClock(fmod);
  if (!clk) {
    llvm::errs() << "[PERF] No clock found in module " << fmod.getName()
                 << " for value ";
    traceValue.print(llvm::errs());
    llvm::errs() << "\n";
    return false;
  }

  mlir::Value reset = findReset(fmod);
  if (!clk) {
    llvm::errs() << "[PERF] No reset found in module " << fmod.getName()
                 << " for value ";

    traceValue.print(llvm::errs());
    llvm::errs() << "\n";
    return false;
  }

  builder.create<circt::perf::PerfTraceOp>(
      traceValue.getLoc(),
      /*cond=*/traceValue,
      /*clk=*/clk,
      /*reset=*/reset,
      /*name=*/builder.getStringAttr(label),
      /*desc=*/description.empty() ? mlir::StringAttr()
                                   : builder.getStringAttr(description));

  return true;
}