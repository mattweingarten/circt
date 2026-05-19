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
  return llvm::isa<circt::firrtl::ResetType>(t)
      || llvm::isa<circt::firrtl::AsyncResetType>(t);
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

bool FIRRTLPerfInserter::insertCounterOp(mlir::Value v,
                                         circt::firrtl::FModuleOp fmod,
                                         llvm::StringRef label,
                                         llvm::StringRef description) {
  assert(!label.empty() && "label cannot be emptry for inserting trace");
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
  if (!reset) {
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
  assert(!label.empty() && "label cannot be empty for inserting trace");
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
  if (!reset) {
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

mlir::FailureOr<mlir::FlatSymbolRefAttr>
FIRRTLPerfInserter::getOrCreateContextHierPath(
    circt::firrtl::AnnoPathValue pathValue, circt::firrtl::FModuleOp fmod,
    mlir::Location loc, llvm::StringRef label) {
  if (pathValue.isLocal())
    return mlir::FlatSymbolRefAttr();

  auto circuit = fmod->getParentOfType<circt::firrtl::CircuitOp>();
  if (!circuit) {
    llvm::errs() << "[PERF] Cannot create context hierpath without parent "
                    "firrtl.circuit\n";
    return mlir::failure();
  }

  mlir::MLIRContext *ctx = fmod.getContext();
  mlir::OpBuilder builder(ctx);

  llvm::SmallVector<mlir::Attribute> path;

  for (auto inst : pathValue.instances) {
    auto parentMod = inst->getParentOfType<circt::firrtl::FModuleOp>();
    if (!parentMod) {
      llvm::errs() << "[PERF] Instance in annotation path has no parent "
                      "firrtl.module: ";
      inst.print(llvm::errs());
      llvm::errs() << "\n";
      return mlir::failure();
    }

    circt::hw::InnerSymAttr innerSym = inst.getInnerSymAttr();

    if (!innerSym) {
      std::string symName = inst.getName().str();
      if (symName.empty())
        symName = "perf_inst";

      for (char &c : symName)
        if (!llvm::isAlnum(c) && c != '_')
          c = '_';

      innerSym =
          circt::hw::InnerSymAttr::get(mlir::StringAttr::get(ctx, symName));
      inst.setInnerSymAttr(innerSym);
    }

    path.push_back(circt::hw::InnerRefAttr::get(
        ctx, mlir::FlatSymbolRefAttr::get(parentMod.getNameAttr()),
        innerSym.getSymName()));
  }
  std::string baseName = "__perf_context";
  if (!label.empty()) {
    baseName += "_";
    baseName += label.str();
  }

  for (auto inst : pathValue.instances) {
    baseName += "_";
    baseName += inst.getName().str();
  }

  for (char &c : baseName)
    if (!llvm::isAlnum(c) && c != '_')
      c = '_';

  mlir::SymbolTable symbolTable(circuit);

  std::string uniqueName = baseName;
  unsigned suffix = 0;
  while (symbolTable.lookup(uniqueName))
    uniqueName = (baseName + "_" + llvm::Twine(suffix++)).str();

  mlir::OpBuilder circuitBuilder(circuit.getBodyBlock(),
                                 circuit.getBodyBlock()->begin());

  circuitBuilder.create<circt::hw::HierPathOp>(
      loc, builder.getStringAttr(uniqueName), builder.getArrayAttr(path));

  return mlir::FlatSymbolRefAttr::get(ctx, uniqueName);
}

bool FIRRTLPerfInserter::insertTraceOp(circt::firrtl::AnnoPathValue pathValue,
                                       llvm::StringRef label,
                                       llvm::StringRef description) {

  assert(!label.empty() && "label cannot be empty for inserting trace");
  mlir::Value v;

  if (auto opRef = llvm::dyn_cast<circt::firrtl::OpAnnoTarget>(pathValue.ref)) {
    mlir::Operation *targetOp = opRef.getOp();

    if (targetOp->getNumResults() != 1) {
      llvm::errs() << "[PERF] Cannot insert trace for target op with "
                   << targetOp->getNumResults() << " results: ";
      targetOp->print(llvm::errs());
      llvm::errs() << "\n";
      return false;
    }

    v = targetOp->getResult(0);
  } else {
    llvm::errs() << "[PERF] Cannot insert trace for unsupported annotation "
                    "target kind\n";
    return false;
  }

  if (!v) {
    llvm::errs() << "[PERF] Cannot insert trace for null annotation value\n";
    return false;
  }

  auto fmod =
      v.getDefiningOp()
          ? v.getDefiningOp()->getParentOfType<circt::firrtl::FModuleOp>()
          : v.getParentBlock()
                ->getParentOp()
                ->getParentOfType<circt::firrtl::FModuleOp>();

  if (!fmod) {
    llvm::errs() << "[PERF] Cannot find parent FIRRTL module for value ";
    llvm::errs() << "\n";
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
  if (!reset) {
    llvm::errs() << "[PERF] No reset found in module " << fmod.getName()
                 << " for value ";
    traceValue.print(llvm::errs());
    llvm::errs() << "\n";
    return false;
  }

  auto contextOrFailure =
      getOrCreateContextHierPath(pathValue, fmod, traceValue.getLoc(), label);

  if (mlir::failed(contextOrFailure))
    return false;

  mlir::FlatSymbolRefAttr contextAttr = *contextOrFailure;

  mlir::OpBuilder opBuilder(traceValue.getContext());
  if (auto defOp = traceValue.getDefiningOp())
    opBuilder.setInsertionPointAfter(defOp);
  else
    opBuilder.setInsertionPointToStart(fmod.getBodyBlock());

  opBuilder.create<circt::perf::PerfTraceOp>(
      traceValue.getLoc(),
      /*cond=*/traceValue,
      /*clk=*/clk,
      /*reset=*/reset,
      /*name=*/builder.getStringAttr(label),
      /*desc=*/description.empty() ? mlir::StringAttr()
                                   : builder.getStringAttr(description),
      /*context=*/contextAttr);

  return true;
}
bool FIRRTLPerfInserter::insertCounterOp(circt::firrtl::AnnoPathValue pathValue,
                                       llvm::StringRef label,
                                       llvm::StringRef description) {

  assert(!label.empty() && "label cannot be empty for inserting counter");
  mlir::Value v;

  if (auto opRef = llvm::dyn_cast<circt::firrtl::OpAnnoTarget>(pathValue.ref)) {
    mlir::Operation *targetOp = opRef.getOp();

    if (targetOp->getNumResults() != 1) {
      llvm::errs() << "[PERF] Cannot insert counter for target op with "
                   << targetOp->getNumResults() << " results: ";
      targetOp->print(llvm::errs());
      llvm::errs() << "\n";
      return false;
    }

    v = targetOp->getResult(0);
  } else {
    llvm::errs() << "[PERF] Cannot insert counter for unsupported annotation "
                    "target kind\n";
    return false;
  }

  if (!v) {
    llvm::errs() << "[PERF] Cannot insert counter for null annotation value\n";
    return false;
  }

  auto fmod =
      v.getDefiningOp()
          ? v.getDefiningOp()->getParentOfType<circt::firrtl::FModuleOp>()
          : v.getParentBlock()
                ->getParentOp()
                ->getParentOfType<circt::firrtl::FModuleOp>();

  if (!fmod) {
    llvm::errs() << "[PERF] Cannot find parent FIRRTL module for value ";
    llvm::errs() << "\n";
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
  if (!reset) {
    llvm::errs() << "[PERF] No reset found in module " << fmod.getName()
                 << " for value ";
    traceValue.print(llvm::errs());
    llvm::errs() << "\n";
    return false;
  }

  auto contextOrFailure =
      getOrCreateContextHierPath(pathValue, fmod, traceValue.getLoc(), label);

  if (mlir::failed(contextOrFailure))
    return false;

  mlir::FlatSymbolRefAttr contextAttr = *contextOrFailure;

  mlir::OpBuilder opBuilder(traceValue.getContext());
  if (auto defOp = traceValue.getDefiningOp())
    opBuilder.setInsertionPointAfter(defOp);
  else
    opBuilder.setInsertionPointToStart(fmod.getBodyBlock());

  opBuilder.create<circt::perf::PerfCounterOp>(
      traceValue.getLoc(),
      /*cond=*/traceValue,
      /*clk=*/clk,
      /*reset=*/reset,
      /*name=*/builder.getStringAttr(label),
      /*desc=*/description.empty() ? mlir::StringAttr()
                                   : builder.getStringAttr(description),
      /*context=*/contextAttr);

  return true;
}
