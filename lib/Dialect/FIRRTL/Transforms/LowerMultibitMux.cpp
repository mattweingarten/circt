//===- LowerMultibitMux.cpp - Lower MultibitMuxes -------------------------*-
//C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file defines the LowerMultibitMux pass.
//
//===----------------------------------------------------------------------===//

//===- LowerMultibitMux.cpp - Lower MultibitMuxes -------------------------*-
//C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file defines the LowerMultibitMux pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

using namespace circt;
using namespace circt::firrtl;

namespace {

struct LowerMultibitMuxPass
    : public LowerMultibitMuxBase<LowerMultibitMuxPass> {
  void runOnOperation() override;
};

} // namespace

void LowerMultibitMuxPass::runOnOperation() {
  auto module = getOperation();
  bool anyFailure = false;

  llvm::SmallVector<circt::firrtl::MultibitMuxOp> multibitMuxes;
  module.walk(
      [&](circt::firrtl::MultibitMuxOp op) { multibitMuxes.push_back(op); });

  for (auto op : multibitMuxes) {
    mlir::OpBuilder builder(op);
    auto loc = op.getLoc();
    mlir::Value index = op.getIndex();
    auto inputs = op.getInputs();

    if (inputs.empty()) {
      op.emitError() << "firrtl.multibit_mux has no inputs";
      anyFailure = true;
      continue;
    }

    if (inputs.size() == 1) {
      op.replaceAllUsesWith(inputs.front());
      op.erase();
      ++numLoweredMuxes;
      continue;
    }

    // FIRRTL/CIRCT multibit_mux ordering is:
    //   index == 0 selects inputs.back()
    //   index == 1 selects inputs[inputs.size()-2]
    //   ...
    // Reverse first so that level[0] corresponds to logical input 0.
    llvm::SmallVector<mlir::Value> level;
    llvm::SmallVector<mlir::Value> inputsVec(inputs.begin(), inputs.end());
    level.reserve(inputsVec.size());
    for (auto it = inputsVec.rbegin(); it != inputsVec.rend(); ++it)
      level.push_back(*it);

    auto indexType = llvm::dyn_cast<circt::firrtl::UIntType>(index.getType());
    if (!indexType) {
      op.emitError() << "expected multibit_mux index to be !firrtl.uint, got "
                     << index.getType();
      anyFailure = true;
      continue;
    }

    int64_t indexWidth = indexType.getBitWidthOrSentinel();
    if (indexWidth <= 0) {
      op.emitError() << "multibit_mux index must have known positive width";
      anyFailure = true;
      continue;
    }

    // Collect already-used "name" attributes in the enclosing module so that
    // any new nodes we create are guaranteed unique.
    llvm::StringSet<> usedNames;
    module.walk([&](mlir::Operation *innerOp) {
      if (auto nameAttr = innerOp->getAttrOfType<mlir::StringAttr>("name"))
        usedNames.insert(nameAttr.getValue());
    });

    auto makeUniqueName = [&](llvm::StringRef base) -> std::string {
      std::string candidate = base.str();
      unsigned suffix = 0;
      while (usedNames.contains(candidate))
        candidate = (base + "_" + std::to_string(++suffix)).str();
      usedNames.insert(candidate);
      return candidate;
    };

    auto materializeWithUniqueNode =
        [&](mlir::Value value, llvm::StringRef baseName) -> mlir::Value {
      std::string uniqueName = makeUniqueName(baseName);
      auto node = builder.create<circt::firrtl::NodeOp>(loc, value, uniqueName);
      return node.getResult();
    };

    bool failedThisOp = false;

    // Extract one selector bit at a time and reduce pairs with firrtl.mux.
    //
    // For each stage:
    //   next[i] = mux(bit_k, level[2*i + 1], level[2*i])
    //
    // If there is an odd leftover element, carry it to the next level.
    for (int64_t bit = 0; level.size() > 1; ++bit) {
      if (bit >= indexWidth) {
        op.emitError() << "multibit_mux has " << inputs.size()
                       << " inputs but index width " << indexWidth
                       << " is insufficient to select all inputs";
        anyFailure = true;
        failedThisOp = true;
        break;
      }

      auto bitTy = circt::firrtl::UIntType::get(builder.getContext(), 1);
      auto selBitExpr = builder.create<circt::firrtl::BitsPrimOp>(
          loc, bitTy, index, /*hi=*/bit, /*lo=*/bit);
      mlir::Value selBit = materializeWithUniqueNode(selBitExpr.getResult(),
                                                     "multibit_mux_selbit");

      llvm::SmallVector<mlir::Value> nextLevel;
      nextLevel.reserve((level.size() + 1) / 2);

      for (size_t i = 0, e = level.size(); i < e; i += 2) {
        if (i + 1 < e) {
          auto muxExpr = builder.create<circt::firrtl::MuxPrimOp>(
              loc, level[i].getType(), selBit, level[i + 1], level[i]);
          mlir::Value muxVal = materializeWithUniqueNode(muxExpr.getResult(),
                                                         "multibit_mux_tmp");
          nextLevel.push_back(muxVal);
        } else {
          // Odd count: preserve the final element into the next stage.
          nextLevel.push_back(level[i]);
        }
      }

      level = std::move(nextLevel);
    }

    if (failedThisOp)
      continue;

    op.replaceAllUsesWith(level.front());
    op.erase();
    ++numLoweredMuxes;
  }

  if (anyFailure)
    signalPassFailure();
}

namespace circt {
namespace firrtl {

std::unique_ptr<mlir::Pass> createLowerMultibitMuxPass() {
  return std::make_unique<LowerMultibitMuxPass>();
}

} // namespace firrtl
} // namespace circt