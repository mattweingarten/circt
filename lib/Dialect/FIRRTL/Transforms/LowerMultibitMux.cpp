#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

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

  // Cheap unique id in case we decide to materialize any debug names later.
  unsigned uniqueId = 0;
  auto makeGeneratedName = [&](llvm::StringRef base) -> std::string {
    return (base + "_MB_LOWER_" + std::to_string(uniqueId++)).str();
  };

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
    level.reserve(inputs.size());
    for (auto it = inputs.begin(); it != inputs.end(); ++it)
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

    // Match FIRRTLLowering behavior:
    // resize the selector to the width required by the number of inputs.
    auto requiredIndexWidth =
        llvm::Log2_64_Ceil(static_cast<uint64_t>(inputs.size()));
    auto resizedIndexTy =
        circt::firrtl::UIntType::get(builder.getContext(), requiredIndexWidth);

    mlir::Value resizedIndex = index;
    if (indexWidth < requiredIndexWidth) {
      resizedIndex = builder
                         .create<circt::firrtl::PadPrimOp>(
                             loc, resizedIndexTy, index, requiredIndexWidth)
                         .getResult();
    } else if (indexWidth > requiredIndexWidth) {
      resizedIndex =
          builder
              .create<circt::firrtl::BitsPrimOp>(loc, resizedIndexTy, index,
                                                 /*hi=*/requiredIndexWidth - 1,
                                                 /*lo=*/0)
              .getResult();
    }

    // Extract one selector bit at a time and reduce pairs with firrtl.mux.
    //
    // For each stage:
    //   next[i] = mux(bit_k, level[2*i + 1], level[2*i])
    //
    // If there is an odd leftover element, carry it to the next level.
    for (int64_t bit = 0; level.size() > 1; ++bit) {
      auto bitTy = circt::firrtl::UIntType::get(builder.getContext(), 1);
      mlir::Value selBit = builder
                               .create<circt::firrtl::BitsPrimOp>(
                                   loc, bitTy, resizedIndex, /*hi=*/bit,
                                   /*lo=*/bit)
                               .getResult();

      llvm::SmallVector<mlir::Value> nextLevel;
      nextLevel.reserve((level.size() + 1) / 2);

      for (size_t i = 0, e = level.size(); i < e; i += 2) {
        if (i + 1 < e) {
          mlir::Value muxVal =
              builder
                  .create<circt::firrtl::MuxPrimOp>(
                      loc, level[i].getType(), selBit, level[i + 1], level[i])
                  .getResult();
          nextLevel.push_back(muxVal);
        } else {
          nextLevel.push_back(level[i]);
        }
      }

      level = std::move(nextLevel);
    }

    op.replaceAllUsesWith(level.front());
    op.erase();
    ++numLoweredMuxes;
  }
}

namespace circt {
namespace firrtl {

std::unique_ptr<mlir::Pass> createLowerMultibitMuxPass() {
  return std::make_unique<LowerMultibitMuxPass>();
}

} // namespace firrtl
} // namespace circt