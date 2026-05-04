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

    // The maximum number of selector values representable by the index.
    //
    // Example:
    //   width 1 can select values 0..1, so only the first 2 logical inputs are
    //   reachable.
    //   width 3 can select values 0..7, so the first 8 logical inputs are
    //   reachable.
    //
    // If there are more inputs than representable selector values, do not
    // error. Those trailing logical inputs are unreachable by construction, so
    // do not materialize comparisons for them.
    uint64_t maxSelectableInputs = inputs.size();

    if (indexWidth < 64)
      maxSelectableInputs =
          std::min<uint64_t>(inputs.size(), uint64_t(1) << indexWidth);

    // Mux definition is:
    //
    //   firrtl.multibit_mux %index, %v_(n-1), ..., %v_1, %v_0
    //
    // where:
    //
    //   index == 0 selects %v_0
    //   index == 1 selects %v_1
    //   ...
    //
    // The IR operand list is physically reversed relative to selector value.
    // Reverse it here so:
    //
    //   logicalInputs[0] == value selected by index == 0
    //   logicalInputs[1] == value selected by index == 1
    //   ...
    llvm::SmallVector<mlir::Value> logicalInputs;
    logicalInputs.reserve(inputs.size());

    for (size_t i = inputs.size(); i > 0; --i)
      logicalInputs.push_back(inputs[i - 1]);

    mlir::Value result = logicalInputs[0];

    auto eqType = circt::firrtl::UIntType::get(builder.getContext(), 1);

    for (uint64_t selectorValue = 1; selectorValue < maxSelectableInputs;
         ++selectorValue) {
      // Build constant for each reachable index value.
      auto constAttrType = mlir::IntegerType::get(
          builder.getContext(), indexWidth, mlir::IntegerType::Unsigned);

      auto constAttr = mlir::IntegerAttr::get(
          constAttrType, llvm::APInt(indexWidth, selectorValue));

      mlir::Value constIndex = builder
                                   .create<circt::firrtl::ConstantOp>(
                                       loc, index.getType(), constAttr)
                                   .getResult();

      // Equals check for each reachable index.
      mlir::Value eq =
          builder
              .create<circt::firrtl::EQPrimOp>(loc, eqType, index, constIndex)
              .getResult();
      // Build mux chain.
      auto mux = builder.create<circt::firrtl::MuxPrimOp>(
          loc, result.getType(), eq, logicalInputs[selectorValue], result);

      // We must build node at each point, otherwise we blow up expression size.
      std::string nodeName = ("GEN_MB_" + llvm::Twine(selectorValue)).str();

      auto node =
          builder.create<circt::firrtl::NodeOp>(loc, mux.getResult(), nodeName);

      result = node.getResult();
    }

    op.replaceAllUsesWith(result);
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