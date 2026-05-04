#include "PassDetails.h"

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

using namespace circt;
using namespace circt::firrtl;

namespace {

static bool isExpressionLike(mlir::Operation *op) {
  return llvm::isa<
      MuxPrimOp,
      EQPrimOp,
      NEQPrimOp,
      LTPrimOp,
      LEQPrimOp,
      GTPrimOp,
      GEQPrimOp,
      AndPrimOp,
      OrPrimOp,
      XorPrimOp,
      NotPrimOp,
      AddPrimOp,
      SubPrimOp,
      MulPrimOp,
      DivPrimOp,
      RemPrimOp,
      PadPrimOp,
      BitsPrimOp,
      HeadPrimOp,
      TailPrimOp,
      ShlPrimOp,
      ShrPrimOp,
      CatPrimOp>(op);
}

static bool hasPassiveFIRRTLResult(mlir::Operation *op) {
  if (op->getNumResults() != 1)
    return false;

  auto type = llvm::dyn_cast<FIRRTLBaseType>(op->getResult(0).getType());
  if (!type)
    return false;

  return type.isPassive();
}

static unsigned getNumUses(mlir::Value value) {
  unsigned count = 0;
  for (auto &use : value.getUses())
    ++count;
  return count;
}

struct MaterializeExpressionsPass
    : public MaterializeExpressionsBase<MaterializeExpressionsPass> {
  void runOnOperation() override;
};

} // namespace

void MaterializeExpressionsPass::runOnOperation() {
  auto module = getOperation();

  llvm::SmallVector<mlir::Operation *> candidates;

  // Maps each expression op to the length of the expression chain ending at it.
  //
  // Example:
  //   %0 = firrtl.bits %x        chain length 0
  //   %1 = firrtl.pad %0         chain length 1
  //   %2 = firrtl.mux ..., %1    chain length 2
  //
  // If an operand's defining op is not expression-like, that operand contributes
  // chain length 0.
  llvm::DenseMap<mlir::Operation *, unsigned> chainLength;

  module.walk([&](mlir::Operation *op) {
    if (!isExpressionLike(op))
      return;

    if (!hasPassiveFIRRTLResult(op))
      return;

    unsigned maxOperandChain = 0;

    for (mlir::Value operand : op->getOperands()) {
      mlir::Operation *defOp = operand.getDefiningOp();

      if (!defOp)
        continue;

      if (!isExpressionLike(defOp))
        continue;

      auto it = chainLength.find(defOp);

      // If the defining op has not been seen yet, treat it as chain length 0.
      // In valid SSA order inside a block it usually should have been seen.
      unsigned operandChain = 0;
      if (it != chainLength.end())
        operandChain = it->second;

      maxOperandChain = std::max(maxOperandChain, operandChain + 1);
    }

    chainLength[op] = maxOperandChain;

    mlir::Value result = op->getResult(0);

    bool shouldMaterialize = false;

    // Original heuristic: materialize often reused expressions.
    if (getNumUses(result) > numUsers)
      shouldMaterialize = true;

    // New heuristic: materialize long expression chains.
    if (maxOperandChain > maxChainLength)
      shouldMaterialize = true;

    if (!shouldMaterialize)
      return;

    candidates.push_back(op);

    // Important: conceptually, after this op is materialized as a node, users
    // should see a fresh named value, not a continuation of the old expression
    // chain. Reset the chain length at this op for later users.
    chainLength[op] = 0;
  });

  unsigned id = 0;

  for (auto *op : candidates) {
    if (op->getNumResults() != 1)
      continue;

    mlir::Value oldResult = op->getResult(0);

    if (oldResult.use_empty())
      continue;

    mlir::OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    std::string nodeName = ("GEN_EXPR_" + llvm::Twine(id++)).str();

    auto node = builder.create<NodeOp>(
        op->getLoc(),
        oldResult,
        nodeName,
        NameKindEnum::InterestingName);

    oldResult.replaceAllUsesExcept(node.getResult(), node);

    ++numNodesAdded;
  }
}

namespace circt {
namespace firrtl {

std::unique_ptr<mlir::Pass> createMaterializeExpressionsPass() {
  return std::make_unique<MaterializeExpressionsPass>();
}

} // namespace firrtl
} // namespace circt