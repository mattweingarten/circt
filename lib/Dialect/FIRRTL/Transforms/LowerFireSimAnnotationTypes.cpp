//===- LowerFireSimAnnotationTypes.cpp ------------------------------------===//
//
// Rewrites FireSim annotation target strings after FIRRTL type lowering.
// Example rewrites:
//   "~FireSim|TSIBridge>tsi.out.ready" -> "~FireSim|TSIBridge>tsi_out_ready"
//   "~FireSim|RationalClockBridge>clocks[0]" ->
//   "~FireSim|RationalClockBridge>clocks_0"
//
// This pass only rewrites annotations whose "class" starts with "firesim:" or
// "firesim." (and recursively rewrites their nested dictionaries/arrays).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Firtool/Firtool.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace circt;
using namespace circt::firrtl;

#define DEBUG_TYPE "firesim-lower"

namespace {

/// Return true if this looks like a FIRRTL target string.
static bool looksLikeFirrtlTarget(llvm::StringRef s) {
  return s.starts_with("~") && s.contains("|") && s.contains(">");
}

/// Rewrite the post-'>' portion of a FIRRTL target:
///   foo.bar[0].baz -> foo_bar_0_baz
static std::string lowerFireSimTarget(llvm::StringRef target) {
  if (!looksLikeFirrtlTarget(target))
    return target.str();

  size_t gt = target.find('>');
  if (gt == llvm::StringRef::npos)
    return target.str();

  std::string prefix = target.substr(0, gt + 1).str();
  llvm::StringRef suffix = target.substr(gt + 1);

  std::string lowered;
  lowered.reserve(suffix.size());

  for (size_t i = 0; i < suffix.size(); ++i) {
    char c = suffix[i];
    switch (c) {
    case '.':
    case '[':
    case ']':
      lowered.push_back('_');
      break;
    default:
      lowered.push_back(c);
      break;
    }
  }

  // Compress repeated underscores that can arise from "][", ".[" etc.
  std::string compressed;
  compressed.reserve(lowered.size());
  bool lastWasUnderscore = false;
  for (char c : lowered) {
    bool isUnderscore = (c == '_');
    if (!(isUnderscore && lastWasUnderscore))
      compressed.push_back(c);
    lastWasUnderscore = isUnderscore;
  }

  // Remove trailing underscore if present.
  if (!compressed.empty() && compressed.back() == '_')
    compressed.pop_back();

  return prefix + compressed;
}

static std::string rewriteTargetWithLog(llvm::StringRef target) {
  std::string oldTarget = target.str();
  std::string newTarget = lowerFireSimTarget(oldTarget);

  if (oldTarget != newTarget)
    LLVM_DEBUG(llvm::dbgs() << "[FIRESIM LOWER ANNOTATIONS] RENAME "
                            << oldTarget << " -> " << newTarget << "\n");

  return newTarget;
}

/// Return true if the dictionary attr has a "class" key starting with
/// "firesim:" or "firesim."
static bool isFireSimAnnotation(DictionaryAttr dict) {
  auto classAttr = dict.getAs<StringAttr>("class");
  if (!classAttr)
    return false;

  llvm::StringRef cls = classAttr.getValue();
  return cls.starts_with("firesim:") || cls.starts_with("firesim.");
}

/// Recursively rewrite nested attributes in a FireSim annotation.
static Attribute rewriteFireSimAttr(Attribute attr, MLIRContext *ctx);

static DictionaryAttr rewriteDictionaryAttr(DictionaryAttr dict,
                                            MLIRContext *ctx) {
  SmallVector<NamedAttribute> rewritten;
  rewritten.reserve(dict.size());

  for (auto na : dict) {
    Attribute value = na.getValue();

    // Rewrite nested attributes recursively.
    Attribute newValue = rewriteFireSimAttr(value, ctx);
    rewritten.push_back(NamedAttribute(na.getName(), newValue));
  }

  return DictionaryAttr::get(ctx, rewritten);
}

static ArrayAttr rewriteArrayAttr(ArrayAttr arr, MLIRContext *ctx) {
  SmallVector<Attribute> rewritten;
  rewritten.reserve(arr.size());

  for (Attribute elt : arr)
    rewritten.push_back(rewriteFireSimAttr(elt, ctx));

  return ArrayAttr::get(ctx, rewritten);
}

static Attribute rewriteFireSimAttr(Attribute attr, MLIRContext *ctx) {
  if (!attr)
    return attr;

  if (auto str = attr.dyn_cast<StringAttr>()) {
    auto oldStr = str.getValue();
    if (!looksLikeFirrtlTarget(oldStr))
      return attr;
    return StringAttr::get(ctx, rewriteTargetWithLog(oldStr));
  }

  if (auto dict = attr.dyn_cast<DictionaryAttr>())
    return rewriteDictionaryAttr(dict, ctx);

  if (auto arr = attr.dyn_cast<ArrayAttr>())
    return rewriteArrayAttr(arr, ctx);

  return attr;
}

/// Rewrite an annotations array, but only for entries whose top-level dict has
/// a FireSim class.
static ArrayAttr rewriteAnnotations(ArrayAttr annos, MLIRContext *ctx) {
  SmallVector<Attribute> rewritten;
  rewritten.reserve(annos.size());

  for (Attribute anno : annos) {
    if (auto dict = anno.dyn_cast<DictionaryAttr>()) {
      if (isFireSimAnnotation(dict)) {
        rewritten.push_back(rewriteFireSimAttr(dict, ctx));
        continue;
      }
    }
    rewritten.push_back(anno);
  }

  return ArrayAttr::get(ctx, rewritten);
}

struct LowerFireSimAnnotationTypesPass
    : public PassWrapper<LowerFireSimAnnotationTypesPass,
                         OperationPass<CircuitOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerFireSimAnnotationTypesPass)

  StringRef getArgument() const final {
    return "firrtl-lower-firesim-annotation-types";
  }

  StringRef getDescription() const final {
    return "Lower FireSim annotation target strings after FIRRTL type lowering";
  }

  void runOnOperation() override {
    CircuitOp circuit = getOperation();
    MLIRContext *ctx = &getContext();

    auto rewriteAnnoArrayOnOp = [&](Operation *op) {
      auto annos = op->getAttrOfType<ArrayAttr>("annotations");
      if (!annos)
        return;

      auto newAnnos = rewriteAnnotations(annos, ctx);
      if (newAnnos != annos)
        op->setAttr("annotations", newAnnos);
    };

    rewriteAnnoArrayOnOp(circuit.getOperation());

    circuit.walk([&](Operation *op) { rewriteAnnoArrayOnOp(op); });
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
circt::firrtl::createLowerFireSimAnnotationTypesPass() {
  return std::make_unique<LowerFireSimAnnotationTypesPass>();
}