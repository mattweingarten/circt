#include "circt/Dialect/FIRRTL/CounterInserter.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/Perf/PerfOps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "counter-inserter"

using namespace circt;
using namespace circt::firrtl;

unsigned CounterInserter::numCountersInserted = 0;
unsigned CounterInserter::globalPortId = 0;


namespace {

static mlir::Region *getOwningRegion(mlir::Value v) {
  if (!v)
    return nullptr;

  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(v)) {
    mlir::Block *owner = blockArg.getOwner();
    return owner ? owner->getParent() : nullptr;
  }

  if (mlir::Operation *defOp = v.getDefiningOp())
    return defOp->getParentRegion();

  return nullptr;
}

static std::string sanitizePortName(StringRef debugName) {
  // Find last '>'
  size_t pos = debugName.rfind('>');
  StringRef suffix =
      (pos == StringRef::npos) ? debugName : debugName.drop_front(pos + 1);

  // Ensure it’s not empty
  if (suffix.empty())
    suffix = "signal";

  return ("io_" + suffix).str();
}

static bool checkValueInRegion(mlir::Value v, mlir::Region *targetRegion,
                               llvm::StringRef what) {
  if (!v) {
    llvm::errs() << "[ERROR] null value for " << what << "\n";
    return false;
  }

  if (!targetRegion) {
    llvm::errs() << "[ERROR] null target region for " << what << "\n";
    return false;
  }

  mlir::Region *valueRegion = getOwningRegion(v);
  if (!valueRegion) {
    llvm::errs() << "[ERROR] value for " << what << " has no owning region\n";
    return false;
  }

  if (valueRegion != targetRegion) {
    llvm::errs() << "[ERROR] value for " << what
                 << " is defined in a different region\n";
    llvm::errs() << "  value: " << v << "\n";
    return false;
  }

  return true;
}

static bool checkInsertionPointInRegion(mlir::OpBuilder &builder,
                                        mlir::Region *targetRegion) {
  mlir::Block *block = builder.getInsertionBlock();
  if (!block) {
    llvm::errs() << "[ERROR] builder has no insertion block\n";
    return false;
  }

  mlir::Region *insertRegion = block->getParent();
  if (!insertRegion) {
    llvm::errs() << "[ERROR] insertion block has no parent region\n";
    return false;
  }

  if (insertRegion != targetRegion) {
    llvm::errs() << "[ERROR] insertion point is in a different region\n";
    return false;
  }

  return true;
}

static bool checkBinaryValuesInRegion(mlir::Value lhs, mlir::Value rhs,
                                      mlir::Region *targetRegion,
                                      llvm::StringRef opName) {
  bool ok = true;
  ok &= checkValueInRegion(lhs, targetRegion, opName);
  ok &= checkValueInRegion(rhs, targetRegion, opName);
  return ok;
}

enum class ExprSortKind { Bool, BV };

struct LoweredExpr {
  enum Kind {
    Leaf,
    Constant,
    Not,
    And,
    Or,
    Xor,
    Eq,
    Distinct,
    Ite,

    BVNot,
    BVAnd,
    BVOr,
    BVXor,
    BVAdd,
    BVSub,
    BVMul,
    BVNeg,
    Concat,
    Extract,

    ULess,
    ULessEq,
    UGreater,
    UGreaterEq,
    SLess,
    SLessEq,
    SGreater,
    SGreaterEq,

    Shl,
    LShr,
    AShr,
  } kind;

  ExprSortKind sortKind = ExprSortKind::BV;
  unsigned bitWidth = 0;

  mlir::Value leafValue;
  llvm::APInt constValue = llvm::APInt(1, 0);

  unsigned extractHi = 0;
  unsigned extractLo = 0;

  llvm::SmallVector<std::shared_ptr<LoweredExpr>, 3> children;

  static std::shared_ptr<LoweredExpr> makeLeaf(mlir::Value v) {
    auto out = std::make_shared<LoweredExpr>();
    out->kind = Leaf;
    out->leafValue = v;

    if (!v) {
      llvm::errs() << "[ERROR] Expected non-null value for leaf, got null.\n";
      return out;
    }

    auto firTy = llvm::cast<FIRRTLType>(v.getType());
    if (!firTy) {
      llvm::errs() << "[ERROR] Expected FIRRTL type for leaf value, got: "
                   << v.getType() << "\n";
      return out;
    }

    if (llvm::isa<UIntType>(firTy)) {
      auto uintTy = llvm::cast<UIntType>(firTy);
      out->sortKind = ExprSortKind::BV;
      out->bitWidth = std::max(1u, (unsigned)uintTy.getWidthOrSentinel());

    } else if (llvm::isa<SIntType>(firTy)) {
      auto sintTy = llvm::cast<SIntType>(firTy);
      out->sortKind = ExprSortKind::BV;
      out->bitWidth = std::max(1u, (unsigned)sintTy.getWidthOrSentinel());

    } else if (llvm::isa<ResetType, AsyncResetType>(firTy)) {
      out->sortKind = ExprSortKind::Bool;
      out->bitWidth = 1;

    } else if (llvm::isa<ClockType>(firTy)) {
      // Clocks should not generally appear in arithmetic/logic expressions.
      out->sortKind = ExprSortKind::BV;
      out->bitWidth = 1;

    } else {
      out->sortKind = ExprSortKind::BV;
      out->bitWidth = 1;
    }

    return out;
  }
  static std::shared_ptr<LoweredExpr> makeConstBool(bool bit) {
    auto out = std::make_shared<LoweredExpr>();
    out->kind = Constant;
    out->sortKind = ExprSortKind::Bool;
    out->bitWidth = 1;
    out->constValue = llvm::APInt(1, bit ? 1 : 0);
    return out;
  }

  static std::shared_ptr<LoweredExpr> makeConstBV(unsigned width,
                                                  const llvm::APInt &value) {
    auto out = std::make_shared<LoweredExpr>();
    out->kind = Constant;
    out->sortKind = ExprSortKind::BV;
    out->bitWidth = std::max(1u, width);
    out->constValue = value.zextOrTrunc(out->bitWidth);
    return out;
  }

  static std::shared_ptr<LoweredExpr>
  makeNode(Kind kind, ExprSortKind sortKind, unsigned bitWidth,
           llvm::SmallVector<std::shared_ptr<LoweredExpr>, 3> kids) {
    auto out = std::make_shared<LoweredExpr>();
    out->kind = kind;
    out->sortKind = sortKind;
    out->bitWidth = bitWidth;
    out->children = std::move(kids);
    return out;
  }

  static std::shared_ptr<LoweredExpr>
  makeExtract(std::shared_ptr<LoweredExpr> in, unsigned hi, unsigned lo) {
    auto out = std::make_shared<LoweredExpr>();
    out->kind = Extract;
    out->sortKind = ExprSortKind::BV;
    out->bitWidth = hi - lo + 1;
    out->extractHi = hi;
    out->extractLo = lo;
    out->children.push_back(std::move(in));
    return out;
  }
};

static bool isClockType(mlir::Type ty) { return ty.isa<ClockType>(); }

static bool isResetLikeType(mlir::Type ty) {
  if (auto firTy = ty.dyn_cast<FIRRTLType>()) {
    if (firTy.isa<ResetType, AsyncResetType>())
      return true;
    if (auto uintTy = firTy.dyn_cast<UIntType>())
      return uintTy.getWidthOrSentinel() == 1;
  }
  return false;
}

struct NamedModulePort {
  mlir::Value value;
  std::string portName;
};

struct ModuleClockReset {
  NamedModulePort clock;
  NamedModulePort reset;
};

static ModuleClockReset findModuleClockAndReset(FModuleOp module) {
  ModuleClockReset result;

  auto *body = module.getBodyBlock();
  if (!body)
    return result;

  for (unsigned i = 0, e = module.getNumPorts(); i != e; ++i) {
    auto arg = body->getArgument(i);
    auto name = module.getPortName(i).str();

    if (!result.clock.value && name == "clock" && isClockType(arg.getType()))
      result.clock = {arg, name};

    if (!result.reset.value && name == "reset" &&
        isResetLikeType(arg.getType()))
      result.reset = {arg, name};
  }

  for (unsigned i = 0, e = module.getNumPorts(); i != e; ++i) {
    auto arg = body->getArgument(i);
    auto name = module.getPortName(i).str();

    if (!result.clock.value && isClockType(arg.getType()))
      result.clock = {arg, name};

    if (!result.reset.value && isResetLikeType(arg.getType()))
      result.reset = {arg, name};
  }

  return result;
}

static bool isZ3BoolConst(const z3::expr &e) {
  if (!e.is_app())
    return false;
  auto dk = e.decl().decl_kind();
  return dk == Z3_OP_TRUE || dk == Z3_OP_FALSE;
}

static bool isZ3AtomicLeaf(const z3::expr &e) {
  return e.is_const() && !e.is_numeral() && !isZ3BoolConst(e) &&
         e.num_args() == 0;
}

static bool getZ3BVNumeralAPInt(const z3::expr &e, llvm::APInt &out) {
  if (!e.is_bv() || !e.is_numeral())
    return false;

  unsigned width = std::max(1u, e.get_sort().bv_size());

  Z3_string s = Z3_get_numeral_string(e.ctx(), e);
  if (!s)
    return false;

  llvm::StringRef str(s);
  if (str.empty())
    return false;

  if (str.consume_front("#b")) {
    llvm::APInt value;
    if (str.getAsInteger(2, value))
      return false;
    out = value.zextOrTrunc(width);
    return true;
  }

  if (str.consume_front("#x")) {
    llvm::APInt value;
    if (str.getAsInteger(16, value))
      return false;
    out = value.zextOrTrunc(width);
    return true;
  }

  // Fallback: decimal numeral string.
  llvm::APInt value;
  if (str.getAsInteger(10, value))
    return false;

  out = value.zextOrTrunc(width);
  return true;
}

static void collectReferencedLeaves(const z3::expr &e,
                                    llvm::SmallVectorImpl<z3::expr> &out,
                                    llvm::SmallPtrSetImpl<Z3_ast> &seen) {
  Z3_ast ast = e;
  if (!seen.insert(ast).second)
    return;

  if (isZ3AtomicLeaf(e)) {
    out.push_back(e);
    return;
  }

  for (unsigned i = 0; i < e.num_args(); ++i)
    collectReferencedLeaves(e.arg(i), out, seen);
}

static mlir::Value getValueFromAnnoPathValue(const AnnoPathValue &apv) {
  if (auto opRef = llvm::dyn_cast<OpAnnoTarget>(apv.ref)) {
    Operation *op = opRef.getOp();
    if (!op)
      return {};

    if (op->getNumResults() == 1)
      return op->getResult(0);

    return {};
  }

  if (auto portRef = llvm::dyn_cast<PortAnnoTarget>(apv.ref)) {
    auto module = llvm::dyn_cast<FModuleOp>(portRef.getModule().getOperation());
    if (!module)
      return {};

    auto *body = module.getBodyBlock();
    if (!body)
      return {};

    unsigned portNo = portRef.getPortNo();
    if (portNo >= body->getNumArguments())
      return {};

    return body->getArgument(portNo);
  }

  return {};
}

static BlockArgument addInputPort(FModuleLike module, StringRef name,
                                  FIRRTLType type) {
  auto fmodule = dyn_cast<FModuleOp>(module.getOperation());
  if (!fmodule)
    return {};

  mlir::OpBuilder b(fmodule);
  PortInfo info{b.getStringAttr(name), type, Direction::In, {}, {}, {}};
  unsigned newPortIdx = module.getNumPorts();
  module.insertPorts({{newPortIdx, info}});

  return fmodule.getBodyBlock()->getArgument(newPortIdx);
}
static bool
findPathToAncestor(circt::firrtl::InstanceGraph &instanceGraph,
                   circt::firrtl::FModuleLike currentModule,
                   circt::firrtl::FModuleLike targetModule,
                   llvm::SmallVectorImpl<circt::firrtl::InstanceOp> &path,
                   llvm::SmallPtrSetImpl<mlir::Operation *> &visited) {

  auto *currentOp = currentModule.getOperation();
  if (!visited.insert(currentOp).second)
    return false;

  if (currentModule == targetModule)
    return true;

  auto *node = instanceGraph.lookup(currentModule);
  if (!node)
    return false;

  for (auto *use : node->uses()) {
    auto instance = use->getInstance<circt::firrtl::InstanceOp>();
    if (!instance)
      continue;

    auto parentModule =
        mlir::dyn_cast<circt::firrtl::FModuleLike>(instance->getParentOp());
    if (!parentModule)
      continue;

    path.push_back(instance);
    if (findPathToAncestor(instanceGraph, parentModule, targetModule, path,
                           visited))
      return true;
    path.pop_back();
  }

  return false;
}

static ExprSortKind getZ3ExprSortKind(const z3::expr &e) {
  if (e.is_bool())
    return ExprSortKind::Bool;
  if (e.is_bv())
    return ExprSortKind::BV;
  return ExprSortKind::BV;
}

static unsigned getZ3ExprBitWidth(const z3::expr &e) {
  if (e.is_bool())
    return 1;
  if (e.is_bv())
    return std::max(1u, e.get_sort().bv_size());
  return 1;
}

static std::shared_ptr<LoweredExpr>
planExprIntoModule(const z3::expr &e,
                   const llvm::DenseMap<Z3_ast, mlir::Value> &leafMap) {
  if (auto it = leafMap.find((Z3_ast)e); it != leafMap.end())
    return LoweredExpr::makeLeaf(it->second);

  if (isZ3AtomicLeaf(e)) {
    llvm::errs() << "[WARN] Atomic Z3 leaf not found in leafMap: "
                 << e.to_string() << "\n";
    return nullptr;
  }

  if (isZ3BoolConst(e)) {
    bool bit = e.decl().decl_kind() == Z3_OP_TRUE;
    return LoweredExpr::makeConstBool(bit);
  }

  if (e.is_numeral()) {
    if (e.is_bool()) {
      bool bit = Z3_get_bool_value(e.ctx(), e) == Z3_L_TRUE;
      return LoweredExpr::makeConstBool(bit);
    }

    if (e.is_bv()) {
      llvm::APInt value(1, 0);
      if (!getZ3BVNumeralAPInt(e, value)) {
        llvm::errs() << "[WARN] Unsupported BV numeral in counter expr: "
                     << e.to_string() << "\n";
        return nullptr;
      }
      return LoweredExpr::makeConstBV(value.getBitWidth(), value);
    }

    llvm::errs() << "[WARN] Unsupported numeral sort in counter expr: "
                 << e.to_string() << "\n";
    return nullptr;
  }

  auto planChild = [&](unsigned i) {
    return planExprIntoModule(e.arg(i), leafMap);
  };

  auto mkNary = [&](LoweredExpr::Kind kind) -> std::shared_ptr<LoweredExpr> {
    llvm::SmallVector<std::shared_ptr<LoweredExpr>, 3> kids;
    kids.reserve(e.num_args());
    for (unsigned i = 0; i < e.num_args(); ++i) {
      auto child = planChild(i);
      if (!child)
        return nullptr;
      kids.push_back(child);
    }
    return LoweredExpr::makeNode(kind, getZ3ExprSortKind(e),
                                 getZ3ExprBitWidth(e), std::move(kids));
  };

  auto mkBinary = [&](LoweredExpr::Kind kind) -> std::shared_ptr<LoweredExpr> {
    auto a = planChild(0);
    auto b = planChild(1);
    if (!a || !b)
      return nullptr;
    return LoweredExpr::makeNode(kind, getZ3ExprSortKind(e),
                                 getZ3ExprBitWidth(e), {a, b});
  };

  auto mkUnary = [&](LoweredExpr::Kind kind) -> std::shared_ptr<LoweredExpr> {
    auto a = planChild(0);
    if (!a)
      return nullptr;
    return LoweredExpr::makeNode(kind, getZ3ExprSortKind(e),
                                 getZ3ExprBitWidth(e), {a});
  };

  switch (e.decl().decl_kind()) {
  case Z3_OP_NOT:
    return mkUnary(LoweredExpr::Not);

  case Z3_OP_AND:
    return mkNary(LoweredExpr::And);

  case Z3_OP_OR:
    return mkNary(LoweredExpr::Or);

  case Z3_OP_XOR:
    return mkNary(LoweredExpr::Xor);

  case Z3_OP_EQ:
    return mkBinary(LoweredExpr::Eq);

  case Z3_OP_DISTINCT:
    return mkNary(LoweredExpr::Distinct);

  case Z3_OP_ITE: {
    auto c = planChild(0);
    auto t = planChild(1);
    auto f = planChild(2);
    if (!c || !t || !f)
      return nullptr;
    return LoweredExpr::makeNode(LoweredExpr::Ite, getZ3ExprSortKind(e),
                                 getZ3ExprBitWidth(e), {c, t, f});
  }

  case Z3_OP_BNOT:
    return mkUnary(LoweredExpr::BVNot);

  case Z3_OP_BAND:
    return mkNary(LoweredExpr::BVAnd);

  case Z3_OP_BOR:
    return mkNary(LoweredExpr::BVOr);

  case Z3_OP_BXOR:
    return mkNary(LoweredExpr::BVXor);

  case Z3_OP_BADD:
    return mkNary(LoweredExpr::BVAdd);

  case Z3_OP_BSUB:
    return mkBinary(LoweredExpr::BVSub);

  case Z3_OP_BMUL:
    return mkNary(LoweredExpr::BVMul);

  case Z3_OP_BNEG:
    return mkUnary(LoweredExpr::BVNeg);

  case Z3_OP_CONCAT:
    return mkBinary(LoweredExpr::Concat);

  case Z3_OP_EXTRACT: {
    if (e.num_args() != 1) {
      llvm::errs() << "[WARN] Unexpected extract arity in counter expr: "
                   << e.to_string() << "\n";
      return nullptr;
    }
    auto in = planChild(0);
    if (!in)
      return nullptr;
    unsigned hi = Z3_get_decl_int_parameter(e.ctx(), e.decl(), 0);
    unsigned lo = Z3_get_decl_int_parameter(e.ctx(), e.decl(), 1);
    return LoweredExpr::makeExtract(in, hi, lo);
  }

  case Z3_OP_ULT:
    return mkBinary(LoweredExpr::ULess);
  case Z3_OP_ULEQ:
    return mkBinary(LoweredExpr::ULessEq);
  case Z3_OP_UGT:
    return mkBinary(LoweredExpr::UGreater);
  case Z3_OP_UGEQ:
    return mkBinary(LoweredExpr::UGreaterEq);

  case Z3_OP_SLT:
    return mkBinary(LoweredExpr::SLess);
  case Z3_OP_SLEQ:
    return mkBinary(LoweredExpr::SLessEq);
  case Z3_OP_SGT:
    return mkBinary(LoweredExpr::SGreater);
  case Z3_OP_SGEQ:
    return mkBinary(LoweredExpr::SGreaterEq);

  case Z3_OP_BSHL:
    return mkBinary(LoweredExpr::Shl);
  case Z3_OP_BLSHR:
    return mkBinary(LoweredExpr::LShr);
  case Z3_OP_BASHR:
    return mkBinary(LoweredExpr::AShr);

  default:
    llvm::errs() << "[WARN] Unsupported Z3 op while lowering counter expr: "
                 << e.to_string() << "\n";
    return nullptr;
  }
}

static mlir::Region *getInsertionRegion(mlir::OpBuilder &b) {
  mlir::Block *block = b.getInsertionBlock();
  return block ? block->getParent() : nullptr;
}

static bool checkValueInInsertionRegion(mlir::OpBuilder &b, mlir::Value v,
                                        llvm::StringRef what) {
  if (!v) {
    llvm::errs() << "[ERROR] null value for " << what << "\n";
    return false;
  }

  mlir::Region *valueRegion = getOwningRegion(v);
  mlir::Region *insertRegion = getInsertionRegion(b);

  if (!valueRegion) {
    llvm::errs() << "[ERROR] value for " << what << " has no owning region\n";
    return false;
  }

  if (!insertRegion) {
    llvm::errs() << "[ERROR] builder has no insertion region for " << what
                 << "\n";
    return false;
  }
  if (valueRegion != insertRegion) {
    llvm::errs() << "[ERROR] value for " << what
                 << " is defined outside the insertion region\n";
    llvm::errs() << "  value: " << v << "\n";

    llvm::errs() << "  valueRegion ptr: " << valueRegion << "\n";
    if (valueRegion && valueRegion->getParentOp()) {
      Operation *parent = valueRegion->getParentOp();
      llvm::errs() << "  valueRegion parent op: " << parent->getName();

      if (auto mod = llvm::dyn_cast<FModuleLike>(parent))
        llvm::errs() << " @" << mod.getModuleName();
      llvm::errs() << "\n";
    } else {
      llvm::errs() << "  valueRegion parent op: <none>\n";
    }

    llvm::errs() << "  insertRegion ptr: " << insertRegion << "\n";
    if (insertRegion && insertRegion->getParentOp()) {
      Operation *parent = insertRegion->getParentOp();
      llvm::errs() << "  insertRegion parent op: " << parent->getName();

      if (auto mod = llvm::dyn_cast<FModuleLike>(parent))
        llvm::errs() << " @" << mod.getModuleName();
      llvm::errs() << "\n";
    } else {
      llvm::errs() << "  insertRegion parent op: <none>\n";
    }

    if (auto *defOp = v.getDefiningOp()) {
      llvm::errs() << "  defining op:\n";
      defOp->dump();
    } else if (auto barg = llvm::dyn_cast<mlir::BlockArgument>(v)) {
      llvm::errs() << "  block argument #" << barg.getArgNumber() << "\n";
      llvm::errs() << "  owner block: " << barg.getOwner() << "\n";
    }

    return false;
  }

  return true;
}

static mlir::Value createUIntConst(mlir::OpBuilder &b, mlir::Location loc,
                                   unsigned width, const llvm::APInt &value) {
  if (!getInsertionRegion(b)) {
    llvm::errs() << "[ERROR] createUIntConst called with no insertion region\n";
    return {};
  }

  width = std::max(1u, width);
  auto ty = UIntType::get(b.getContext(), width);
  auto attrTy = mlir::IntegerType::get(b.getContext(), width,
                                       mlir::IntegerType::Unsigned);
  auto attr = mlir::IntegerAttr::get(attrTy, value.zextOrTrunc(width));
  return b.create<ConstantOp>(loc, ty, attr);
}

static mlir::Value createSIntConst(mlir::OpBuilder &b, mlir::Location loc,
                                   unsigned width, const llvm::APInt &value) {
  if (!getInsertionRegion(b)) {
    llvm::errs() << "[ERROR] createSIntConst called with no insertion region\n";
    return {};
  }

  width = std::max(1u, width);
  auto ty = SIntType::get(b.getContext(), width);
  auto attrTy =
      mlir::IntegerType::get(b.getContext(), width, mlir::IntegerType::Signed);
  auto attr = mlir::IntegerAttr::get(attrTy, value.sextOrTrunc(width));
  return b.create<ConstantOp>(loc, ty, attr);
}

static mlir::Value coerceToUInt(mlir::OpBuilder &b, mlir::Location loc,
                                mlir::Value v) {
  if (!checkValueInInsertionRegion(b, v, "coerceToUInt input"))
    return {};

  auto firTy = v.getType().dyn_cast<FIRRTLType>();
  if (!firTy)
    return {};

  if (firTy.isa<UIntType>())
    return v;
  if (firTy.isa<SIntType, ResetType, AsyncResetType>())
    return b.create<AsUIntPrimOp>(loc, v);

  return {};
}

static mlir::Value coerceToSInt(mlir::OpBuilder &b, mlir::Location loc,
                                mlir::Value v) {
  if (!checkValueInInsertionRegion(b, v, "coerceToSInt input"))
    return {};

  auto firTy = v.getType().dyn_cast<FIRRTLType>();
  if (!firTy)
    return {};

  if (firTy.isa<SIntType>())
    return v;
  if (firTy.isa<UIntType>())
    return b.create<AsSIntPrimOp>(loc, v);
  if (firTy.isa<ResetType, AsyncResetType>()) {
    auto asUInt = b.create<AsUIntPrimOp>(loc, v);
    return b.create<AsSIntPrimOp>(loc, asUInt);
  }

  return {};
}

static mlir::Value coerceToPredicate(mlir::OpBuilder &b, mlir::Location loc,
                                     mlir::Value v) {
  if (!checkValueInInsertionRegion(b, v, "coerceToPredicate input"))
    return {};

  auto firTy = v.getType().dyn_cast<FIRRTLType>();
  if (!firTy)
    return {};

  if (firTy.isa<ResetType, AsyncResetType>())
    return b.create<AsUIntPrimOp>(loc, v);

  if (auto uintTy = firTy.dyn_cast<UIntType>()) {
    unsigned w = std::max(1u, (unsigned)uintTy.getWidthOrSentinel());
    if (w == 1)
      return v;
    auto zero = createUIntConst(b, loc, w, llvm::APInt(w, 0));
    if (!zero)
      return {};
    return b.create<NEQPrimOp>(loc, v, zero);
  }

  if (auto sintTy = firTy.dyn_cast<SIntType>()) {
    unsigned w = std::max(1u, (unsigned)sintTy.getWidthOrSentinel());
    auto zero = createSIntConst(b, loc, w, llvm::APInt(w, 0));
    if (!zero)
      return {};
    return b.create<NEQPrimOp>(loc, v, zero);
  }

  return {};
}

static mlir::Value materializeExpr(mlir::OpBuilder &b, mlir::Location loc,
                                   const std::shared_ptr<LoweredExpr> &expr);

static mlir::Value materializeNary(
    mlir::OpBuilder &b, mlir::Location loc,
    const llvm::SmallVectorImpl<std::shared_ptr<LoweredExpr>> &children,
    llvm::function_ref<mlir::Value(mlir::Value, mlir::Value)> combine) {
  if (children.empty())
    return {};

  auto acc = materializeExpr(b, loc, children[0]);
  if (!acc)
    return {};

  for (size_t i = 1; i < children.size(); ++i) {
    auto rhs = materializeExpr(b, loc, children[i]);
    if (!rhs)
      return {};
    acc = combine(acc, rhs);
    if (!acc)
      return {};
  }
  return acc;
}

static mlir::Value materializeCompare(mlir::OpBuilder &b, mlir::Location loc,
                                      LoweredExpr::Kind kind, mlir::Value lhs,
                                      mlir::Value rhs) {
  if (!lhs || !rhs)
    return {};

  switch (kind) {
  case LoweredExpr::ULess:
    lhs = coerceToUInt(b, loc, lhs);
    rhs = coerceToUInt(b, loc, rhs);
    return (lhs && rhs) ? b.create<LTPrimOp>(loc, lhs, rhs) : mlir::Value{};
  case LoweredExpr::ULessEq:
    lhs = coerceToUInt(b, loc, lhs);
    rhs = coerceToUInt(b, loc, rhs);
    return (lhs && rhs) ? b.create<LEQPrimOp>(loc, lhs, rhs) : mlir::Value{};
  case LoweredExpr::UGreater:
    lhs = coerceToUInt(b, loc, lhs);
    rhs = coerceToUInt(b, loc, rhs);
    return (lhs && rhs) ? b.create<GTPrimOp>(loc, lhs, rhs) : mlir::Value{};
  case LoweredExpr::UGreaterEq:
    lhs = coerceToUInt(b, loc, lhs);
    rhs = coerceToUInt(b, loc, rhs);
    return (lhs && rhs) ? b.create<GEQPrimOp>(loc, lhs, rhs) : mlir::Value{};

  case LoweredExpr::SLess:
    lhs = coerceToSInt(b, loc, lhs);
    rhs = coerceToSInt(b, loc, rhs);
    return (lhs && rhs) ? b.create<LTPrimOp>(loc, lhs, rhs) : mlir::Value{};
  case LoweredExpr::SLessEq:
    lhs = coerceToSInt(b, loc, lhs);
    rhs = coerceToSInt(b, loc, rhs);
    return (lhs && rhs) ? b.create<LEQPrimOp>(loc, lhs, rhs) : mlir::Value{};
  case LoweredExpr::SGreater:
    lhs = coerceToSInt(b, loc, lhs);
    rhs = coerceToSInt(b, loc, rhs);
    return (lhs && rhs) ? b.create<GTPrimOp>(loc, lhs, rhs) : mlir::Value{};
  case LoweredExpr::SGreaterEq:
    lhs = coerceToSInt(b, loc, lhs);
    rhs = coerceToSInt(b, loc, rhs);
    return (lhs && rhs) ? b.create<GEQPrimOp>(loc, lhs, rhs) : mlir::Value{};

  default:
    return {};
  }
}

static mlir::Value materializeExpr(mlir::OpBuilder &b, mlir::Location loc,
                                   const std::shared_ptr<LoweredExpr> &expr) {
  if (!expr)
    return {};

  switch (expr->kind) {
  case LoweredExpr::Leaf:
    return expr->leafValue;

  case LoweredExpr::Constant:
    return createUIntConst(b, loc, expr->bitWidth, expr->constValue);

  case LoweredExpr::Not: {
    auto a = materializeExpr(b, loc, expr->children[0]);
    if (!a)
      return {};
    a = coerceToPredicate(b, loc, a);
    return a ? b.create<NotPrimOp>(loc, a) : mlir::Value{};
  }

  case LoweredExpr::And: {
    return materializeNary(
        b, loc, expr->children, [&](mlir::Value lhs, mlir::Value rhs) {
          lhs = coerceToPredicate(b, loc, lhs);
          rhs = coerceToPredicate(b, loc, rhs);
          return (lhs && rhs) ? b.create<AndPrimOp>(loc, lhs, rhs)
                              : mlir::Value{};
        });
  }

  case LoweredExpr::Or: {
    return materializeNary(
        b, loc, expr->children, [&](mlir::Value lhs, mlir::Value rhs) {
          lhs = coerceToPredicate(b, loc, lhs);
          rhs = coerceToPredicate(b, loc, rhs);
          return (lhs && rhs) ? b.create<OrPrimOp>(loc, lhs, rhs)
                              : mlir::Value{};
        });
  }

  case LoweredExpr::Xor: {
    return materializeNary(
        b, loc, expr->children, [&](mlir::Value lhs, mlir::Value rhs) {
          lhs = coerceToPredicate(b, loc, lhs);
          rhs = coerceToPredicate(b, loc, rhs);
          return (lhs && rhs) ? b.create<XorPrimOp>(loc, lhs, rhs)
                              : mlir::Value{};
        });
  }

  case LoweredExpr::Eq: {
    auto lhs = materializeExpr(b, loc, expr->children[0]);
    auto rhs = materializeExpr(b, loc, expr->children[1]);
    if (!lhs || !rhs)
      return {};
    return b.create<EQPrimOp>(loc, lhs, rhs);
  }
  case LoweredExpr::Distinct: {
    if (expr->children.size() < 2)
      return createUIntConst(b, loc, 1, llvm::APInt(1, 1));

    mlir::Value acc;
    for (size_t i = 0; i < expr->children.size(); ++i) {
      for (size_t j = i + 1; j < expr->children.size(); ++j) {
        auto a = materializeExpr(b, loc, expr->children[i]);
        auto c = materializeExpr(b, loc, expr->children[j]);
        if (!a || !c)
          return {};

        auto neq = b.create<NEQPrimOp>(loc, a, c).getResult();
        if (!acc)
          acc = neq;
        else
          acc = b.create<AndPrimOp>(loc, acc, neq).getResult();
      }
    }
    return acc;
  }

  case LoweredExpr::Ite: {
    auto cond = materializeExpr(b, loc, expr->children[0]);
    auto tval = materializeExpr(b, loc, expr->children[1]);
    auto fval = materializeExpr(b, loc, expr->children[2]);
    if (!cond || !tval || !fval)
      return {};
    cond = coerceToPredicate(b, loc, cond);
    return cond ? b.create<MuxPrimOp>(loc, cond, tval, fval) : mlir::Value{};
  }

  case LoweredExpr::BVNot: {
    auto a = materializeExpr(b, loc, expr->children[0]);
    return a ? b.create<NotPrimOp>(loc, a) : mlir::Value{};
  }

  case LoweredExpr::BVAnd: {
    return materializeNary(b, loc, expr->children,
                           [&](mlir::Value lhs, mlir::Value rhs) {
                             return b.create<AndPrimOp>(loc, lhs, rhs);
                           });
  }

  case LoweredExpr::BVOr: {
    return materializeNary(b, loc, expr->children,
                           [&](mlir::Value lhs, mlir::Value rhs) {
                             return b.create<OrPrimOp>(loc, lhs, rhs);
                           });
  }

  case LoweredExpr::BVXor: {
    return materializeNary(b, loc, expr->children,
                           [&](mlir::Value lhs, mlir::Value rhs) {
                             return b.create<XorPrimOp>(loc, lhs, rhs);
                           });
  }

  case LoweredExpr::BVAdd: {
    return materializeNary(b, loc, expr->children,
                           [&](mlir::Value lhs, mlir::Value rhs) {
                             return b.create<AddPrimOp>(loc, lhs, rhs);
                           });
  }

  case LoweredExpr::BVSub: {
    auto lhs = materializeExpr(b, loc, expr->children[0]);
    auto rhs = materializeExpr(b, loc, expr->children[1]);
    return (lhs && rhs) ? b.create<SubPrimOp>(loc, lhs, rhs) : mlir::Value{};
  }

  case LoweredExpr::BVMul: {
    return materializeNary(b, loc, expr->children,
                           [&](mlir::Value lhs, mlir::Value rhs) {
                             return b.create<MulPrimOp>(loc, lhs, rhs);
                           });
  }

  case LoweredExpr::BVNeg: {
    auto a = materializeExpr(b, loc, expr->children[0]);
    if (!a)
      return {};
    auto firTy = a.getType().dyn_cast<FIRRTLType>();
    if (!firTy)
      return {};
    unsigned w = expr->bitWidth ? expr->bitWidth : 1;
    auto zero = createUIntConst(b, loc, w, llvm::APInt(w, 0));
    return b.create<SubPrimOp>(loc, zero, a);
  }

  case LoweredExpr::Concat: {
    auto lhs = materializeExpr(b, loc, expr->children[0]);
    auto rhs = materializeExpr(b, loc, expr->children[1]);
    return (lhs && rhs) ? b.create<CatPrimOp>(loc, lhs, rhs) : mlir::Value{};
  }

  case LoweredExpr::Extract: {
    auto in = materializeExpr(b, loc, expr->children[0]);
    return in ? b.create<BitsPrimOp>(loc, in, expr->extractHi, expr->extractLo)
              : mlir::Value{};
  }

  case LoweredExpr::ULess:
  case LoweredExpr::ULessEq:
  case LoweredExpr::UGreater:
  case LoweredExpr::UGreaterEq:
  case LoweredExpr::SLess:
  case LoweredExpr::SLessEq:
  case LoweredExpr::SGreater:
  case LoweredExpr::SGreaterEq: {
    auto lhs = materializeExpr(b, loc, expr->children[0]);
    auto rhs = materializeExpr(b, loc, expr->children[1]);
    return materializeCompare(b, loc, expr->kind, lhs, rhs);
  }

  case LoweredExpr::Shl: {
    auto lhs = materializeExpr(b, loc, expr->children[0]);
    auto rhs = materializeExpr(b, loc, expr->children[1]);
    return (lhs && rhs) ? b.create<DShlPrimOp>(loc, lhs, rhs) : mlir::Value{};
  }

  case LoweredExpr::LShr: {
    auto lhs = materializeExpr(b, loc, expr->children[0]);
    auto rhs = materializeExpr(b, loc, expr->children[1]);
    return (lhs && rhs) ? b.create<DShrPrimOp>(loc, lhs, rhs) : mlir::Value{};
  }

  case LoweredExpr::AShr: {
    auto lhs = materializeExpr(b, loc, expr->children[0]);
    auto rhs = materializeExpr(b, loc, expr->children[1]);
    if (!lhs || !rhs)
      return {};
    lhs = coerceToSInt(b, loc, lhs);
    return (lhs && rhs) ? b.create<DShrPrimOp>(loc, lhs, rhs) : mlir::Value{};
  }
  }

  return {};
}

static mlir::Value buildNamedCounterWire(FModuleLike module, mlir::Location loc,
                                         StringRef counterName) {
  auto fmodule = llvm::dyn_cast<FModuleOp>(module.getOperation());
  if (!fmodule)
    return {};

  mlir::Block *body = fmodule.getBodyBlock();
  if (!body)
    return {};

  mlir::OpBuilder b(body, body->end());
  auto *ctx = b.getContext();

  auto emptyAnnos = b.getArrayAttr({});
  auto wireNameAttr = b.getStringAttr(counterName);

  auto nameKindAttr = circt::firrtl::NameKindEnumAttr::get(
      ctx, circt::firrtl::NameKindEnum::InterestingName);

  mlir::OperationState wireState(loc,
                                 circt::firrtl::WireOp::getOperationName());
  wireState.addTypes(UIntType::get(ctx, 1));
  wireState.addAttribute("name", wireNameAttr);
  wireState.addAttribute("nameKind", nameKindAttr);
  wireState.addAttribute("annotations", emptyAnnos);

  auto *wireRaw = b.create(wireState);
  auto wireOp = llvm::cast<circt::firrtl::WireOp>(wireRaw);
  return wireOp.getResult();
}

static bool connectExprInModule(FModuleOp module, mlir::Location loc,
                                const std::shared_ptr<LoweredExpr> &expr,
                                mlir::Value destWire) {
  if (!module || !destWire || !expr)
    return false;

  mlir::Block *body = module.getBodyBlock();
  if (!body)
    return false;

  mlir::OpBuilder b(body, body->end());
  auto v = materializeExpr(b, loc, expr);
  if (!v)
    return false;

  v = coerceToPredicate(b, loc, v);
  if (!v)
    return false;

  b.create<ConnectOp>(loc, destWire, v);
  return true;
}

static bool isBooleanLikeTargetType(FIRRTLType type) {
  if (!type)
    return false;

  if (auto uintTy = llvm::dyn_cast<UIntType>(type))
    return uintTy.getWidthOrSentinel() == 1;

  if (type.isa<ResetType, AsyncResetType>())
    return true;

  return false;
}
} // namespace

bool CounterInserter::tryInsertDirectTargetCounter(
    const std::string &counterName, const z3::expr &counterExpr,
    const std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap,
    const std::string descr) {
  llvm::errs() << "Trying to insert direct counter for " << counterName
               << "with " << descr << "\n";
  auto it = annoMap.find(descr);
  if (it == annoMap.end()) {
    llvm::errs() << "No direct target annotation for counter '" << counterName
                 << "'; falling back to leaf-based insertion\n";
    return false;
  }

  if (insertCache.count(descr)) {
    llvm::errs() << "Counter '" << counterName
                 << "' already inserted for description '" << descr
                 << "'; skipping redundant insertion\n";
    return true;
  }

  const auto &apv = it->second;

  if (!apv.ref) {
    llvm::errs() << "[ERROR] CounterInserter: target '" << counterName
                 << "' has null annotation target\n";
    return false;
  }

  auto targetModule = apv.ref.getModule();
  if (!targetModule) {
    llvm::errs() << "[ERROR] CounterInserter: target '" << counterName
                 << "' has no containing module\n";
    return true;
  }

  auto targetFModule = llvm::dyn_cast<FModuleOp>(targetModule.getOperation());
  if (!targetFModule) {
    llvm::errs() << "[ERROR] CounterInserter: target '" << counterName
                 << "' is not contained in an FModuleOp\n";
    return true;
  }

  mlir::Value targetValue = getValueFromAnnoPathValue(apv);
  if (!targetValue) {
    llvm::errs() << "[ERROR] CounterInserter: could not recover mlir::Value "
                 << "for direct target '" << counterName << "'\n";
    return true;
  }

  auto firTy = targetValue.getType().dyn_cast<FIRRTLType>();
  if (!firTy) {
    llvm::errs() << "[ERROR] CounterInserter: direct target '" << counterName
                 << "' does not have a FIRRTL type\n";
    return true;
  }

  if (!isBooleanLikeTargetType(firTy)) {
    llvm::errs() << "[WARN] CounterInserter: direct target '" << counterName
                 << "' is not boolean-like (type: " << firTy
                 << "); skipping direct counter insertion\n";
    return true;
  }

  mlir::Block *body = targetFModule.getBodyBlock();
  if (!body) {
    llvm::errs() << "[ERROR] CounterInserter: target module '"
                 << targetModule.getModuleName() << "' has no body block\n";
    return true;
  }

  auto clockReset = findModuleClockAndReset(targetFModule);
  if (!clockReset.clock.value) {
    llvm::errs() << "[ERROR] CounterInserter: could not find visible clock in "
                 << "module " << targetModule.getModuleName() << "\n";
    return true;
  }
  mlir::OpBuilder b(body, body->end());
  mlir::Location loc = targetModule.getLoc();
  // Create a 1-bit wire for the condition
  mlir::Value wire = buildNamedCounterWire(targetModule, loc, counterName);
  if (!wire) {
    llvm::errs() << "[ERROR] CounterInserter: failed to create wire for '"
                 << counterName << "'\n";
    return true;
  }

  mlir::Value pred = targetValue;
  if (!isBooleanLikeTargetType(firTy)) {
    llvm::errs() << "[WARN] CounterInserter: direct target '" << counterName
                 << "' is not boolean-like (type: " << firTy
                 << "); coercing via != 0\n";
    pred = coerceToPredicate(b, loc, targetValue);
    if (!pred)
      return true;
  }

  insertCache.insert({descr, pred});
  b.create<circt::firrtl::ConnectOp>(loc, wire, pred);

  auto wireNameAttr = b.getStringAttr(counterName);
  auto descAttr = b.getStringAttr(counterExpr.to_string());

  b.create<circt::perf::PerfCounterOp>(loc, wire, clockReset.clock.value,
                                       clockReset.reset.value, wireNameAttr,
                                       descAttr);

  llvm::errs() << "CounterInserter: inserted direct counter '" << counterName
               << "' in module " << targetModule.getModuleName() << "\n";
  ;

  return true;
}

std::optional<CounterInserter::AddedOutputPortInfo>
CounterInserter::addOutputPortToModuleAndInstances(
    FModuleLike module, mlir::Value exportedValue, StringRef debugName,
    circt::firrtl::InstanceOp pathInstance,
    std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap,
    circt::firrtl::FModuleLike targetModule) {
  auto fmodule = mlir::dyn_cast<FModuleOp>(module.getOperation());
  if (!fmodule)
    return std::nullopt;

  auto loc = exportedValue.getLoc();
  auto portType = exportedValue.getType();
  unsigned newPortIdx = fmodule.getNumPorts();

  auto cleanName = sanitizePortName(debugName);

  auto targetModuleName = targetModule.getName().str();
  auto sourceModuleName = fmodule.getModuleName().str();


  cleanName = cleanName + "_lift_" + sourceModuleName + "_to_" +
              targetModuleName + "_id" + std::to_string(globalPortId++);

  PortInfo newPortInfo{
      StringAttr::get(fmodule.getContext(), cleanName),
      portType,
      Direction::Out,
      {},
      loc,
      AnnotationSet(portType.getContext()),
  };

  llvm::errs() << "Number of old ports: " << fmodule.getNumPorts() << "\n";
  fmodule.insertPorts({{newPortIdx, newPortInfo}});
  llvm::errs() << "Number of old ports: " << fmodule.getNumPorts() << "\n";
  auto *body = fmodule.getBodyBlock();
  if (!body)
    return std::nullopt;

  if (newPortIdx >= body->getNumArguments())
    return std::nullopt;

  auto newPortArg = body->getArgument(newPortIdx);
  OpBuilder b = OpBuilder::atBlockEnd(body);
  b.create<ConnectOp>(loc, newPortArg, exportedValue);

  auto *node = instanceGraph.lookup(module);
  if (!node)
    return std::nullopt;

  SmallVector<InstanceOp> instancesToRewrite;
  for (auto *use : node->uses()) {
    if (auto inst = use->getInstance<InstanceOp>())
      instancesToRewrite.push_back(inst);
  }

  InstanceOp rewrittenPathInstance;
  for (auto inst : instancesToRewrite) {
    OpBuilder ib(inst);
    SmallVector<Attribute> annos;
    if (auto a = inst->getAttrOfType<ArrayAttr>("annotations"))
      annos.append(a.begin(), a.end());

    auto emptyAnnoArray = ArrayAttr::get(inst.getContext(), {});

    SmallVector<Attribute> portAnnos;
    portAnnos.resize(inst->getNumResults(), emptyAnnoArray);

    if (auto a = inst->getAttrOfType<ArrayAttr>("portAnnotations")) {
      unsigned n = std::min<unsigned>(a.size(), portAnnos.size());
      for (unsigned i = 0; i < n; ++i)
        portAnnos[i] = a[i];
    }

    portAnnos.push_back(emptyAnnoArray);

    SmallVector<Type> resultTypes;
    SmallVector<Direction> portDirections;
    SmallVector<Attribute> portNames;
    for (auto port : module.getPorts()) {
      resultTypes.push_back(port.type);
      portDirections.push_back(port.direction);
      portNames.push_back(port.name);
    }

    SmallVector<Attribute> layers;
    if (auto la = inst->getAttrOfType<ArrayAttr>("layers"))
      layers.append(la.begin(), la.end());

    llvm::errs() << "Old inst: ";
    // inst->print(llvm::errs());
    llvm::errs() << "\n";
    llvm::errs() << "  old num results      = " << inst->getNumResults()
                 << "\n";
    llvm::errs() << "  old portAnnos size   = " << portAnnos.size() - 1 << "\n";
    llvm::errs() << "  new portAnnos size   = " << portAnnos.size() << "\n";
    llvm::errs() << "  rebuilt resultTypes  = " << resultTypes.size() << "\n";
    llvm::errs() << "  rebuilt directions   = " << portDirections.size()
                 << "\n";
    llvm::errs() << "  rebuilt portNames    = " << portNames.size() << "\n";

    auto newInst = ib.create<InstanceOp>(
        inst.getLoc(), TypeRange(resultTypes), module.getModuleName(),
        inst.getInstanceName(), inst.getNameKind(),
        ArrayRef<Direction>(portDirections), ArrayRef<Attribute>(portNames),
        ArrayRef<Attribute>(annos), ArrayRef<Attribute>(portAnnos),
        ArrayRef<Attribute>(layers), inst.getLowerToBind(),
        inst.getInnerSymAttr());

    for (auto namedAttr : inst->getAttrs()) {
      auto name = namedAttr.getName().getValue();

      if (name == "moduleName" || name == "name" || name == "nameKind" ||
          name == "annotations" || name == "portAnnotations" ||
          name == "portDirections" || name == "portNames" || name == "layers" ||
          name == "lowerToBind" || name == "inner_sym" || name == "innerSym")
        continue;

      newInst->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    updatePortInsertCacheValues(inst, newInst);

    for (unsigned i = 0, e = inst->getNumResults(); i < e; ++i)
      inst.getResult(i).replaceAllUsesWith(newInst.getResult(i));

    instanceGraph.replaceInstance(inst, newInst);

    if (inst == pathInstance)
      rewrittenPathInstance = newInst;

    llvm::errs() << "New inst: ";
    // newInst->print(llvm::errs());
    llvm::errs() << "\n";

    updateAnnoMap(annoMap, inst, newInst);
    inst.erase();
  }

  if (!rewrittenPathInstance)
    return std::nullopt;

  return CounterInserter::AddedOutputPortInfo{newPortIdx,
                                              rewrittenPathInstance};
}

mlir::Value CounterInserter::makeValueVisibleInModule(
    const AnnoPathValue &apv, mlir::Value value, FModuleLike targetModule,
    StringRef debugName,
    std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap) {

  if (!value)
    return {};

  auto cacheKey = std::make_pair(value, targetModule);
  auto cacheIt = portInsertCache.find(cacheKey);
  if (cacheIt != portInsertCache.end())
    return cacheIt->second;

  // Trivial cases, value already is in the target module, either as an op
  // result or block argument.
  Operation *targetOp = targetModule.getOperation();
  if (Operation *defOp = value.getDefiningOp()) {
    if (defOp->getParentOp() == targetOp)
      return value;
  }

  if (auto barg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    if (barg.getOwner()->getParentOp() == targetOp)
      return value;
  }

  auto sourceModule = apv.ref.getModule();
  if (!sourceModule || !value || !targetModule)
    return {};

  if (sourceModule == targetModule)
    return value;

  llvm::errs() << "[PLUMBING DEBUG] For leaf '" << debugName
               << "', source value: " << value
               << "target module name:" << targetModule.getName() << "\n";

  llvm::SmallVector<InstanceOp, 8> path;
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;

  bool found = findPathToAncestor(instanceGraph, sourceModule, targetModule,
                                  path, visited);

  if (!found) {
    llvm::errs() << "[CounterInserter] no instance path found for signal '"
                 << debugName << "' from module "
                 << sourceModule.getModuleName() << " to module "
                 << targetModule.getModuleName() << "\n";
    return {};
  }

  mlir::Value currentValue = value;
  FModuleLike currentModule = sourceModule;

  for (auto instOnPath : path) {
    auto parentModule = mlir::dyn_cast<FModuleLike>(instOnPath->getParentOp());
    if (!parentModule)
      return {};

    auto added = addOutputPortToModuleAndInstances(currentModule, currentValue,
                                                   debugName, instOnPath,
                                                   annoMap, targetModule);
    if (!added)
      return {};

    auto updatedInst = added->rewrittenPathInstance;
    unsigned newPortIdx = added->newPortIdx;

    if (newPortIdx >= updatedInst->getNumResults()) {
      llvm::errs() << "[CounterInserter] new port index out of range on "
                      "rewritten instance:\n";
      updatedInst->print(llvm::errs());
      llvm::errs() << "\n";
      return {};
    }

    currentValue = updatedInst->getResult(newPortIdx);
    currentModule = parentModule;
  }

  portInsertCache[cacheKey] = currentValue;
  return currentValue;
}

void CounterInserter::insertPerfCounters(
    z3::expr counterExpr, const std::string &counterName,
    std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap,
    const std::string descr, bool onlyDirect) {
  auto circuitOp = llvm::dyn_cast<circt::firrtl::CircuitOp>(module);
  if (!circuitOp) {
    llvm::errs() << "[ERROR] CounterInserter: module is not a CircuitOp\n";
    return;
  }

  if (tryInsertDirectTargetCounter(counterName, counterExpr, annoMap, descr)) {
    numCountersInserted++;
    return;
  }

  if (onlyDirect) {
    numCountersInserted++; // Just so we don't get unnecessary misses for now
    return;
  }

  if (insertCache.count(counterExpr.to_string())) {
    numCountersInserted++;
    return;
  }
  
  llvm::SmallVector<z3::expr, 16> leaves;
  llvm::SmallPtrSet<Z3_ast, 16> seen;
  collectReferencedLeaves(counterExpr, leaves, seen);

  if (leaves.empty()) {
    llvm::errs()
        << "[WARN] CounterInserter: no referenced leaves in expression "
        << counterExpr.to_string() << " and counterName: " << counterName
        << "\n";
    return;
  }

  llvm::SmallVector<const AnnoPathValue *, 16> annoPaths;
  annoPaths.reserve(leaves.size());

  llvm::SmallVector<std::string, 16> leafNames;
  leafNames.reserve(leaves.size());

  // First collect names and anno paths from the current anno map.
  for (const auto &leaf : leaves) {
    std::string name = leaf.decl().name().str();

    auto annoIt = annoMap.find(name);
    if (annoIt == annoMap.end()) {
      llvm::errs()
          << "[ERROR] CounterInserter: missing AnnoPathValue for leaf '" << name
          << "'\n";
      return;
    }

    leafNames.push_back(name);
    annoPaths.push_back(&annoIt->second);
  }

  // Compute the insertion module before any visibility rewrites.
  FModuleLike lcaModule = findLowestCommonModule(circuitOp, annoPaths);
  if (!lcaModule) {
    llvm::errs()
        << "[ERROR] CounterInserter: failed to compute common module for '"
        << counterName << "'\n";
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "CounterInserter: chosen insertion module: "
                          << lcaModule.getModuleName() << "\n");

  // First pass: force all required rewrites to make each leaf visible in the
  // LCA module. Do not cache returned Values here, since later rewrites may
  // stale them.
  for (const auto &name : leafNames) {
    auto annoIt = annoMap.find(name);
    if (annoIt == annoMap.end()) {
      llvm::errs()
          << "[ERROR] CounterInserter: missing AnnoPathValue for leaf '" << name
          << "' during visibility rewrite\n";
      return;
    }

    const auto &apv = annoIt->second;
    mlir::Value sourceValue = getValueFromAnnoPathValue(apv);
    if (!sourceValue) {
      llvm::errs()
          << "[ERROR] CounterInserter: could not recover mlir::Value for '"
          << name << "' from AnnoPathValue\n";
      return;
    }

    mlir::Value visibleValue =
        makeValueVisibleInModule(apv, sourceValue, lcaModule, name, annoMap);
    if (!visibleValue) {
      llvm::errs() << "[ERROR] CounterInserter: failed to make '" << name
                   << "' visible in module " << lcaModule.getModuleName()
                   << "\n";
      return;
    }
  }

  // Second pass: rebuild the final leaf map from the final IR state.
  // Do NOT use getValueFromAnnoPathValue(apv) directly unless annoMap.ref has
  // been updated to point at the visible value. Instead, ask
  // makeValueVisibleInModule again and use the returned value directly.
  llvm::DenseMap<Z3_ast, mlir::Value> leafMap;
  for (const auto &leaf : leaves) {
    std::string name = leaf.decl().name().str();

    auto annoIt = annoMap.find(name);
    if (annoIt == annoMap.end()) {
      llvm::errs()
          << "[ERROR] CounterInserter: missing final AnnoPathValue for '"
          << name << "' after rewrites\n";
      return;
    }

    const auto &apv = annoIt->second;
    mlir::Value sourceValue = getValueFromAnnoPathValue(apv);
    if (!sourceValue) {
      llvm::errs()
          << "[ERROR] CounterInserter: failed to recover source value for '"
          << name << "' after rewrites\n";
      return;
    }

    mlir::Value finalVisibleValue =
        makeValueVisibleInModule(apv, sourceValue, lcaModule, name, annoMap);
    if (!finalVisibleValue) {
      llvm::errs()
          << "[ERROR] CounterInserter: failed to recover final visible "
          << "value for '" << name << "' after rewrites\n";
      return;
    }

    leafMap[(Z3_ast)leaf] = finalVisibleValue;
  }

  auto lcaFModule = llvm::dyn_cast<FModuleOp>(lcaModule.getOperation());
  if (!lcaFModule) {
    llvm::errs()
        << "[ERROR] CounterInserter: insertion module is not an FModuleOp\n";
    return;
  }

  mlir::Block *body = lcaFModule.getBodyBlock();
  if (!body) {
    llvm::errs()
        << "[ERROR] CounterInserter: insertion module has no body block\n";
    return;
  }

  mlir::Location loc = lcaModule.getLoc();

  auto plan = planExprIntoModule(counterExpr, leafMap);
  if (!plan) {
    llvm::errs() << "[WARN] CounterInserter: skipping counter '" << counterName
                 << "' because expression planning failed: "
                 << counterExpr.to_string() << "\n";
    return;
  }

  auto clockReset = findModuleClockAndReset(lcaFModule);
  if (!clockReset.clock.value) {
    llvm::errs()
        << "[ERROR] CounterInserter: could not find visible clock in module "
        << lcaModule.getModuleName() << "\n";
    return;
  }

  mlir::Value wiredCond = buildNamedCounterWire(lcaModule, loc, counterName);
  if (!wiredCond) {
    llvm::errs() << "[ERROR] CounterInserter: failed to create counter wire '"
                 << counterName << "'\n";
    return;
  }

  if (!connectExprInModule(lcaFModule, loc, plan, wiredCond)) {
    llvm::errs() << "[WARN] CounterInserter: failed to materialize expression "
                 << "for '" << counterName << "' in module "
                 << lcaModule.getModuleName() << "\n";
    return;
  }

  // DEBUGGING
  auto *defOp = wiredCond ? wiredCond.getDefiningOp() : nullptr;
  auto wireOp = defOp ? llvm::dyn_cast<circt::firrtl::WireOp>(defOp) : nullptr;

  circt::firrtl::ConnectOp drivingConnect;
  mlir::Value drivingExpr;

  if (wireOp) {
    for (mlir::OpOperand &use : wireOp.getResult().getUses()) {
      auto connectOp = llvm::dyn_cast<circt::firrtl::ConnectOp>(use.getOwner());
      if (!connectOp)
        continue;

      if (connectOp.getDest() == wireOp.getResult()) {
        drivingConnect = connectOp;
        drivingExpr = connectOp.getSrc();
        break;
      }
    }
  }

  insertCache.insert({counterExpr.to_string(), drivingExpr});

  llvm::errs() << "Original expression: " << counterExpr.to_string() << "\n";

  llvm::errs() << "For " << counterName << " inserted wire: ";
  if (wireOp)
    llvm::errs() << *wireOp << "\n";
  else
    llvm::errs() << "<null wireOp>\n";

  llvm::errs() << "For " << counterName << " driving connect: ";
  if (drivingConnect)
    llvm::errs() << drivingConnect << "\n";
  else
    llvm::errs() << "<null drivingConnect>\n";

  llvm::errs() << "For " << counterName << " driving expression: ";
  if (drivingExpr) {
    if (auto *exprDef = drivingExpr.getDefiningOp())
      llvm::errs() << *exprDef << "\n";
    else
      llvm::errs() << "<block argument or value with no defining op>\n";
  } else {
    llvm::errs() << "<null drivingExpr>\n";
  }
  // DEBUGGING

  mlir::OpBuilder perfBuilder(body, body->end());
  auto wireNameAttr = perfBuilder.getStringAttr(counterName);
  auto descAttr = perfBuilder.getStringAttr(counterExpr.to_string());

  perfBuilder.create<circt::perf::PerfCounterOp>(
      loc, wiredCond, clockReset.clock.value, clockReset.reset.value,
      wireNameAttr, descAttr);
  numCountersInserted++;

  LLVM_DEBUG({
    llvm::dbgs() << "CounterInserter: inserted counter '" << counterName
                 << "' in module " << lcaModule.getModuleName()
                 << " using clock port '" << clockReset.clock.portName << "'";
    if (clockReset.reset.value)
      llvm::dbgs() << " and reset port '" << clockReset.reset.portName << "'";
    llvm::dbgs() << "\n";
  });
}

InstanceOp CounterInserter::resolveCurrentInstance(InstanceOp inst) {
  if (!inst)
    return {};

  mlir::Operation *start = inst.getOperation();
  mlir::Operation *op = start;
  SmallVector<mlir::Operation *, 8> chain;
  SmallPtrSet<mlir::Operation *, 8> visited;

  while (true) {
    if (!visited.insert(op).second) {
      llvm::errs() << "[CounterInserter] cycle in rewrittenInstanceMap\n";
      return {};
    }

    chain.push_back(op);

    auto it = rewrittenInstanceMap.find(op);
    if (it == rewrittenInstanceMap.end())
      break;

    if (!it->second)
      return {};

    op = it->second;
  }

  for (auto *oldOp : chain)
    rewrittenInstanceMap[oldOp] = op;

  return dyn_cast_or_null<InstanceOp>(op);
}
llvm::SmallVector<FModuleLike, 8>
CounterInserter::getModulePath(const AnnoPathValue &apv) {
  llvm::SmallVector<FModuleLike, 8> mods;

  if (apv.instances.empty()) {
    auto m = apv.ref.getModule();
    if (m)
      mods.push_back(m);
    return mods;
  }

  auto firstParent =
      dyn_cast<FModuleLike>(apv.instances.front()->getParentOp());
  if (firstParent)
    mods.push_back(firstParent);

  for (auto inst : apv.instances) {
    auto parent = dyn_cast<FModuleLike>(inst->getParentOp());
    if (!parent)
      continue;

    auto *parentNode = instanceGraph.lookup(parent);
    if (!parentNode)
      continue;

    for (auto *record : *parentNode) {
      auto recInst = record->getInstance<InstanceOp>();
      if (recInst != inst)
        continue;

      auto child = record->getTarget()->getModule<FModuleLike>();
      if (child)
        mods.push_back(child);
      break;
    }
  }

  return mods;
}

FModuleLike CounterInserter::findLowestCommonModule(
    circt::firrtl::CircuitOp circuit,
    llvm::ArrayRef<const AnnoPathValue *> paths) {
  if (paths.empty())
    return {};

  llvm::SmallVector<FModuleLike, 8> base = getModulePath(*paths.front());
  if (base.empty()) {
    llvm::errs() << "[PATH] No module path found for first annotation path for "
                 << *paths.front() << "\n";
    return {};
  }

  unsigned commonLen = base.size();

  for (size_t i = 1; i < paths.size(); ++i) {
    auto cur = getModulePath(*paths[i]);
    unsigned len = std::min<unsigned>(commonLen, cur.size());
    unsigned j = 0;
    while (j < len && base[j] == cur[j])
      ++j;
    commonLen = j;
  }

  if (commonLen == 0) {
    llvm::errs() << "[WARN] No common module found for paths:\n";
    for (size_t i = 0; i < paths.size(); ++i) {
      llvm::errs() << "  Path " << i << ": ";
      for (auto mod : getModulePath(*paths[i]))
        llvm::errs() << mod.getModuleName() << " ";
      llvm::errs() << "annoPathValue: " << *(paths[i]) << " ";
      llvm::errs() << "\n";
    }
    return {};
  }

  return base[commonLen - 1];
}

void CounterInserter::updateAnnoMap(
    std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap,
    circt::firrtl::InstanceOp oldInst, circt::firrtl::InstanceOp newInst) {

  for (auto &it : annoMap) {
    circt::firrtl::AnnoPathValue &apv = it.second;

    if (apv.isLocal())
      continue;

    for (auto &inst : apv.instances) {
      if (inst == oldInst)
        inst = newInst;
    }
  }
}

void CounterInserter::updatePortInsertCacheValues(
    circt::firrtl::InstanceOp oldInst, circt::firrtl::InstanceOp newInst) {
  unsigned oldNumResults = oldInst->getNumResults();
  unsigned newNumResults = newInst->getNumResults();
  unsigned n = std::min(oldNumResults, newNumResults);

  for (auto &it : portInsertCache) {
    mlir::Value &cachedValue = it.second;
    if (!cachedValue)
      continue;

    for (unsigned i = 0; i < n; ++i) {
      if (cachedValue == oldInst.getResult(i)) {
        cachedValue = newInst.getResult(i);
        break;
      }
    }
  }
}

std::unique_ptr<CounterInserter>
CounterInserter::create(mlir::Operation *module,
                        circt::firrtl::InstanceGraph &instanceGraph) {
  auto circuitOp = llvm::dyn_cast<circt::firrtl::CircuitOp>(module);
  if (!circuitOp) {
    llvm::errs() << "CounterInserter: module is not a CircuitOp\n";
    return nullptr;
  }

  return std::make_unique<CounterInserter>(module, instanceGraph);
}