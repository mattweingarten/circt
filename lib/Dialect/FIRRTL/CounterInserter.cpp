#include "circt/Dialect/FIRRTL/CounterInserter.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "counter-inserter"

using namespace circt;
using namespace circt::firrtl;

namespace {

struct LoweredExpr {
  enum Kind { Leaf, Constant, Not, And, Or, Xor, Eq } kind;

  mlir::Value leafValue;
  llvm::APInt constValue = llvm::APInt(1, 0);
  unsigned constWidth = 0;

  llvm::SmallVector<std::shared_ptr<LoweredExpr>, 2> children;

  static std::shared_ptr<LoweredExpr> makeLeaf(mlir::Value v) {
    auto out = std::make_shared<LoweredExpr>();
    out->kind = Leaf;
    out->leafValue = v;
    return out;
  }

  static std::shared_ptr<LoweredExpr> makeConst(unsigned width,
                                                const llvm::APInt &value) {
    auto out = std::make_shared<LoweredExpr>();
    out->kind = Constant;
    out->constWidth = width;
    out->constValue = value;
    return out;
  }

  static std::shared_ptr<LoweredExpr>
  makeNode(Kind kind, llvm::SmallVector<std::shared_ptr<LoweredExpr>, 2> kids) {
    auto out = std::make_shared<LoweredExpr>();
    out->kind = kind;
    out->children = std::move(kids);
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

static bool isZ3BoolLikeLeaf(const z3::expr &e) {
  if (!isZ3AtomicLeaf(e))
    return false;
  if (e.is_bool())
    return true;
  if (e.is_bv() && e.get_sort().bv_size() == 1)
    return true;
  return false;
}

static void collectReferencedLeaves(const z3::expr &e,
                                    llvm::SmallVectorImpl<z3::expr> &out,
                                    llvm::SmallPtrSetImpl<Z3_ast> &seen) {
  Z3_ast ast = e;
  if (!seen.insert(ast).second)
    return;

  if (isZ3BoolLikeLeaf(e)) {
    out.push_back(e);
    return;
  }

  for (unsigned i = 0; i < e.num_args(); ++i)
    collectReferencedLeaves(e.arg(i), out, seen);
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

static llvm::SmallVector<FModuleLike, 8>
getModulePath(CircuitOp circuit, const AnnoPathValue &apv) {
  llvm::SmallVector<FModuleLike, 8> mods;

  if (apv.instances.empty()) {
    auto m = apv.ref.getModule();
    if (m)
      mods.push_back(m);
    return mods;
  }

  auto firstParent = apv.instances.front()->getParentOfType<FModuleLike>();
  if (firstParent)
    mods.push_back(firstParent);

  for (auto inst : apv.instances) {
    auto child = circuit.lookupSymbol<FModuleLike>(inst.getModuleNameAttr());
    if (child)
      mods.push_back(child);
  }

  return mods;
}

static FModuleLike
findLowestCommonModule(CircuitOp circuit,
                       llvm::ArrayRef<const AnnoPathValue *> paths) {
  if (paths.empty())
    return {};

  llvm::SmallVector<FModuleLike, 8> base =
      getModulePath(circuit, *paths.front());
  if (base.empty())
    return {};

  unsigned commonLen = base.size();

  for (size_t i = 1; i < paths.size(); ++i) {
    auto cur = getModulePath(circuit, *paths[i]);
    unsigned len = std::min<unsigned>(commonLen, cur.size());
    unsigned j = 0;
    while (j < len && base[j] == cur[j])
      ++j;
    commonLen = j;
  }

  if (commonLen == 0)
    return {};

  return base[commonLen - 1];
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

static mlir::Value makeValueVisibleInModule(const AnnoPathValue &apv,
                                            mlir::Value value,
                                            FModuleLike targetModule,
                                            StringRef debugName) {
  auto sourceModule = apv.ref.getModule();
  if (!sourceModule || !value || !targetModule)
    return {};

  if (sourceModule == targetModule)
    return value;

  llvm::errs() << "[TODO] plumb signal '" << debugName << "' from module "
               << sourceModule.getModuleName() << " to module "
               << targetModule.getModuleName() << "\n";
  return {};
}

static std::shared_ptr<LoweredExpr>
planExprIntoModule(const z3::expr &e,
                   const llvm::DenseMap<Z3_ast, mlir::Value> &leafMap) {
  if (auto it = leafMap.find((Z3_ast)e); it != leafMap.end())
    return LoweredExpr::makeLeaf(it->second);

  auto isAtomicVar = [](const z3::expr &expr) {
    if (!expr.is_const() || expr.num_args() != 0)
      return false;
    auto dk = expr.decl().decl_kind();
    return dk != Z3_OP_TRUE && dk != Z3_OP_FALSE && !expr.is_numeral();
  };

  if (isAtomicVar(e)) {
    LLVM_DEBUG(llvm::dbgs() << "[WARN] Atomic Z3 leaf not found in leafMap: "
                            << e.to_string() << "\n");
    return nullptr;
  }

  if (isZ3BoolConst(e)) {
    bool bit = e.decl().decl_kind() == Z3_OP_TRUE;
    return LoweredExpr::makeConst(1, llvm::APInt(1, bit ? 1 : 0));
  }

  if (e.is_numeral()) {
    if (e.is_bool()) {
      bool bit = Z3_get_bool_value(e.ctx(), e) == Z3_L_TRUE;
      return LoweredExpr::makeConst(1, llvm::APInt(1, bit ? 1 : 0));
    }

    if (e.is_bv()) {
      unsigned width = e.get_sort().bv_size();
      if (width == 0)
        width = 1;

      uint64_t value = 0;
      if (!e.is_numeral_u64(value)) {
        llvm::errs()
            << "[WARN] Unsupported non-u64 BV numeral in counter expr: "
            << e.to_string() << "\n";
        return nullptr;
      }

      return LoweredExpr::makeConst(width, llvm::APInt(width, value));
    }

    llvm::errs() << "[WARN] Unsupported numeral sort in counter expr: "
                 << e.to_string() << "\n";
    return nullptr;
  }

  auto planChild = [&](unsigned i) {
    return planExprIntoModule(e.arg(i), leafMap);
  };

  switch (e.decl().decl_kind()) {
  case Z3_OP_NOT: {
    auto a = planChild(0);
    if (!a)
      return nullptr;
    return LoweredExpr::makeNode(LoweredExpr::Not, {a});
  }
  case Z3_OP_AND: {
    llvm::SmallVector<std::shared_ptr<LoweredExpr>, 2> kids;
    kids.reserve(e.num_args());
    for (unsigned i = 0; i < e.num_args(); ++i) {
      auto child = planChild(i);
      if (!child)
        return nullptr;
      kids.push_back(child);
    }
    return LoweredExpr::makeNode(LoweredExpr::And, std::move(kids));
  }
  case Z3_OP_OR: {
    llvm::SmallVector<std::shared_ptr<LoweredExpr>, 2> kids;
    kids.reserve(e.num_args());
    for (unsigned i = 0; i < e.num_args(); ++i) {
      auto child = planChild(i);
      if (!child)
        return nullptr;
      kids.push_back(child);
    }
    return LoweredExpr::makeNode(LoweredExpr::Or, std::move(kids));
  }
  case Z3_OP_XOR: {
    llvm::SmallVector<std::shared_ptr<LoweredExpr>, 2> kids;
    kids.reserve(e.num_args());
    for (unsigned i = 0; i < e.num_args(); ++i) {
      auto child = planChild(i);
      if (!child)
        return nullptr;
      kids.push_back(child);
    }
    return LoweredExpr::makeNode(LoweredExpr::Xor, std::move(kids));
  }
  case Z3_OP_EQ: {
    auto a = planChild(0);
    auto b = planChild(1);
    if (!a || !b)
      return nullptr;
    return LoweredExpr::makeNode(LoweredExpr::Eq, {a, b});
  }
  default:
    llvm::errs() << "[WARN] Unsupported Z3 op while lowering counter expr: "
                 << e.to_string() << "\n";
    return nullptr;
  }
}

static mlir::Value materializeExpr(mlir::OpBuilder &b, mlir::Location loc,
                                   const std::shared_ptr<LoweredExpr> &expr) {
  if (!expr)
    return {};

  switch (expr->kind) {
  case LoweredExpr::Leaf:
    return expr->leafValue;

  case LoweredExpr::Constant: {
    auto ty = UIntType::get(b.getContext(), expr->constWidth);
    auto attrTy = mlir::IntegerType::get(b.getContext(), expr->constWidth,
                                         mlir::IntegerType::Unsigned);
    auto attr = mlir::IntegerAttr::get(attrTy, expr->constValue);
    return b.create<ConstantOp>(loc, ty, attr);
  }

  case LoweredExpr::Not: {
    auto a = materializeExpr(b, loc, expr->children[0]);
    if (!a)
      return {};
    return b.create<NotPrimOp>(loc, a);
  }

  case LoweredExpr::And: {
    auto acc = materializeExpr(b, loc, expr->children[0]);
    if (!acc)
      return {};
    for (size_t i = 1; i < expr->children.size(); ++i) {
      auto rhs = materializeExpr(b, loc, expr->children[i]);
      if (!rhs)
        return {};
      acc = b.create<AndPrimOp>(loc, acc, rhs);
    }
    return acc;
  }

  case LoweredExpr::Or: {
    auto acc = materializeExpr(b, loc, expr->children[0]);
    if (!acc)
      return {};
    for (size_t i = 1; i < expr->children.size(); ++i) {
      auto rhs = materializeExpr(b, loc, expr->children[i]);
      if (!rhs)
        return {};
      acc = b.create<OrPrimOp>(loc, acc, rhs);
    }
    return acc;
  }

  case LoweredExpr::Xor: {
    auto acc = materializeExpr(b, loc, expr->children[0]);
    if (!acc)
      return {};
    for (size_t i = 1; i < expr->children.size(); ++i) {
      auto rhs = materializeExpr(b, loc, expr->children[i]);
      if (!rhs)
        return {};
      acc = b.create<XorPrimOp>(loc, acc, rhs);
    }
    return acc;
  }

  case LoweredExpr::Eq: {
    auto a = materializeExpr(b, loc, expr->children[0]);
    auto rhs = materializeExpr(b, loc, expr->children[1]);
    if (!a || !rhs)
      return {};
    return b.create<EQPrimOp>(loc, a, rhs);
  }
  }

  return {};
}

static mlir::Value createUInt1Const(mlir::OpBuilder &b, mlir::Location loc,
                                    bool bit) {
  auto ty = UIntType::get(b.getContext(), 1);
  auto attrTy =
      mlir::IntegerType::get(b.getContext(), 1, mlir::IntegerType::Unsigned);
  auto attr = mlir::IntegerAttr::get(attrTy, bit ? 1 : 0);
  return b.create<ConstantOp>(loc, ty, attr);
}

static mlir::Value buildNamedCounterWire(FModuleLike module, mlir::Location loc,
                                         StringRef counterName,
                                         StringRef exprNodeName) {
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
  auto exprNodeNameAttr = b.getStringAttr(exprNodeName);

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
  mlir::Value wiredCond = wireOp.getResult();

  auto zero = createUInt1Const(b, loc, false);

  mlir::OperationState nodeState(loc,
                                 circt::firrtl::NodeOp::getOperationName());
  nodeState.addOperands(zero);
  nodeState.addTypes(zero.getType());
  nodeState.addAttribute("name", exprNodeNameAttr);
  nodeState.addAttribute("nameKind", nameKindAttr);
  nodeState.addAttribute("annotations", emptyAnnos);

  auto *nodeRaw = b.create(nodeState);
  auto nodeOp = llvm::cast<circt::firrtl::NodeOp>(nodeRaw);
  mlir::Value namedZero = nodeOp.getResult();

  b.create<ConnectOp>(loc, wiredCond, namedZero);
  return wiredCond;
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

  b.create<ConnectOp>(loc, destWire, v);
  return true;
}

} // namespace

void CounterInserter::insertPerfCounters(
    z3::expr counterExpr, const std::string &counterName,
    const std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap) {

  auto circuitOp = llvm::dyn_cast<circt::firrtl::CircuitOp>(module);
  if (!circuitOp) {
    llvm::errs() << "[ERROR] CounterInserter: module is not a CircuitOp\n";
    return;
  }

  llvm::SmallVector<z3::expr, 16> leaves;
  llvm::SmallPtrSet<Z3_ast, 16> seen;
  collectReferencedLeaves(counterExpr, leaves, seen);

  if (leaves.empty()) {
    llvm::errs() << "CounterInserter: no referenced leaves in expression\n";
    return;
  }

  llvm::SmallVector<const AnnoPathValue *, 16> annoPaths;
  annoPaths.reserve(leaves.size());

  for (const auto &leaf : leaves) {
    std::string name = leaf.decl().name().str();

    auto it = annoMap.find(name);
    if (it == annoMap.end()) {
      llvm::errs() << "[WARNING] CounterInserter: missing annoMap entry for '"
                   << name << "'\n";
      return;
    }

    if (!isZ3BoolLikeLeaf(leaf)) {
      llvm::errs() << "[WARNING] CounterInserter: non-bool leaf '" << name
                   << "' in expr: " << leaf.to_string() << "\n";
      return;
    }

    const AnnoPathValue &apv = it->second;
    if (!apv.ref) {
      llvm::errs() << "CounterInserter: null annotation target for '" << name
                   << "'\n";
      return;
    }

    FIRRTLType ty = apv.ref.getType();
    if (!isBooleanLikeTargetType(ty)) {
      llvm::errs() << "CounterInserter: target '" << name
                   << "' is not width-1 / bool-like FIRRTL type\n";
      return;
    }

    if (!apv.ref.getModule()) {
      llvm::errs() << "CounterInserter: target '" << name
                   << "' has no containing module\n";
      return;
    }

    annoPaths.push_back(&apv);
  }

  FModuleLike lcaModule = findLowestCommonModule(circuitOp, annoPaths);
  if (!lcaModule) {
    llvm::errs() << "CounterInserter: failed to compute common module for '"
                 << counterName << "'\n";
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "CounterInserter: chosen insertion module: "
                          << lcaModule.getModuleName() << "\n");

  llvm::DenseMap<Z3_ast, mlir::Value> leafMap;

  for (const auto &leaf : leaves) {
    std::string name = leaf.decl().name().str();
    const auto &apv = annoMap.at(name);

    mlir::Value sourceValue = getValueFromAnnoPathValue(apv);
    if (!sourceValue) {
      llvm::errs()
          << "[ERROR] CounterInserter: could not recover mlir::Value for '"
          << name << "' from AnnoPathValue\n";
      return;
    }

    mlir::Value visibleValue =
        makeValueVisibleInModule(apv, sourceValue, lcaModule, name);
    if (!visibleValue) {
      llvm::errs() << "CounterInserter: failed to make '" << name
                   << "' visible in module " << lcaModule.getModuleName()
                   << "\n";
      return;
    }

    leafMap[(Z3_ast)leaf] = visibleValue;
  }

  auto lcaFModule = llvm::dyn_cast<FModuleOp>(lcaModule.getOperation());
  if (!lcaFModule) {
    llvm::errs() << "CounterInserter: insertion module is not an FModuleOp\n";
    return;
  }

  mlir::Block *body = lcaFModule.getBodyBlock();
  if (!body) {
    llvm::errs() << "CounterInserter: insertion module has no body block\n";
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
    llvm::errs() << "CounterInserter: could not find visible clock in module "
                 << lcaModule.getModuleName() << "\n";
    return;
  }

  mlir::Value wiredCond = buildNamedCounterWire(lcaModule, loc, counterName,
                                                counterName + "_expr");
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

  mlir::OpBuilder perfBuilder(body, body->end());
  auto wireNameAttr = perfBuilder.getStringAttr(counterName);
  auto descAttr = perfBuilder.getStringAttr(counterExpr.to_string());

  perfBuilder.create<circt::perf::PerfCounterOp>(
      loc, wiredCond, clockReset.clock.value, clockReset.reset.value,
      wireNameAttr, descAttr);

  LLVM_DEBUG({
    llvm::dbgs() << "CounterInserter: inserted counter '" << counterName
                 << "' in module " << lcaModule.getModuleName()
                 << " using clock port '" << clockReset.clock.portName << "'";
    if (clockReset.reset.value)
      llvm::dbgs() << " and reset port '" << clockReset.reset.portName << "'";
    llvm::dbgs() << "\n";
  });
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