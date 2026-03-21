#include "circt/Dialect/FIRRTL/CounterInserter.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::firrtl;

namespace {

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
  if (auto uintTy = dyn_cast<UIntType>(type))
    return uintTy.getWidthOrSentinel() == 1;
  if (type.isa<ResetType, AsyncResetType>())
    return true;
  return false;
}

/// Return the module path from circuit root down to the module containing
/// `apv`. Example:
///   top -> A -> B
/// represented by instances [top->A inst, A->B inst]
/// then this returns [top, A, B]
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

/// Lowest common module from AnnoPathValue.instance chains.
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
/// Try to recover the targeted SSA value from the annotation target.
/// This assumes the target is an op target with exactly one result or a block
/// arg. You may need to specialize this for your exact annotations.
static mlir::Value getValueFromAnnoPathValue(const AnnoPathValue &apv) {
  if (auto opRef = apv.ref.dyn_cast<OpAnnoTarget>()) {
    Operation *op = opRef.getOp();
    if (!op)
      return {};

    if (op->getNumResults() == 1)
      return op->getResult(0);

    // If your annotations target named wires/nodes/regs with one result, this
    // is enough. Otherwise this needs refinement.
    return {};
  }

  return {};
}

/// Insert a new input port on `module`.
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
/// Placeholder for actual cross-module plumbing.
/// Structural port insertion through instance chains is the hard part and is
/// left as a TODO, but this helper centralizes that work.
static mlir::Value makeValueVisibleInModule(const AnnoPathValue &apv,
                                            mlir::Value value,
                                            FModuleLike targetModule,
                                            StringRef debugName) {
  auto sourceModule = apv.ref.getModule();
  if (!sourceModule || !value || !targetModule)
    return {};

  if (sourceModule == targetModule)
    return value;

  llvm::errs() << "TODO: plumb signal '" << debugName << "' from module "
               << sourceModule.getModuleName() << " to module "
               << targetModule.getModuleName()
               << " using ports along path: " << apv << "\n";

  return {};
}

static mlir::Value
emitExprIntoModule(mlir::OpBuilder &b, mlir::Location loc, const z3::expr &e,
                   const llvm::DenseMap<Z3_ast, mlir::Value> &leafMap) {
  if (auto it = leafMap.find((Z3_ast)e); it != leafMap.end())
    return it->second;

  if (isZ3BoolConst(e)) {
    bool bit = e.decl().decl_kind() == Z3_OP_TRUE;
    auto ty = UIntType::get(b.getContext(), 1);
    auto attr = b.getIntegerAttr(b.getIntegerType(1), bit ? 1 : 0);
    return b.create<ConstantOp>(loc, ty, attr);
  }

  auto emitChild = [&](unsigned i) {
    return emitExprIntoModule(b, loc, e.arg(i), leafMap);
  };

  switch (e.decl().decl_kind()) {
  case Z3_OP_NOT: {
    auto a = emitChild(0);
    return b.create<NotPrimOp>(loc, a);
  }
  case Z3_OP_AND: {
    mlir::Value acc = emitChild(0);
    for (unsigned i = 1; i < e.num_args(); ++i)
      acc = b.create<AndPrimOp>(loc, acc, emitChild(i));
    return acc;
  }
  case Z3_OP_OR: {
    mlir::Value acc = emitChild(0);
    for (unsigned i = 1; i < e.num_args(); ++i)
      acc = b.create<OrPrimOp>(loc, acc, emitChild(i));
    return acc;
  }
  case Z3_OP_XOR: {
    mlir::Value acc = emitChild(0);
    for (unsigned i = 1; i < e.num_args(); ++i)
      acc = b.create<XorPrimOp>(loc, acc, emitChild(i));
    return acc;
  }
  case Z3_OP_EQ: {
    auto a = emitChild(0);
    auto bv = emitChild(1);
    return b.create<EQPrimOp>(loc, a, bv);
  }
  default:
    llvm::errs() << "Unsupported Z3 op while lowering counter expr: "
                 << e.to_string() << "\n";
    return {};
  }
}

} // namespace

void CounterInserter::insertPerfCounters(
    z3::expr counterExpr, const std::string &counterName,
    const std::map<std::string, circt::firrtl::AnnoPathValue> &annoMap) {

  // Step 1:
  // Go through the entire expression. For each variable, look up the value in
  // the annoMap and check that is is there, and that is a bitvector with width
  // one and a boolean.

  // Step 2:
  // If everything all the values are not in the same module, we need to create
  // additional ports to make them accessible. If they are not all in the same
  // module, we must create additional ports to get every signal to the lowest
  // common denominator module in the calltree (i.e in this case the instance
  // graph.)

  // Step 3: Create new operations so that we translate this z3 expression into
  // FIRRTL IR.

  // Step 4: Insert `prof dialect` operations.
  llvm::errs() << "Insert counter: " << counterName
               << " with expr: " << counterExpr.to_string() << "\n";

  auto circuitOp = llvm::dyn_cast<circt::firrtl::CircuitOp>(module);
  if (!circuitOp) {
    llvm::errs() << "CounterInserter: module is not a CircuitOp\n";
    return;
  }

  // Step 1: collect referenced leaves and validate against annoMap.
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
      llvm::errs() << "CounterInserter: missing annoMap entry for '" << name
                   << "'\n";
      return;
    }

    if (!isZ3BoolLikeLeaf(leaf)) {
      llvm::errs() << "CounterInserter: non-bool leaf '" << name
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

  // Step 2: find lowest common denominator module from AnnoPathValue.instances.
  FModuleLike lcaModule = findLowestCommonModule(circuitOp, annoPaths);
  if (!lcaModule) {
    llvm::errs() << "CounterInserter: failed to compute common module for '"
                 << counterName << "'\n";
    return;
  }

  llvm::errs() << "CounterInserter: chosen insertion module: "
               << lcaModule.getModuleName() << "\n";

  // Build leaf -> visible FIRRTL value mapping in the LCA module.
  llvm::DenseMap<Z3_ast, mlir::Value> leafMap;

  for (const auto &leaf : leaves) {
    std::string name = leaf.decl().name().str();
    const auto &apv = annoMap.at(name);

    mlir::Value sourceValue = getValueFromAnnoPathValue(apv);
    if (!sourceValue) {
      llvm::errs() << "CounterInserter: could not recover mlir::Value for '"
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

  // Step 3: lower the Z3 expression into FIRRTL IR in the LCA module.
  auto lcaFModule = dyn_cast<FModuleOp>(lcaModule.getOperation());
  if (!lcaFModule) {
    llvm::errs() << "CounterInserter: insertion module is not an FModuleOp\n";
    return;
  }

  mlir::Block *body = lcaFModule.getBodyBlock();
  if (!body) {
    llvm::errs() << "CounterInserter: insertion module has no body block\n";
    return;
  }

  mlir::OpBuilder b(body, body->end());
  mlir::Location loc = lcaModule.getLoc();

  mlir::Value cond = emitExprIntoModule(b, loc, counterExpr, leafMap);
  if (!cond) {
    llvm::errs() << "CounterInserter: failed to lower expression for counter '"
                 << counterName << "'\n";
    return;
  }

  llvm::errs()
      << "CounterInserter: successfully emitted counter condition for '"
      << counterName << "' in module " << lcaModule.getModuleName() << "\n";

  // Step 4: insert perf ops here.
  // TODO
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