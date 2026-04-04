#include "circt/Dialect/FIRRTL/TransformZ3Graph.h"

#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <iostream>
#include <stack>
#include <unordered_set>
#include <z3++.h>

using RewriteResult = Z3Graph::Canonicalizer::RewriteResult;
using RewriteRule = Z3Graph::Canonicalizer::RewriteRule;

#define DEBUG_TYPE "z3graph"

void Z3Graph::walkAST(const z3::expr &root,
                      const std::function<void(const z3::expr &)> &fn) {
  std::stack<z3::expr> work;
  std::unordered_set<z3::expr, ExprHash, ExprEq> visited;

  work.push(root);

  while (!work.empty()) {
    z3::expr e = work.top();
    work.pop();

    if (visited.find(e) != visited.end())
      continue;

    visited.insert(e);

    fn(e);

    Z3_ast_kind kind = Z3_get_ast_kind(e.ctx(), e);

    switch (kind) {
    case Z3_APP_AST:
      for (unsigned i = 0; i < e.num_args(); ++i)
        work.push(e.arg(i));
      break;

    case Z3_NUMERAL_AST:
    case Z3_VAR_AST:
    case Z3_QUANTIFIER_AST:
    default:
      break;
    }
  }
}
static void collectSubexpressions(const z3::expr &expr,
                                  llvm::SmallVectorImpl<z3::expr> &out,
                                  llvm::SmallPtrSet<Z3_ast, 16> &seen) {
  Z3_ast ast = expr;
  if (!seen.insert(ast).second)
    return;

  auto isBoolConst = [](const z3::expr &e) {
    auto dk = e.decl().decl_kind();
    return dk == Z3_OP_TRUE || dk == Z3_OP_FALSE;
  };

  auto isAtomicVar = [&](const z3::expr &e) {
    return e.is_const() && !e.is_numeral() && !isBoolConst(e) &&
           e.num_args() == 0;
  };

  auto isMaterializableSort = [&](const z3::expr &e) {
    return e.is_bool() || (e.is_bv() && e.get_sort().bv_size() == 1);
  };

  auto isInterestingStructuredExpr = [&](const z3::expr &e) {
    if (!isMaterializableSort(e))
      return false;
    if (isAtomicVar(e))
      return false;
    if (e.is_numeral())
      return false;
    return e.num_args() > 0;
  };

  if (expr.is_quantifier()) {
    llvm::errs() << "[Z3] quantifier expr in collectSubexpressions: "
                 << expr.to_string() << "\n";
    return;
  }

  if (!expr.is_app()) {
    llvm::errs() << "[Z3] non-app expr in collectSubexpressions: "
                 << expr.to_string() << "\n";
    return;
  }

  for (unsigned i = 0; i < expr.num_args(); ++i) {
    z3::expr child = expr.arg(i);

    if (isInterestingStructuredExpr(child)) {
      out.push_back(child);
      continue;
    }

    collectSubexpressions(child, out, seen);
  }
}

static z3::expr makeNamedExprForNode(z3::context &ctx, const std::string &name,
                                     const z3::expr &templ) {
  if (templ.is_bool())
    return ctx.bool_const(name.c_str());

  if (templ.is_bv())
    return ctx.bv_const(name.c_str(), templ.get_sort().bv_size());

  llvm::errs() << "unsupported sort for materialized node '" << name << "'\n";
  abort();
}

static z3::expr substituteExact(const z3::expr &root,
                                const z3::expr_vector &src,
                                const z3::expr_vector &dst) {
  z3::expr r = root;
  return r.substitute(src, dst);
}

void Z3Graph::buildZ3Graph() {
  nodes.clear();
  root = nullptr;

  //--------------------------------------------------------------------------
  // Step 1: Build each top-level node.
  //--------------------------------------------------------------------------
  for (size_t i = 0; i < nodeNames.size(); ++i) {
    const std::string &name = nodeNames[i];
    const z3::expr &expr = nodeExprs[i];

    const z3::expr canonicalized = canonicalizer.canonicalize(expr);

    bool isReg = registers.count(name) > 0;
    auto node = std::make_shared<Z3Graph::Node>(name, canonicalized, isReg);

    if (name == rootName)
      root = node;

    nodes.push_back(node);
  }

  //--------------------------------------------------------------------------
  // Step 2: Build a fast name -> node map.
  //--------------------------------------------------------------------------
  std::unordered_map<std::string, std::shared_ptr<Node>> nameToNode;
  for (const auto &node : nodes)
    nameToNode[node->name] = node;
  //--------------------------------------------------------------------------
  // Step 3: Materialize selected subexpression nodes before edge construction.
  // Repeatedly decompose until a fixed point is reached.
  //--------------------------------------------------------------------------
  llvm::DenseMap<Z3_ast, std::shared_ptr<Node>> exprToNode;
  llvm::SmallPtrSet<Node *, 32> nodeSet;
  unsigned materializedCounter = 0;

  for (const auto &n : nodes) {
    if (n)
      nodeSet.insert(n.get());
  }

  auto getOrCreateMaterializedNode =
      [&](const z3::expr &expr) -> std::shared_ptr<Node> {
    z3::expr canon = canonicalizer.canonicalize(expr);
    LLVM_DEBUG(llvm::dbgs()
               << "[CANONICALIZING] For expr: " << expr.to_string()
               << " got canonical form: " << canon.to_string() << "\n");
    Z3_ast ast = canon;

    auto it = exprToNode.find(ast);
    if (it != exprToNode.end())
      return it->second;

    std::string name = "mat_" + std::to_string(materializedCounter++);

    auto newNode = std::make_shared<Z3Graph::Node>(name, canon,
                                                   /*isRegister=*/false,
                                                   /*isQuasiRegister=*/false);

    exprToNode[ast] = newNode;
    return newNode;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<std::shared_ptr<Node>> nodesToAppend;

    // Only process nodes present at the start of this round.
    size_t roundNodeCount = nodes.size();

    for (size_t i = 0; i < roundNodeCount; ++i) {
      auto &node = nodes[i];
      if (!node || node->isRegister)
        continue;

      llvm::SmallVector<z3::expr, 8> subexprs;
      llvm::SmallPtrSet<Z3_ast, 16> seen;
      collectSubexpressions(node->expr, subexprs, seen);

      if (subexprs.empty())
        continue;

      z3::expr_vector src(node->expr.ctx());
      z3::expr_vector dst(node->expr.ctx());

      for (const auto &subexpr : subexprs) {
        auto matNode = getOrCreateMaterializedNode(subexpr);

        // llvm::errs() << "For node: " << node->name << " with expr "
        //              << node->expr.to_string()
        //              << " materializing subexpr: " << subexpr.to_string()
        //              << " as " << matNode->name << "\n";

        src.push_back(subexpr);
        dst.push_back(
            makeNamedExprForNode(node->expr.ctx(), matNode->name, subexpr));

        if (!nodeSet.contains(matNode.get())) {
          nodesToAppend.push_back(matNode);
          nodeSet.insert(matNode.get());
        }
      }

      z3::expr rewritten = node->expr.substitute(src, dst);
      rewritten = canonicalizer.canonicalize(rewritten);

      if (!z3::eq(rewritten, node->expr)) {
        node->expr = rewritten;
        changed = true;
      }
    }

    if (!nodesToAppend.empty()) {
      for (auto &n : nodesToAppend)
        nodes.push_back(n);
      changed = true;
    }
  }

  // Rebuild map now that nodes has grown.
  nameToNode.clear();
  for (const auto &node : nodes)
    nameToNode[node->name] = node;
  //--------------------------------------------------------------------------
  // Step 4: Connect edges.
  //
  // Prefer direct subexpression-node edges when a child subexpression has a
  // materialized node. Otherwise, fall back to symbol-reference edges.
  //--------------------------------------------------------------------------
  for (const auto &node : nodes) {
    node->edges.clear();
    node->isBackwardEdge.clear();

    llvm::SmallPtrSet<Node *, 8> seenChildren;

    auto addEdge = [&](const std::shared_ptr<Node> &child) {
      if (!child)
        return;
      if (!seenChildren.insert(child.get()).second)
        return;
      node->edges.push_back(child);
      node->isBackwardEdge.push_back(false);
    };

    // if (!node->expr.is_app()) {
    //   llvm::errs() << "[Z3] non-app expr in edge construction: "
    //                << node->expr.to_string() << "\n";
    //   continue;
    // }

    // First: connect to direct AST children if they have materialized nodes.
    for (unsigned i = 0; i < node->expr.num_args(); ++i) {
      z3::expr childExpr = canonicalizer.canonicalize(node->expr.arg(i));
      std::string childName = childExpr.to_string();

      auto it = nameToNode.find(childName);
      if (it != nameToNode.end())
        addEdge(it->second);
    }

    // Second: also connect symbol leaves referenced anywhere in the expression.
    walkAST(node->expr, [&](const z3::expr &e) {
      if (e.is_app() && e.num_args() == 0) {
        std::string symName = e.decl().name().str();

        auto it = nameToNode.find(symName);
        if (it != nameToNode.end())
          addEdge(it->second);
      }
    });

    assert(node->edges.size() == node->isBackwardEdge.size() &&
           "isBackwardEdge must stay parallel to edges");
  }

  //--------------------------------------------------------------------------
  // Step 5: Run DFS and mark back edges.
  //--------------------------------------------------------------------------
  enum class Color { White, Gray, Black };

  std::unordered_map<Node *, Color> color;
  for (const auto &node : nodes)
    color[node.get()] = Color::White;

  std::function<void(const std::shared_ptr<Node> &)> dfs =
      [&](const std::shared_ptr<Node> &node) {
        color[node.get()] = Color::Gray;

        for (size_t i = 0; i < node->edges.size(); ++i) {
          const auto &child = node->edges[i];
          Node *childRaw = child.get();

          if (color[childRaw] == Color::Gray)
            node->isBackwardEdge[i] = true;

          if (color[childRaw] == Color::White)
            dfs(child);
        }

        color[node.get()] = Color::Black;
      };

  for (const auto &node : nodes) {
    if (color[node.get()] == Color::White)
      dfs(node);
  }
}

void Z3Graph::printFullGraphDot(llvm::raw_ostream &os) const {
  os << "digraph Z3Graph {\n";

  std::unordered_set<std::string> nodeSet;
  for (const auto &n : nodes)
    nodeSet.insert(n->name);

  for (const auto &node : nodes) {
    os << "  \"" << node->name << "\" [label=\"";

    os << node->name << " : " << node->expr.get_sort().to_string();

    if (node->name == rootName || (!node->isRegister && !node->isQuasiRegister))
      os << "\\n" << node->expr.to_string();

    os << "\"";

    if (node->isRegister)
      os << ", style=filled, fillcolor=lightcoral";

    if (node->isQuasiRegister)
      os << ", style=filled, fillcolor=lightblue";

    os << "];\n";

    for (const auto &edge : node->edges) {
      if (!edge || !nodeSet.count(edge->name))
        continue;

      os << "  \"" << node->name << "\" -> \"" << edge->name << "\";\n";
    }
  }

  os << "}\n";
}

z3::expr Z3Graph::buildInlinedExpr() const {
  // Find the root node by name.
  std::shared_ptr<Node> root = nullptr;
  for (const auto &n : nodes) {
    if (n && n->name == rootName) {
      root = n;
      break;
    }
  }

  assert(root && "rootName not found in graph");

  std::unordered_map<const Node *, z3::expr> memo;
  std::unordered_set<const Node *> active;

  auto makeSymbol = [&](const std::shared_ptr<Node> &n) -> z3::expr {
    return ctx.constant(n->name.c_str(), n->expr.get_sort());
  };

  std::function<z3::expr(const std::shared_ptr<Node> &, bool)> expandNode =
      [&](const std::shared_ptr<Node> &node, bool isRoot) -> z3::expr {
    assert(node && "expandNode got null node");

    // Root is special: even though it is always a register, we still expand it.
    // Any other register/quasi-register is a stopping point.
    if (!isRoot && (node->isRegister || node->isQuasiRegister))
      return makeSymbol(node);

    auto memoIt = memo.find(node.get());
    if (memoIt != memo.end())
      return memoIt->second;

    // Break cycles conservatively.
    if (active.count(node.get()))
      return makeSymbol(node);

    active.insert(node.get());

    z3::expr result = node->expr;
    for (size_t i = 0; i < node->edges.size(); ++i) {
      const auto &child = node->edges[i];
      if (!child)
        continue;

      bool backward =
          (i < node->isBackwardEdge.size()) && node->isBackwardEdge[i];

      z3::expr childSym = makeSymbol(child);
      z3::expr replacement =
          backward ? childSym : expandNode(child, /*isRoot=*/false);

      z3::expr_vector from(ctx);
      z3::expr_vector to(ctx);

      from.push_back(childSym);
      to.push_back(replacement);

      result = result.substitute(from, to);
    }

    active.erase(node.get());
    memo.emplace(node.get(), result);
    return result;
  };

  return expandNode(root, /*isRoot=*/true);
}

void Z3Graph::printSubgraphDot(llvm::raw_ostream &os,
                               const std::string nodeName) const {
  os << "digraph Z3Subgraph {\n";
  os << "  rankdir=TB;\n";

  // Find starting node
  std::shared_ptr<Node> start = nullptr;
  for (const auto &n : nodes) {
    if (n->name == nodeName) {
      start = n;
      break;
    }
  }

  if (!start) {
    os << "}\n";
    return;
  }

  // Force root node to top
  os << "  { rank=min; \"" << start->name << "\" }\n";

  std::unordered_set<std::string> visited;
  std::unordered_set<std::string> emittedEdges;

  std::function<void(const std::shared_ptr<Node> &, bool)> dfs =
      [&](const std::shared_ptr<Node> &node, bool isRoot) {
        if (!node || visited.count(node->name))
          return;

        visited.insert(node->name);

        os << "  \"" << node->name << "\" [label=\"";

        os << node->name << " : " << node->expr.get_sort().to_string();

        if (!node->isRegister || isRoot)
          os << "\\n" << node->expr.to_string();

        os << "\"";

        if (node->isRegister)
          os << ", style=filled, fillcolor=lightcoral";

        os << "];\n";

        if (node->isRegister && !isRoot)
          return;

        for (const auto &child : node->edges) {
          std::string edgeKey = node->name + "->" + child->name;

          if (!emittedEdges.count(edgeKey)) {
            os << "  \"" << node->name << "\" -> \"" << child->name << "\";\n";
            emittedEdges.insert(edgeKey);
          }

          dfs(child, false);
        }
      };

  dfs(start, true);

  os << "}\n";
}
using RewriteResult = std::optional<z3::expr>;

// static z3::expr rebuildAppWithArgs(const z3::expr &e,
//                                    const z3::expr_vector &args) {
//   z3::context &ctx = e.ctx();

//   if (!e.is_app()) {
//     llvm::errs()
//                << "[Z3] rebuildAppWithArgs got invalid/non-app expr\n";
//     return e;
//   }

//   std::vector<Z3_ast> rawArgs;
//   rawArgs.reserve(args.size());

//   for (unsigned i = 0; i < args.size(); ++i) {
//     z3::expr arg = args[i];
//     // if () {
//     //   llvm::errs()
//     //              << "[Z3] rebuildAppWithArgs got null arg at index " << i
//     //              << " for expr: " << e.to_string() << "\n";
//     //   // assert(false && "unexpected null arg in rebuildAppWithArgs");
//     //   return e;
//     // }
//     rawArgs.push_back(arg);
//   }

//   Z3_ast newAst = Z3_mk_app(ctx, e.decl(), rawArgs.size(), rawArgs.data());
//   ctx.check_error();

//   return z3::expr(ctx, newAst);
// }

static z3::expr rebuildAppWithArgs(const z3::expr &e,
                                   const z3::expr_vector &args) {
  z3::context &ctx = e.ctx();

  std::vector<Z3_ast> rawArgs;
  rawArgs.reserve(args.size());

  for (unsigned i = 0; i < args.size(); ++i) {
    rawArgs.push_back(args[i]);
  }

  Z3_ast newAst = Z3_mk_app(ctx, e.decl(), rawArgs.size(), rawArgs.data());
  return z3::expr(ctx, newAst);
}

Z3Graph::Canonicalizer::Canonicalizer(z3::context &ctx) : ctx(ctx) {

  auto isOp = [](const z3::expr &e, Z3_decl_kind k, unsigned arity) -> bool {
    return e.is_app() && e.decl().decl_kind() == k && e.num_args() == arity;
  };

  auto getBV1Numeral = [](const z3::expr &e, uint64_t &out) -> bool {
    if (!e.is_bv())
      return false;
    if (e.get_sort().bv_size() != 1)
      return false;
    if (!e.is_numeral())
      return false;
    return Z3_get_numeral_uint64(e.ctx(), e, &out);
  };

  auto isBoolTrue = [&](const z3::expr &e) -> bool {
    return e.is_bool() && e.is_app() && e.decl().decl_kind() == Z3_OP_TRUE;
  };

  auto isBoolFalse = [&](const z3::expr &e) -> bool {
    return e.is_bool() && e.is_app() && e.decl().decl_kind() == Z3_OP_FALSE;
  };

  // (ite c x x) -> x
  addRule([&](const z3::expr &e) -> RewriteResult {
    if (!isOp(e, Z3_OP_ITE, 3))
      return std::nullopt;

    z3::expr t = e.arg(1);
    z3::expr f = e.arg(2);

    if (z3::eq(t, f))
      return t;

    return std::nullopt;
  });

  // (not (not x)) -> x
  addRule([&](const z3::expr &e) -> RewriteResult {
    if (!isOp(e, Z3_OP_NOT, 1))
      return std::nullopt;

    z3::expr x = e.arg(0);
    if (isOp(x, Z3_OP_NOT, 1))
      return x.arg(0);

    return std::nullopt;
  });

  // (= x x) -> true
  addRule([&](const z3::expr &e) -> RewriteResult {
    if (!isOp(e, Z3_OP_EQ, 2))
      return std::nullopt;

    if (z3::eq(e.arg(0), e.arg(1)))
      return ctx.bool_val(true);

    return std::nullopt;
  });

  // Bool-only:
  // (ite c false true) -> (not c)
  addRule([&](const z3::expr &e) -> RewriteResult {
    if (!isOp(e, Z3_OP_ITE, 3))
      return std::nullopt;

    z3::expr cond = e.arg(0);
    z3::expr t = e.arg(1);
    z3::expr f = e.arg(2);

    if (!cond.is_bool() || !t.is_bool() || !f.is_bool())
      return std::nullopt;

    if (isBoolFalse(t) && isBoolTrue(f))
      return !cond;

    return std::nullopt;
  });

  // Bool-only:
  // (ite c true false) -> c
  addRule([&](const z3::expr &e) -> RewriteResult {
    if (!isOp(e, Z3_OP_ITE, 3))
      return std::nullopt;

    z3::expr cond = e.arg(0);
    z3::expr t = e.arg(1);
    z3::expr f = e.arg(2);

    if (!cond.is_bool() || !t.is_bool() || !f.is_bool())
      return std::nullopt;

    if (isBoolTrue(t) && isBoolFalse(f))
      return cond;

    return std::nullopt;
  });

  // (= (ite c #b0 #b1) #b0) -> c
  // (= (ite c #b0 #b1) #b1) -> !c
  // (= (ite c #b1 #b0) #b0) -> !c
  // (= (ite c #b1 #b0) #b1) -> c
  addRule([&](const z3::expr &e) -> RewriteResult {
    if (!isOp(e, Z3_OP_EQ, 2))
      return std::nullopt;

    z3::expr lhs = e.arg(0);
    z3::expr rhs = e.arg(1);

    if (!isOp(lhs, Z3_OP_ITE, 3))
      return std::nullopt;

    z3::expr cond = lhs.arg(0);
    z3::expr t = lhs.arg(1);
    z3::expr f = lhs.arg(2);

    if (!cond.is_bool())
      return std::nullopt;

    uint64_t tv, fv, rv;
    if (!getBV1Numeral(t, tv) || !getBV1Numeral(f, fv) ||
        !getBV1Numeral(rhs, rv))
      return std::nullopt;

    // (= (ite c #b0 #b1) #b0) -> c
    if (tv == 0 && fv == 1 && rv == 0)
      return cond;

    // (= (ite c #b0 #b1) #b1) -> !c
    if (tv == 0 && fv == 1 && rv == 1)
      return !cond;

    // (= (ite c #b1 #b0) #b0) -> !c
    if (tv == 1 && fv == 0 && rv == 0)
      return !cond;

    // (= (ite c #b1 #b0) #b1) -> c
    if (tv == 1 && fv == 0 && rv == 1)
      return cond;

    return std::nullopt;
  });

  // Symmetric version: constant on LHS
  // (= #b0 (ite c #b0 #b1)) -> c, etc.
  addRule([&](const z3::expr &e) -> RewriteResult {
    if (!isOp(e, Z3_OP_EQ, 2))
      return std::nullopt;

    z3::expr lhs = e.arg(0);
    z3::expr rhs = e.arg(1);

    if (!isOp(rhs, Z3_OP_ITE, 3))
      return std::nullopt;

    z3::expr cond = rhs.arg(0);
    z3::expr t = rhs.arg(1);
    z3::expr f = rhs.arg(2);

    if (!cond.is_bool())
      return std::nullopt;

    uint64_t lv, tv, fv;
    if (!getBV1Numeral(lhs, lv) || !getBV1Numeral(t, tv) ||
        !getBV1Numeral(f, fv))
      return std::nullopt;

    // (= #b0 (ite c #b0 #b1)) -> c
    if (lv == 0 && tv == 0 && fv == 1)
      return cond;

    // (= #b1 (ite c #b0 #b1)) -> !c
    if (lv == 1 && tv == 0 && fv == 1)
      return !cond;

    // (= #b0 (ite c #b1 #b0)) -> !c
    if (lv == 0 && tv == 1 && fv == 0)
      return !cond;

    // (= #b1 (ite c #b1 #b0)) -> c
    if (lv == 1 && tv == 1 && fv == 0)
      return cond;

    return std::nullopt;
  });

  // (= (ite c x #b0) #b1) -> (and c (= x #b1))
  // (= (ite c #b0 x) #b1) -> (and (!c) (= x #b1))
  addRule([&](const z3::expr &e) -> RewriteResult {
    if (!isOp(e, Z3_OP_EQ, 2))
      return std::nullopt;

    z3::expr lhs = e.arg(0);
    z3::expr rhs = e.arg(1);

    if (!isOp(lhs, Z3_OP_ITE, 3))
      return std::nullopt;

    z3::expr cond = lhs.arg(0);
    z3::expr t = lhs.arg(1);
    z3::expr f = lhs.arg(2);

    if (!cond.is_bool())
      return std::nullopt;
    if (!rhs.is_bv() || rhs.get_sort().bv_size() != 1)
      return std::nullopt;

    uint64_t rv, tv, fv;
    bool tIsNum = getBV1Numeral(t, tv);
    bool fIsNum = getBV1Numeral(f, fv);
    if (!getBV1Numeral(rhs, rv))
      return std::nullopt;

    if (rv != 1)
      return std::nullopt;

    // (= (ite c x #b0) #b1) -> and(c, (= x #b1))
    if (fIsNum && fv == 0)
      return cond && (t == ctx.bv_val(1, 1));

    // (= (ite c #b0 x) #b1) -> and(!c, (= x #b1))
    if (tIsNum && tv == 0)
      return (!cond) && (f == ctx.bv_val(1, 1));

    return std::nullopt;
  });

  // (ite c x y) -> (or (and c x) (and (!c) y))
  addRule([&](const z3::expr &e) -> RewriteResult {
    if (!isOp(e, Z3_OP_ITE, 3))
      return std::nullopt;

    z3::expr cond = e.arg(0);
    z3::expr t = e.arg(1);
    z3::expr f = e.arg(2);

    if (!cond.is_bool() || !t.is_bool() || !f.is_bool())
      return std::nullopt;

    return (cond && t) || ((!cond) && f);
  });

  // (= a b) -> (or (and a b) (and (!a) (!b)))
  addRule([&](const z3::expr &e) -> RewriteResult {
    if (!isOp(e, Z3_OP_EQ, 2))
      return std::nullopt;

    z3::expr a = e.arg(0);
    z3::expr b = e.arg(1);

    // Only apply to boolean equality
    if (!a.is_bool() || !b.is_bool() || !a.is_bv() ||
        a.get_sort().bv_size() != 1 || !b.is_bv() ||
        b.get_sort().bv_size() != 1)
      return std::nullopt;

    return (a && b) || ((!a) && (!b));
  });

  // addRule([&](const z3::expr &e) -> RewriteResult {
  //   if (!isOp(e, Z3_OP_ITE, 3))
  //     return std::nullopt;

  //   z3::expr cond = e.arg(0);
  //   z3::expr t = e.arg(1);
  //   z3::expr f = e.arg(2);

  //   if (!cond.is_bool())
  //     return std::nullopt;

  //   auto isBV1Val = [](const z3::expr &x, unsigned expected) -> bool {
  //     if (!x.is_bv() || x.get_sort().bv_size() != 1 || !x.is_numeral())
  //       return false;
  //     uint64_t v = 0;
  //     if (!Z3_get_numeral_uint64(x.ctx(), x, &v))
  //       return false;
  //     return v == expected;
  //   };

  //   // (ite c #b1 #b0) -> (= c true)-like canonical Bool form => c
  //   if (isBV1Val(t, 1) && isBV1Val(f, 0))
  //     return cond;

  //   // (ite c #b0 #b1) -> !c
  //   if (isBV1Val(t, 0) && isBV1Val(f, 1))
  //     return !cond;

  //   return std::nullopt;
  // });
}

void Z3Graph::Canonicalizer::addRule(
    std::function<RewriteResult(const z3::expr &)> rule) {
  rules.push_back(rule);
}

z3::expr Z3Graph::Canonicalizer::canonicalize(const z3::expr &e) {
  return canonicalizeSubtreeToFixpoint(e);
}

z3::expr
Z3Graph::Canonicalizer::canonicalizeSubtreeToFixpoint(const z3::expr &e) {
  z3::expr cur = e;

  while (true) {
    // First canonicalize all children recursively.
    z3::expr rebuilt = rebuildWithCanonicalChildren(cur);

    // Then try to rewrite this node itself to a fixpoint.
    z3::expr rewritten = applyLocalRulesToFixpoint(rebuilt);

    // If nothing changed in the whole subtree, we are done.
    if (z3::eq(cur, rewritten))
      return rewritten;

    cur = rewritten;
  }
}

z3::expr
Z3Graph::Canonicalizer::rebuildWithCanonicalChildren(const z3::expr &e) {
  if (!e.is_app())
    return e;

  unsigned n = e.num_args();
  if (n == 0)
    return e;

  z3::expr_vector newArgs(ctx);
  bool changed = false;

  for (unsigned i = 0; i < n; ++i) {
    z3::expr oldArg = e.arg(i);
    z3::expr newArg = canonicalizeSubtreeToFixpoint(oldArg);

    if (!z3::eq(oldArg, newArg))
      changed = true;

    newArgs.push_back(newArg);
  }

  if (!changed)
    return e;

  return rebuildAppWithArgs(e, newArgs);
}

z3::expr Z3Graph::Canonicalizer::applyLocalRulesToFixpoint(const z3::expr &e) {
  z3::expr cur = e;

  while (true) {
    bool changed = false;

    for (const auto &rule : rules) {
      auto result = rule(cur);
      if (!result.has_value())
        continue;

      if (!z3::eq(cur, *result)) {
        cur = *result;
        changed = true;
        break;
      }
    }

    if (!changed)
      return cur;
  }
}

void Z3Graph::DominatorTree::printFullGraphDot(llvm::raw_ostream &os) const {
  os << "digraph DominatorTree {\n";

  for (const auto &node : nodes) {
    if (!node->graphNode)
      continue;

    const auto &g = node->graphNode;

    os << "  \"" << g->name << "\" [label=\"";
    os << g->name << " : " << g->expr.get_sort().to_string();

    if (!g->isRegister)
      os << "\\n" << g->expr.to_string();

    os << "\"";

    if (g->isRegister)
      os << ", style=filled, fillcolor=lightcoral";

    os << "];\n";

    for (const auto &child : node->children) {
      if (!child->graphNode)
        continue;

      os << "  \"" << g->name << "\" -> \"" << child->graphNode->name
         << "\";\n";
    }
  }

  os << "}\n";
}

void Z3Graph::DominatorTree::printSubgraphDot(
    llvm::raw_ostream &os, const Z3Graph &graph,
    const std::string nodeName) const {
  auto doms = state.doms;
  os << "digraph DominatorTreeSubgraph {\n";
  os << "  rankdir=TB;\n";

  // --------------------------------------------------------------------------
  // Step 1: Find the start node in the original Z3Graph
  // --------------------------------------------------------------------------
  std::shared_ptr<Z3Graph::Node> start = nullptr;
  for (const auto &n : graph.nodes) {
    if (n->name == nodeName) {
      start = n;
      break;
    }
  }

  if (!start) {
    os << "}\n";
    return;
  }

  // Force the requested root node to the top
  os << "  { rank=min; \"" << start->name << "\" }\n";

  // --------------------------------------------------------------------------
  // Step 2: Recompute the original subgraph node set using the same traversal
  // logic as Z3Graph::printSubgraphDot()
  // --------------------------------------------------------------------------
  std::unordered_set<std::string> subgraphNodes;

  std::function<void(const std::shared_ptr<Z3Graph::Node> &, bool)> collect =
      [&](const std::shared_ptr<Z3Graph::Node> &node, bool isRoot) {
        if (!node || subgraphNodes.count(node->name))
          return;

        subgraphNodes.insert(node->name);

        // Stop recursion at registers unless this is the starting node
        if (node->isRegister && !isRoot)
          return;

        for (const auto &child : node->edges)
          collect(child, false);
      };

  collect(start, true);

  // --------------------------------------------------------------------------
  // Step 3: Emit nodes in the selected subgraph
  // --------------------------------------------------------------------------
  std::unordered_set<std::string> emittedNodes;
  for (const auto &domNode : nodes) {
    if (!domNode || !domNode->graphNode)
      continue;

    const auto &g = domNode->graphNode;
    const auto &name = g->name;
    if (!subgraphNodes.count(name))
      continue;

    if (emittedNodes.insert(name).second) {
      os << "  \"" << name << "\" [label=\"";
      os << name << " : " << g->expr.get_sort().to_string();
      os << "\\n" << g->expr.to_string();
      os << "\"";

      if (g->isRegister)
        os << ", style=filled, fillcolor=lightcoral";

      os << "];\n";
    }
  }

  // --------------------------------------------------------------------------
  // Step 4: Emit dominance edges using the doms relation
  //
  // If A is in doms[B], then A dominates B.
  // We restrict to nodes inside the selected subgraph.
  // --------------------------------------------------------------------------
  std::unordered_set<std::string> emittedEdges;

  for (const auto &bNode : nodes) {
    if (!bNode || !bNode->graphNode)
      continue;

    auto b = bNode->graphNode;
    const auto &bName = b->name;

    if (!subgraphNodes.count(bName))
      continue;

    auto domIt = doms.find(b);
    if (domIt == doms.end())
      continue;

    for (const auto &a : domIt->second) {
      if (!a)
        continue;
      if (a == b)
        continue;
      if (!subgraphNodes.count(a->name))
        continue;

      std::string edgeKey = a->name + "->" + bName;
      if (emittedEdges.insert(edgeKey).second)
        os << "  \"" << a->name << "\" -> \"" << bName << "\";\n";
    }
  }

  os << "}\n";
}
std::unique_ptr<Z3Graph::DominatorTree>
Z3Graph::DominatorTree::build(const Z3Graph &graph) {

  auto tree = std::make_unique<DominatorTree>();

  auto entry = graph.root;

  if (!entry) {
    llvm::errs()
        << "Warning: No entry node found for dominator tree construction\n";
    return tree;
  }

  tree->nodes.reserve(graph.nodes.size());

  for (auto &node : graph.nodes) {
    auto new_node = std::make_shared<Node>(Node{node, {}});

    tree->nodes.push_back(new_node);
    tree->nameToNode[node->name] = new_node;
    tree->state.preds[node] = {};

    if (node == entry) {
      tree->state.doms[node] = {node};
    } else {
      tree->state.doms[node] = std::set<std::shared_ptr<Z3Graph::Node>>(
          graph.nodes.begin(), graph.nodes.end());
    }
  }

  // Build predecessor map
  for (auto &node : graph.nodes) {
    for (auto &succ : node->edges) {
      tree->state.preds[succ].insert(node);
    }
  }

  // Dom(node) := {node} U (intersection of Dom(pred) for all pred in
  // predecessors(node))

  bool changed = true;
  while (changed) {
    changed = false;

    for (auto &node : graph.nodes) {
      if (node == entry)
        continue;

      std::set<std::shared_ptr<Z3Graph::Node>> newDom;

      if (tree->state.preds[node].empty()) {
        newDom = {node};
      } else {
        bool firstPred = true;

        for (auto &pred : tree->state.preds[node]) {
          if (firstPred) {
            newDom = tree->state.doms[pred];
            firstPred = false;
          } else {
            std::set<std::shared_ptr<Z3Graph::Node>> intersection;
            std::set_intersection(
                newDom.begin(), newDom.end(), tree->state.doms[pred].begin(),
                tree->state.doms[pred].end(),
                std::inserter(intersection, intersection.begin()));
            newDom = std::move(intersection);
          }
        }

        newDom.insert(node);
      }

      if (newDom != tree->state.doms[node]) {
        tree->state.doms[node] = std::move(newDom);
        changed = true;
      }
    }
  }

  // Debugging
  // for (auto node : tree->nodes) {
  //   auto doms = tree->state.doms[node->graphNode];
  //   llvm::errs() << node->graphNode->name << " <--";
  //   for (auto &dom : doms) {
  //     llvm::errs() << " " << dom->name;
  //   }
  //   llvm::errs() << "\n";
  // }

  // Now use doms to build the tree structure.
  for (auto &node : tree->nodes) {
    auto gnode = node->graphNode;

    // Entry has no immediate dominator.
    if (gnode == entry) {
      node->parent = nullptr;
      continue;
    }

    const auto &domSet = tree->state.doms[gnode];

    std::shared_ptr<Z3Graph::Node> immediateDom = nullptr;

    for (auto &candidate : domSet) {
      if (candidate == gnode)
        continue; // strict dominators only

      bool isImmediate = true;

      for (auto &other : domSet) {
        if (other == gnode || other == candidate)
          continue;

        // If candidate dominates other, then candidate is higher up the tree
        // and cannot be the immediate dominator.
        if (tree->state.doms[other].count(candidate)) {
          isImmediate = false;
          break;
        }
      }

      if (isImmediate) {
        immediateDom = candidate;
        break;
      }
    }

    if (!immediateDom) {
      node->parent = nullptr;
      continue;
    }

    auto parentIt = tree->nameToNode.find(immediateDom->name);
    if (parentIt == tree->nameToNode.end()) {
      node->parent = nullptr;
      continue;
    }

    node->parent = parentIt->second;
    parentIt->second->children.push_back(node);
  }

  return tree;
}

void Z3Graph::buildDominatorTree() {
  domTree = std::move(DominatorTree::build(*this));
}
std::unique_ptr<Z3Graph>
Z3Graph::createSubgraph(const std::string &rootNodeName) const {

  std::shared_ptr<Node> start = nullptr;
  for (const auto &n : nodes) {
    if (n->name == rootNodeName) {
      start = n;
      break;
    }
  }

  if (!start)
    return nullptr;

  std::vector<std::shared_ptr<Node>> origSubNodes;
  std::vector<std::string> subNames;
  std::vector<z3::expr> subExprs;
  std::set<std::string> subRegs;
  std::unordered_set<std::string> visited;

  std::function<void(const std::shared_ptr<Node> &, bool)> dfs =
      [&](const std::shared_ptr<Node> &node, bool isRoot) {
        if (!node || visited.count(node->name))
          return;

        visited.insert(node->name);

        origSubNodes.push_back(node);
        subNames.push_back(node->name);
        subExprs.push_back(node->expr);

        if (node->isRegister)
          subRegs.insert(node->name);

        if (node->isRegister && !isRoot)
          return;

        for (const auto &child : node->edges)
          dfs(child, false);
      };

  dfs(start, true);

  auto subgraph =
      std::make_unique<Z3Graph>(ctx, subNames, rootNodeName, subExprs, subRegs);

  if (!subgraph || !subgraph->valid)
    return subgraph;

  subgraph->nodes.clear();
  subgraph->root = nullptr;

  std::unordered_map<std::string, std::shared_ptr<Node>> newNodeMap;
  newNodeMap.reserve(origSubNodes.size());

  for (const auto &origNode : origSubNodes) {
    auto newNode = std::make_shared<Node>(origNode->name, origNode->expr,
                                          origNode->isRegister);
    newNode->isQuasiRegister = origNode->isQuasiRegister;
    newNode->edges.clear();
    newNode->isBackwardEdge.clear();
    subgraph->nodes.push_back(newNode);
    newNodeMap[newNode->name] = newNode;
  }

  auto rootIt = newNodeMap.find(rootNodeName);
  if (rootIt != newNodeMap.end())
    subgraph->root = rootIt->second;

  std::unordered_set<std::string> allowed(subNames.begin(), subNames.end());
  std::unordered_set<std::string> rebuilt;

  std::function<void(const std::shared_ptr<Node> &, bool)> rebuildEdges =
      [&](const std::shared_ptr<Node> &origNode, bool isRoot) {
        if (!origNode || !allowed.count(origNode->name))
          return;

        if (rebuilt.count(origNode->name))
          return;
        rebuilt.insert(origNode->name);

        auto newIt = newNodeMap.find(origNode->name);
        if (newIt == newNodeMap.end())
          return;

        auto newNode = newIt->second;

        if (origNode->isRegister && !isRoot)
          return;

        std::unordered_set<std::string> seenChildren;
        for (const auto &origChild : origNode->edges) {
          if (!origChild || !allowed.count(origChild->name))
            continue;

          auto childIt = newNodeMap.find(origChild->name);
          if (childIt == newNodeMap.end())
            continue;

          if (seenChildren.insert(origChild->name).second) {
            newNode->edges.push_back(childIt->second);
            newNode->isBackwardEdge.push_back(
                false); // fill now, recompute below
          }

          rebuildEdges(origChild, false);
        }
      };

  rebuildEdges(start, true);

  // Recompute backward edges for the rebuilt subgraph.
  {
    enum class Color { White, Gray, Black };

    std::unordered_map<Node *, Color> color;
    for (const auto &node : subgraph->nodes) {
      assert(node->edges.size() == node->isBackwardEdge.size() &&
             "edges and isBackwardEdge must stay parallel");
      color[node.get()] = Color::White;
    }

    std::function<void(const std::shared_ptr<Node> &)> markBackedges =
        [&](const std::shared_ptr<Node> &node) {
          color[node.get()] = Color::Gray;

          for (size_t i = 0; i < node->edges.size(); ++i) {
            const auto &child = node->edges[i];
            assert(i < node->isBackwardEdge.size() &&
                   "edges and isBackwardEdge out of sync");

            if (!child)
              continue;

            if (color[child.get()] == Color::Gray)
              node->isBackwardEdge[i] = true;

            if (color[child.get()] == Color::White)
              markBackedges(child);
          }

          color[node.get()] = Color::Black;
        };

    for (const auto &node : subgraph->nodes) {
      if (color[node.get()] == Color::White)
        markBackedges(node);
    }
  }

  // subgraph->domTree = std::move(DominatorTree::build(*subgraph));

  return subgraph;
}

static void collectDeepBitVectorSubexprs(const z3::expr &expr, int depth,
                                         llvm::SmallVectorImpl<z3::expr> &out,
                                         llvm::SmallPtrSet<Z3_ast, 16> &seen) {
  Z3_ast ast = expr;

  if (!seen.insert(ast).second)
    return;

  if (depth > 1 && expr.get_sort().is_bv()) {
    out.push_back(expr);
    return;
  }

  for (unsigned i = 0; i < expr.num_args(); ++i)
    collectDeepBitVectorSubexprs(expr.arg(i), depth + 1, out, seen);
}

std::shared_ptr<Z3Graph::Node>
Z3Graph::getOrCreateExprNode(const z3::expr &expr) {
  std::string name = expr.to_string();

  for (const auto &n : nodes) {
    if (!n)
      continue;
    if (z3::eq(n->expr, expr))
      return n;
  }

  auto newNode = std::make_shared<Node>(name, expr, false, false);

  nodes.push_back(newNode);
  return newNode;
}

bool Z3Graph::materializeSubexpressionNode(
    const std::shared_ptr<Node> &parentNode, const z3::expr &subexpr,
    std::shared_ptr<Node> &outNode) {
  if (!parentNode)
    return false;

  outNode = getOrCreateExprNode(subexpr);
  if (!outNode)
    return false;

  auto addUniqueEdge = [](const std::shared_ptr<Node> &src,
                          const std::shared_ptr<Node> &dst) {
    if (!src || !dst)
      return;
    for (const auto &e : src->edges) {
      if (e == dst)
        return;
    }
    src->edges.push_back(dst);
  };

  // Parent depends on this materialized subexpr.
  addUniqueEdge(parentNode, outNode);

  // This new node depends on its direct arguments.
  for (unsigned i = 0; i < subexpr.num_args(); ++i) {
    auto childNode = getOrCreateExprNode(subexpr.arg(i));
    if (!childNode)
      continue;
    addUniqueEdge(outNode, childNode);
  }

  return true;
}

// std::unique_ptr<Z3Graph> Z3Graph::createQuasiRegisterGraph() {
//   assert(domTree &&
//          "Dominator tree must be built before creating quasi-register
//          graph");

//   std::set<const Z3Graph::Node *> excludeSet;
//   std::set<const Z3Graph::Node *> promoteSet;

//   for (const auto &node : domTree->nodes) {
//     if (!node || !node->graphNode)
//       continue;

//     if (excludeSet.count(node->graphNode.get())) {
//       continue;
//     }

//     const auto &e = node->graphNode->expr;
//     if (e.is_bool() || (e.is_bv() && e.get_sort().bv_size() == 1)) {
//       continue;
//     }

//     // We have a bitvector or integer expression. Find the closest boolean
//     node
//     // that dominates this node.
//     auto findClosestBooleanDominator =
//         [](const std::shared_ptr<Z3Graph::DominatorTree::Node> &start)
//         -> std::shared_ptr<Z3Graph::DominatorTree::Node> {
//       auto cur = start ? start->parent : nullptr;

//       while (cur) {
//         if (cur->graphNode) {
//           const auto &e = cur->graphNode->expr;

//           if (e.is_bool() || (e.is_bv() && e.get_sort().bv_size() == 1))
//             return cur;
//         }

//         cur = cur->parent;
//       }

//       return nullptr;
//     };

//     auto closestBoolDom = findClosestBooleanDominator(node);
//     if (!closestBoolDom || !closestBoolDom->graphNode) {
//       llvm::errs() << "No boolean dominator found for node "
//                    << node->graphNode->name
//                    << " with expression: " <<
//                    node->graphNode->expr.to_string()
//                    << " and type: "
//                    << node->graphNode->expr.get_sort().to_string()
//                    << ", skipping promotion\n";
//       // assert(false);
//       continue;
//     }

//     // Check if the boolean dominator is complicated expression where the
//     // bitvector is lower in the expression. If so, create a new node for
//     this
//     // part of the the expression, so we only exclude the minimal amolunt of
//     // nodes.

//     auto closestExpr = closestBoolDom->graphNode->expr;

//     llvm::errs() << "Closest bool dominator: "
//                  << closestBoolDom->graphNode->name << " with expression: "
//                  << closestBoolDom->graphNode->expr.to_string() << " and
//                  type: "
//                  << closestBoolDom->graphNode->expr.get_sort().to_string()
//                  << " for node: " << node->graphNode->name
//                  << " with expresson: " << node->graphNode->expr.to_string()
//                  << " and type: "
//                  << node->graphNode->expr.get_sort().to_string() << "\n";

//     promoteSet.insert(closestBoolDom->graphNode.get());

//     // llvm::errs() << node->graphNode->name << " <promote> "
//     //              << closestBoolDom->graphNode->name << "\n";

//     // for (auto n : promoteSet) {
//     //   llvm::errs() << "Promoting " << n->name
//     //                << " to quasi-register with expression: "
//     //                << n->expr.to_string()
//     //                << " and type: " << n->expr.get_sort().to_string() <<
//     //                "\n";
//     // }

//     // Traverse downward and exclude all dominated nodes, as they will be all
//     // formed toeghetr into the quasi register
//     auto excludeDominated = [&](const std::shared_ptr<
//                                 Z3Graph::DominatorTree::Node> &start) {
//       std::function<void(const std::shared_ptr<Z3Graph::DominatorTree::Node>
//       &)>
//           dfs = [&](const std::shared_ptr<Z3Graph::DominatorTree::Node> &n) {
//             if (!n || !n->graphNode)
//               return;

//             excludeSet.insert(n->graphNode.get());

//             // The promotion is superceded if we decide to promote a more
//             // dominant node.
//             if (promoteSet.count(n->graphNode.get()))
//               promoteSet.erase(n->graphNode.get());

//             for (const auto &child : n->children) {
//               if (!child || !child->graphNode)
//                 continue;

//               // Stop if another boolean node is encountered
//               if (child->graphNode->expr.is_bool())
//                 continue;

//               dfs(child);
//             }
//           };

//       dfs(start);
//     };

//     for (auto child : closestBoolDom->children) {
//       excludeDominated(child);
//     }
//     // Traverse downward in the normal graph and exclude all reachable nodes,
//     as
//     // they will be formed together into the quasi-register.
//     auto excludeReachableInGraph =
//         [&](const std::shared_ptr<Z3Graph::Node> &start) {
//           std::set<const Z3Graph::Node *> visited;

//           std::function<void(const std::shared_ptr<Z3Graph::Node> &)> dfs =
//               [&](const std::shared_ptr<Z3Graph::Node> &n) {
//                 if (!n)
//                   return;

//                 if (visited.count(n.get()))
//                   return;
//                 visited.insert(n.get());

//                 excludeSet.insert(n.get());

//                 // The promotion is superceded if we decide to promote a more
//                 // dominant node.
//                 if (promoteSet.count(n.get()))
//                   promoteSet.erase(n.get());

//                 assert(n->edges.size() == n->isBackwardEdge.size() &&
//                        "edges and isBackwardEdge must stay parallel");

//                 for (size_t i = 0; i < n->edges.size(); ++i) {
//                   const auto &child = n->edges[i];
//                   bool isBackedge = n->isBackwardEdge[i];

//                   if (!child) {
//                     continue;
//                   }

//                   // Never traverse into backward edges.
//                   if (isBackedge)
//                     continue;

//                   dfs(child);
//                 }
//               };

//           if (!start)
//             return;

//           dfs(start);
//         };

//     // Start from the graph children of the boolean dominator, not the
//     dominator
//     // tree.
//     int c = 0;
//     for (const auto &child : closestBoolDom->graphNode->edges) {
//       if (!child) {
//         c++;
//         continue;
//       }

//       if (!(node->graphNode->isBackwardEdge.size() > c)) {
//         llvm::errs() << "Node: " << node->graphNode->name
//                      << ", child index: " << c
//                      << ", edges size: " << node->graphNode->edges.size()
//                      << ", backward edge size: "
//                      << node->graphNode->isBackwardEdge.size() << "c" << c
//                      << "\n";
//       }
//       assert(closestBoolDom->graphNode->isBackwardEdge.size() > c &&
//              "Backward edge issue");
//       if (closestBoolDom->graphNode->isBackwardEdge[c]) {
//         c++;
//         continue;
//       }

//       excludeReachableInGraph(child);
//       c++;
//     }
//   }

//   llvm::errs() << "Exclude size: " << excludeSet.size()
//                << ", total size: " << this->nodes.size() << "\n";

//   // TODO: Promotion name inheritance so we can still understand where this
//   // symbol is coming from.

//   llvm::errs() << "Nodes to promote:\n";
//   for (auto n : promoteSet) {
//     llvm::errs() << "Promoting " << n->name << " to quasi-register\n";
//   }
//   // llvm::errs() << "\n\n\n";

//   std::vector<std::shared_ptr<Node>> newNodes;
//   std::vector<std::string> newNodeNames;
//   std::vector<z3::expr> newNodeExprs;
//   std::set<std::string> newRegs;
//   std::string newRootName = rootName;

//   std::unordered_map<const Node *, std::shared_ptr<Node>> oldToNew;

//   for (const auto &node : nodes) {
//     if (!node)
//       continue;

//     if (excludeSet.count(node.get())) {
//       continue;
//     }

//     std::shared_ptr<Node> newNode;
//     if (promoteSet.count(node.get())) {
//       newNode = std::make_shared<Node>(node->name, node->expr,
//       node->isRegister,
//                                        true);
//     } else {
//       newNode =
//           std::make_shared<Node>(node->name, node->expr, node->isRegister);
//     }

//     newNodes.push_back(newNode);
//     oldToNew[node.get()] = newNode;

//     newNodeNames.push_back(newNode->name);
//     newNodeExprs.push_back(newNode->expr);
//     if (newNode->isRegister)
//       newRegs.insert(newNode->name);

//     if (node->name == rootName)
//       newRootName = newNode->name;
//   }

//   // Rebuild edges: only keep edges whose target is not excluded.
//   for (const auto &node : nodes) {
//     if (!node)
//       continue;

//     if (excludeSet.count(node.get()))
//       continue;

//     auto srcIt = oldToNew.find(node.get());
//     if (srcIt == oldToNew.end())
//       continue;

//     auto newSrc = srcIt->second;

//     for (const auto &edge : node->edges) {
//       if (!edge)
//         continue;

//       if (excludeSet.count(edge.get()))
//         continue;

//       auto dstIt = oldToNew.find(edge.get());
//       if (dstIt == oldToNew.end())
//         continue;

//       // Exclude self-edges.
//       if (dstIt->second.get() == newSrc.get())
//         continue;

//       newSrc->edges.push_back(dstIt->second);
//     }
//   }

//   std::unique_ptr<Z3Graph> quasiRegGraph = std::make_unique<Z3Graph>(
//       ctx, newNodeNames, newRootName, newNodeExprs, newRegs);
//   quasiRegGraph->nodes = newNodes;

//   return quasiRegGraph;
// }

std::unique_ptr<Z3Graph> Z3Graph::createQuasiRegisterGraph() {
  std::set<const Z3Graph::Node *> excludeSet;
  std::set<const Z3Graph::Node *> promoteSet;

  auto isBoolLike = [](const z3::expr &e) {
    return e.is_bool() || (e.is_bv() && e.get_sort().bv_size() == 1);
  };

  auto excludeReachableInGraph =
      [&](const std::shared_ptr<Z3Graph::Node> &start) {
        std::set<const Z3Graph::Node *> visited;

        std::function<void(const std::shared_ptr<Z3Graph::Node> &)> dfs =
            [&](const std::shared_ptr<Z3Graph::Node> &n) {
              if (!n)
                return;

              if (visited.count(n.get()))
                return;
              visited.insert(n.get());

              excludeSet.insert(n.get());

              // If something below a promoted boundary was also marked for
              // promote, cancel that deeper promotion.
              if (promoteSet.count(n.get()))
                promoteSet.erase(n.get());

              assert(n->edges.size() == n->isBackwardEdge.size() &&
                     "edges and isBackwardEdge must stay parallel");

              for (size_t i = 0; i < n->edges.size(); ++i) {
                const auto &child = n->edges[i];
                bool isBackedge = n->isBackwardEdge[i];

                if (!child)
                  continue;
                if (isBackedge)
                  continue;

                dfs(child);
              }
            };

        if (start)
          dfs(start);
      };

  // Traverse the graph forward. Only boolean/BV1 nodes can become
  // quasi-register boundaries.
  std::set<const Z3Graph::Node *> visited;

  std::function<void(const std::shared_ptr<Z3Graph::Node> &)> walk =
      [&](const std::shared_ptr<Z3Graph::Node> &node) {
        if (!node)
          return;

        if (visited.count(node.get()))
          return;
        visited.insert(node.get());

        assert(node->edges.size() == node->isBackwardEdge.size() &&
               "edges and isBackwardEdge must stay parallel");

        bool nodeIsBoolLike = isBoolLike(node->expr);

        for (size_t i = 0; i < node->edges.size(); ++i) {
          const auto &child = node->edges[i];
          bool isBackedge = node->isBackwardEdge[i];

          if (!child)
            continue;

          // Never walk backward edges.
          if (isBackedge)
            continue;

          bool childIsBoolLike = isBoolLike(child->expr);

          // Boundary condition:
          // current node is bool/BV1, but child is not.
          if (nodeIsBoolLike && !childIsBoolLike) {
            // llvm::errs() << "Promoting " << node->name
            //              << " to quasi-register because child " <<
            //              child->name
            //              << " has non-bool-like expr: "
            //              << child->expr.to_string() << " with type "
            //              << child->expr.get_sort().to_string() << "\n";
            promoteSet.insert(node.get());

            // llvm::errs() << "Promoting " << node->name
            //              << " to quasi-register because child " <<
            //              child->name
            //              << " has non-bool-like expr: "
            //              << child->expr.to_string() << " with type "
            //              << child->expr.get_sort().to_string() << "\n";

            excludeReachableInGraph(child);
            continue;
          }

          walk(child);
        }
      };

  // Start from the graph root if present.
  std::shared_ptr<Z3Graph::Node> rootNode = nullptr;
  for (const auto &node : nodes) {
    if (!node)
      continue;
    if (node->name == rootName) {
      rootNode = node;
      break;
    }
  }

  if (rootNode) {
    walk(rootNode);
  } else {
    // Fallback: walk all nodes.
    for (const auto &node : nodes) {
      if (!node)
        continue;
      walk(node);
    }
  }

  // llvm::errs() << "Exclude size: " << excludeSet.size()
  //              << ", total size: " << this->nodes.size() << "\n";

  // llvm::errs() << "Nodes to promote:\n";
  // for (auto n : promoteSet) {
  //   llvm::errs() << "Promoting " << n->name << " to quasi-register\n";
  // }

  std::vector<std::shared_ptr<Node>> newNodes;
  std::vector<std::string> newNodeNames;
  std::vector<z3::expr> newNodeExprs;
  std::set<std::string> newRegs;
  std::string newRootName = rootName;

  std::unordered_map<const Node *, std::shared_ptr<Node>> oldToNew;

  for (const auto &node : nodes) {
    if (!node)
      continue;

    if (excludeSet.count(node.get()))
      continue;

    std::shared_ptr<Node> newNode;
    if (promoteSet.count(node.get())) {
      newNode = std::make_shared<Node>(node->name, node->expr, node->isRegister,
                                       true);
    } else {
      newNode =
          std::make_shared<Node>(node->name, node->expr, node->isRegister);
    }

    newNodes.push_back(newNode);
    oldToNew[node.get()] = newNode;

    newNodeNames.push_back(newNode->name);
    newNodeExprs.push_back(newNode->expr);
    if (newNode->isRegister)
      newRegs.insert(newNode->name);

    if (node->name == rootName)
      newRootName = newNode->name;
  }

  // Rebuild edges: only keep edges whose target is not excluded.
  for (const auto &node : nodes) {
    if (!node)
      continue;

    if (excludeSet.count(node.get()))
      continue;

    auto srcIt = oldToNew.find(node.get());
    if (srcIt == oldToNew.end())
      continue;

    auto newSrc = srcIt->second;

    assert(node->edges.size() == node->isBackwardEdge.size() &&
           "edges and isBackwardEdge must stay parallel");

    for (size_t i = 0; i < node->edges.size(); ++i) {
      const auto &edge = node->edges[i];
      if (!edge)
        continue;

      if (excludeSet.count(edge.get()))
        continue;

      auto dstIt = oldToNew.find(edge.get());
      if (dstIt == oldToNew.end())
        continue;

      // Exclude self-edges.
      if (dstIt->second.get() == newSrc.get())
        continue;

      newSrc->edges.push_back(dstIt->second);
      newSrc->isBackwardEdge.push_back(node->isBackwardEdge[i]);
    }
  }

  std::unique_ptr<Z3Graph> quasiRegGraph = std::make_unique<Z3Graph>(
      ctx, newNodeNames, newRootName, newNodeExprs, newRegs);
  quasiRegGraph->nodes = newNodes;

  return quasiRegGraph;
}

z3::expr Z3Graph::Canonicalizer::toCNF(const z3::expr &e) {
  z3::context &ctx = e.ctx();

  // First convert to NNF.
  z3::goal g(ctx);
  g.add(e);

  z3::tactic t(ctx, "nnf");
  z3::apply_result r = t(g);

  assert(r.size() > 0 && "nnf tactic returned no subgoals");
  z3::expr nnf = r[0].as_expr();

  auto isOp = [](const z3::expr &x, Z3_decl_kind k) {
    return x.is_app() && x.decl().decl_kind() == k;
  };

  std::function<z3::expr(const z3::expr &, const z3::expr &)>
      distributeOrOverAnd;
  std::function<z3::expr(const z3::expr &)> cnf;

  distributeOrOverAnd = [&](const z3::expr &a, const z3::expr &b) -> z3::expr {
    // (A1 & A2 & ...) | B  ==>  (A1|B) & (A2|B) & ...
    if (isOp(a, Z3_OP_AND)) {
      z3::expr_vector clauses(ctx);
      for (unsigned i = 0; i < a.num_args(); ++i)
        clauses.push_back(distributeOrOverAnd(a.arg(i), b));
      return z3::mk_and(clauses);
    }

    // A | (B1 & B2 & ...)  ==>  (A|B1) & (A|B2) & ...
    if (isOp(b, Z3_OP_AND)) {
      z3::expr_vector clauses(ctx);
      for (unsigned i = 0; i < b.num_args(); ++i)
        clauses.push_back(distributeOrOverAnd(a, b.arg(i)));
      return z3::mk_and(clauses);
    }

    // Base case: neither side is AND, so just OR them.
    z3::expr_vector disj(ctx);
    disj.push_back(a);
    disj.push_back(b);
    return z3::mk_or(disj);
  };

  cnf = [&](const z3::expr &x) -> z3::expr {
    // AND: recursively CNF each child, then rebuild AND.
    if (isOp(x, Z3_OP_AND)) {
      z3::expr_vector args(ctx);
      for (unsigned i = 0; i < x.num_args(); ++i)
        args.push_back(cnf(x.arg(i)));
      return z3::mk_and(args);
    }

    // OR: recursively CNF children, then distribute pairwise.
    if (isOp(x, Z3_OP_OR)) {
      assert(x.num_args() > 0 && "OR node must have at least one argument");

      z3::expr acc = cnf(x.arg(0));
      for (unsigned i = 1; i < x.num_args(); ++i) {
        z3::expr rhs = cnf(x.arg(i));
        acc = distributeOrOverAnd(acc, rhs);
      }
      return acc;
    }

    // Literal (or NOT of literal, since input is already NNF)
    return x;
  };

  return cnf(nnf).simplify();
}

z3::expr Z3Graph::Canonicalizer::toDNF(const z3::expr &e) {
  z3::context &ctx = e.ctx();

  // First convert to NNF.
  z3::goal g(ctx);
  g.add(e);

  z3::tactic t(ctx, "nnf");
  z3::apply_result r = t(g);

  assert(r.size() > 0 && "nnf tactic returned no subgoals");
  z3::expr nnf = r[0].as_expr();

  auto isOp = [](const z3::expr &x, Z3_decl_kind k) {
    return x.is_app() && x.decl().decl_kind() == k;
  };

  std::function<z3::expr(const z3::expr &, const z3::expr &)>
      distributeAndOverOr;
  std::function<z3::expr(const z3::expr &)> dnf;

  distributeAndOverOr = [&](const z3::expr &a, const z3::expr &b) -> z3::expr {
    // (A1 | A2 | ...) & B  ==>  (A1&B) | (A2&B) | ...
    if (isOp(a, Z3_OP_OR)) {
      z3::expr_vector clauses(ctx);
      for (unsigned i = 0; i < a.num_args(); ++i)
        clauses.push_back(distributeAndOverOr(a.arg(i), b));
      return z3::mk_or(clauses);
    }

    // A & (B1 | B2 | ...)  ==>  (A&B1) | (A&B2) | ...
    if (isOp(b, Z3_OP_OR)) {
      z3::expr_vector clauses(ctx);
      for (unsigned i = 0; i < b.num_args(); ++i)
        clauses.push_back(distributeAndOverOr(a, b.arg(i)));
      return z3::mk_or(clauses);
    }

    // Base case: neither side is OR, so just AND them.
    z3::expr_vector conj(ctx);
    conj.push_back(a);
    conj.push_back(b);
    return z3::mk_and(conj);
  };

  dnf = [&](const z3::expr &x) -> z3::expr {
    // OR: recursively DNF each child, then rebuild OR.
    if (isOp(x, Z3_OP_OR)) {
      z3::expr_vector args(ctx);
      for (unsigned i = 0; i < x.num_args(); ++i)
        args.push_back(dnf(x.arg(i)));
      return z3::mk_or(args);
    }

    // AND: recursively DNF children, then distribute pairwise.
    if (isOp(x, Z3_OP_AND)) {
      assert(x.num_args() > 0 && "AND node must have at least one argument");

      z3::expr acc = dnf(x.arg(0));
      for (unsigned i = 1; i < x.num_args(); ++i) {
        z3::expr rhs = dnf(x.arg(i));
        acc = distributeAndOverOr(acc, rhs);
      }
      return acc;
    }

    // Literal (or NOT of literal, since input is already NNF)
    return x;
  };

  return dnf(nnf).simplify();
}

z3::expr Z3Graph::Canonicalizer::toNNF(const z3::expr &e) {
  z3::context &ctx = e.ctx();

  z3::goal g(ctx);
  g.add(e);

  z3::tactic t(ctx, "nnf");
  z3::apply_result r = t(g);

  // Fast path (most common)
  if (r.size() == 1 && r[0].size() == 1)
    return r[0][0];

  // Otherwise AND everything together
  z3::expr out = ctx.bool_val(true);
  for (unsigned i = 0; i < r.size(); ++i)
    for (unsigned j = 0; j < r[i].size(); ++j)
      out = out && r[i][j];

  return out;
}

z3::expr Z3Graph::Canonicalizer::eliminateITE(const z3::expr &e) {
  // z3::context &ctx = e.ctx();
  // std::unordered_map<Z3_ast, z3::expr> cache;

  // std::function<z3::expr(const z3::expr &)> rewrite =
  //     [&](const z3::expr &cur) -> z3::expr {
  //   Z3_ast ast = cur;
  //   auto it = cache.find(ast);
  //   if (it != cache.end())
  //     return it->second;

  //   z3::expr result = cur;

  //   // Leaf / atom
  //   if (cur.num_args() == 0) {
  //     result = cur;
  //   } else {
  //     Z3_decl_kind dk = cur.decl().decl_kind();

  //     // Special-case ite
  //     if (dk == Z3_OP_ITE && cur.num_args() == 3) {
  //       z3::expr c = rewrite(cur.arg(0));
  //       z3::expr t = rewrite(cur.arg(1));
  //       z3::expr f = rewrite(cur.arg(2));

  //       if (cur.is_bool()) {
  //         result = (c && t) || ((!c) && f);
  //       } else {
  //         // For non-bool results, keep ite for now.
  //         // A full elimination for BV-valued ite would require introducing
  //         // mux-style encodings or fresh variables.
  //         result = z3::ite(c, t, f);
  //       }
  //     } else {
  //       // Generic rebuild of children
  //       z3::expr_vector args(ctx);
  //       args.resize(cur.num_args());
  //       for (unsigned i = 0; i < cur.num_args(); ++i)
  //         args[i] = rewrite(cur.arg(i));

  //       result = cur.decl()(args);
  //     }
  //   }

  //   cache.emplace(ast, result);
  //   return result;
  // };

  // return rewrite(e);
  return e;
}