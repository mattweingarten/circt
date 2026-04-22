
#ifndef CIRCT_DIALECT_FIRRTL_TransformZ3Graph_H
#define CIRCT_DIALECT_FIRRTL_TransformZ3Graph_H

#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <z3++.h>

namespace llvm {
class raw_ostream;
}

class Z3Graph {
public:
  struct Node {
    std::string name;
    z3::expr expr;
    std::vector<std::shared_ptr<Node>> edges;
    std::vector<bool>
        isBackwardEdge; // parallel vector to children, true if edge is a
                        // backedge in the original graph
    bool isRegister;
    bool isQuasiRegister;

    Node(std::string name, const z3::expr &expr, bool isRegister,
         bool isQuasiRegister = false)
        : name(std::move(name)), expr(expr), isRegister(isRegister),
          isQuasiRegister(isQuasiRegister) {}

    // static std::unique_ptr<Node> makeQuasiNode() const {
    //   return std::make_unique<Node>(name, expr, isRegister, true);
    // }

    // static std::unique_ptr<Node> makeNode() const {
    //   return std::make_unique<Node>(name, expr, isRegister, false);
    // }

    bool operator==(const Node &other) const { return name == other.name; }
  };

  class DominatorTree {
  public:
    struct Node {
      std::shared_ptr<Z3Graph::Node> graphNode;
      std::vector<std::shared_ptr<Node>> children;
      std::shared_ptr<Node> parent;
    };

    struct DominatorTreeBuilderState {
      std::map<std::shared_ptr<Z3Graph::Node>,
               std::set<std::shared_ptr<Z3Graph::Node>>>
          doms;
      std::map<std::shared_ptr<Z3Graph::Node>,
               std::set<std::shared_ptr<Z3Graph::Node>>>
          preds;
    };

    static std::unique_ptr<DominatorTree> build(const Z3Graph &graph);

    void printFullGraphDot(llvm::raw_ostream &os) const;

    void printSubgraphDot(llvm::raw_ostream &os, const Z3Graph &graph,
                          const std::string nodeName) const;
    std::vector<std::shared_ptr<Node>> nodes;
    std::map<std::string, std::shared_ptr<Node>> nameToNode;
    DominatorTreeBuilderState state;
  };

  class Canonicalizer {
  public:
    using RewriteResult = std::optional<z3::expr>;
    using RewriteRule = std::function<RewriteResult(const z3::expr &)>;

    Canonicalizer(z3::context &ctx);

    z3::expr canonicalize(const z3::expr &e);

    void addRule(RewriteRule rule);

    // Shouldn't be here in Canonicalzor, but TODO
    static z3::expr toCNF(const z3::expr &e);
    static z3::expr toDNF(const z3::expr &e);
    static z3::expr toNNF(const z3::expr &e);
    static z3::expr eliminateITE(const z3::expr &e);

  private:
    z3::context &ctx;
    std::vector<RewriteRule> rules;

    z3::expr canonicalizeSubtreeToFixpoint(const z3::expr &e);
    z3::expr rebuildWithCanonicalChildren(const z3::expr &e);
    z3::expr applyLocalRulesToFixpoint(const z3::expr &e);

    static bool isBVConst(const z3::expr &e, uint64_t value);
    static bool isBoolConstTrue(const z3::expr &e);
    static bool isBoolConstFalse(const z3::expr &e);
    static bool isOP(const z3::expr &e, Z3_decl_kind kind);
  };

  z3::context &ctx;
  std::string rootName;
  bool valid;
  std::vector<std::string> nodeNames;
  std::set<std::string> registers;
  std::vector<z3::expr> nodeExprs;
  Canonicalizer canonicalizer;

  // Dataflow graph
  std::shared_ptr<Node> root;
  std::vector<std::shared_ptr<Z3Graph::Node>> nodes;

  std::unique_ptr<DominatorTree> domTree;

  struct ExprHash {
    std::size_t operator()(const z3::expr &e) const noexcept {
      return Z3_get_ast_id(e.ctx(), e);
    }
  };

  struct NodeHash {
    std::size_t operator()(const Node &n) const {
      return std::hash<std::string>()(n.name);
    }
  };

  struct ExprEq {
    bool operator()(const z3::expr &a, const z3::expr &b) const noexcept {
      return Z3_is_eq_ast(a.ctx(), a, b);
    }
  };

  Z3Graph(z3::context &context, const std::vector<std::string> &names,
          std::string rootName, const std::vector<z3::expr> &exprs,
          const std::set<std::string> &regs)
      : ctx(context), rootName(rootName), valid(true),
        canonicalizer(Canonicalizer(context)), domTree(nullptr) {

    if (names.size() != exprs.size()) {
      valid = false;
      return;
    }

    for (const auto &e : exprs) {
      if (&e.ctx() != &ctx) {
        valid = false;
        return;
      }
    }

    nodeNames = names;
    registers = regs;

    nodeExprs.reserve(exprs.size());
    for (const auto &e : exprs) {
      nodeExprs.push_back(e);
    }
  }

  void buildZ3Graph();

  z3::expr buildInlinedExpr() const;

  int size() const { return nodes.size(); }

  void buildDominatorTree();

  std::unique_ptr<Z3Graph>
  createSubgraph(const std::string &rootNodeName) const;

  bool isValid() const { return valid; }

  // Static helper methods: TODO, should be in seprare helper class probably
  static void walkAST(const z3::expr &root,
                      const std::function<void(const z3::expr &)> &fn);

  void printFullGraphDot(llvm::raw_ostream &os) const;
  void printSubgraphDot(llvm::raw_ostream &os, const std::string node) const;
  std::unique_ptr<Z3Graph> createQuasiRegisterGraph();

  // Promotion helpers:
  std::shared_ptr<Z3Graph::Node> getOrCreateExprNode(const z3::expr &expr);
  bool materializeSubexpressionNode(const std::shared_ptr<Node> &parentNode,
                                    const z3::expr &subexpr,
                                    std::shared_ptr<Node> &outNode);
};

#endif // CIRCT_DIALECT_FIRRTL_TransformZ3Graph_H
