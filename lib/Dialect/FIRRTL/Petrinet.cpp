#include "circt/Dialect/FIRRTL/Petrinet.h"
#include "circt/Dialect/FIRRTL/TransformZ3Graph.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FormatVariadic.h"

#include <algorithm>
#include <variant>

unsigned Petrinet::transitionID = 0;
unsigned Petrinet::arcID = 0;
unsigned Petrinet::nodeID = 0;

std::unique_ptr<Petrinet> Petrinet::createFromZ3(const z3::expr &e,
                                                 const std::string &name) {
  auto net = std::make_unique<Petrinet>();
  net->integrateExpression(e, name, /*markRootFinal=*/true);
  return net;
}

void Petrinet::extendGraphWithExpression(const z3::expr &e,
                                         const std::string &name) {
  integrateExpression(e, name, /*markRootFinal=*/false);
}
void Petrinet::integrateExpression(const z3::expr &e, const std::string &name,
                                   bool markRootFinal) {
  std::unordered_map<std::string, std::shared_ptr<Place>> nameToGraph;
  std::unordered_map<std::string, std::shared_ptr<Transition>> transitionByKey;
  llvm::DenseMap<Z3_ast, std::string> exprToPlaceName;
  llvm::DenseMap<Z3_ast, std::shared_ptr<Place>> exprToBuiltPlace;
  unsigned intermediateCounter = 0;

  // Seed map from existing places/transitions so extension reuses already-built
  // graph structure.
  for (const auto &node : nodes) {
    if (!node)
      continue;

    if (llvm::isa<Place>(node.get())) {
      auto place = std::static_pointer_cast<Place>(node);
      nameToGraph[place->name] = place;
      continue;
    }

    if (llvm::isa<Transition>(node.get())) {
      auto t = std::static_pointer_cast<Transition>(node);
      transitionByKey[t->name] = t;
      continue;
    }
  }

  auto isBoolConst = [](const z3::expr &expr) {
    auto dk = expr.decl().decl_kind();
    return dk == Z3_OP_TRUE || dk == Z3_OP_FALSE;
  };

  auto isAtomicVar = [&](const z3::expr &expr) {
    return expr.is_const() && !expr.is_numeral() && !isBoolConst(expr) &&
           expr.num_args() == 0;
  };

  auto placeNameForExpr = [&](const z3::expr &expr) -> std::string {
    if (z3::eq(expr, e))
      return name;

    if (isAtomicVar(expr))
      return expr.decl().name().str();

    Z3_ast ast = expr;
    auto it = exprToPlaceName.find(ast);
    if (it != exprToPlaceName.end())
      return it->second;

    std::string generatedName = "tmp_" + std::to_string(intermediateCounter++);
    exprToPlaceName[ast] = generatedName;
    return generatedName;
  };

  auto exprKey = [&](const z3::expr &expr) -> std::string {
    return expr.to_string();
  };

  auto transitionKeyForExpr = [&](llvm::StringRef opName, const z3::expr &expr,
                                  std::optional<unsigned> childIndex =
                                      std::nullopt) -> std::string {
    std::string key = opName.str();
    key += "::";
    key += exprKey(expr);
    if (childIndex)
      key += "::" + std::to_string(*childIndex);
    return key;
  };

  auto getOrCreateTransition = [&](llvm::StringRef key, llvm::StringRef kind,
                                   int depth) -> std::shared_ptr<Transition> {
    auto it = transitionByKey.find(key.str());
    if (it != transitionByKey.end())
      return it->second;

    auto t = makeTransition(kind.str(), depth);
    t->name = key.str();
    transitionByKey[key.str()] = t;
    return t;
  };

  auto hasEdge = [](const auto &src, const auto &dst) -> bool {
    for (const auto &edge : src->outgoingEdges) {
      if (edge && edge->to.get() == dst.get())
        return true;
    }

    for (const auto &edge : dst->incomingEdges) {
      if (edge && edge->from.get() == src.get())
        return true;
    }

    return false;
  };

  auto addArcIfMissing = [&](const auto &src, const auto &dst,
                             llvm::StringRef label) {
    if (!hasEdge(src, dst))
      addArc(src, dst, label.str());
  };

  auto addInhibitorArcIfMissing = [&](const std::shared_ptr<Place> &src,
                                      const std::shared_ptr<Transition> &dst,
                                      llvm::StringRef label) {
    if (!hasEdge(src, dst))
      addInhibitorArc(src, dst, label.str());
  };

  // Ensure root place exists.
  auto rootPlace = getOrCreatePlace(name, nameToGraph, /*rank=*/0, e);
  if (markRootFinal)
    rootPlace->final = true;

  // Special case: if the whole expression is just a top-level variable,
  // still create a normal transition from that variable place to the root
  // place.
  if (isAtomicVar(e)) {
    auto varPlace =
        getOrCreatePlace(e.decl().name().str(), nameToGraph, /*rank=*/1, e);

    auto t = getOrCreateTransition(transitionKeyForExpr("var", e), "var",
                                   /*depth=*/0);

    addArcIfMissing(varPlace, t, "in");
    addArcIfMissing(t, rootPlace, "out");

    exprToBuiltPlace[Z3_ast(e)] = rootPlace;
    return;
  }

  // First pass: create places for atomic variables, preserving first-seen rank.
  std::function<void(const z3::expr &, int)> createPlaces =
      [&](const z3::expr &expr, int rank) {
        if (isAtomicVar(expr)) {
          std::string varName = expr.decl().name().str();
          (void)getOrCreatePlace(varName, nameToGraph, rank, expr);
          return;
        }

        for (unsigned i = 0; i < expr.num_args(); ++i)
          createPlaces(expr.arg(i), rank + 1);
      };

  createPlaces(e, 0);

  std::function<std::shared_ptr<Place>(const z3::expr &, int)> buildExpr;
  std::function<void(const z3::expr &, const std::shared_ptr<Transition> &,
                     int)>
      connectChildToTransition;

  connectChildToTransition = [&](const z3::expr &child,
                                 const std::shared_ptr<Transition> &t,
                                 int rank) {
    if (child.decl().decl_kind() == Z3_OP_NOT) {
      assert(child.num_args() == 1 && "NOT must have exactly one operand");
      auto operandPlace = buildExpr(child.arg(0), rank + 1);
      addInhibitorArcIfMissing(operandPlace, t, "inh");
      return;
    }

    auto childPlace = buildExpr(child, rank + 1);
    addArcIfMissing(childPlace, t, "in");
  };

  buildExpr = [&](const z3::expr &expr, int rank) -> std::shared_ptr<Place> {
    Z3_ast ast = expr;

    auto builtIt = exprToBuiltPlace.find(ast);
    if (builtIt != exprToBuiltPlace.end())
      return builtIt->second;

    if (isAtomicVar(expr)) {
      auto p =
          getOrCreatePlace(expr.decl().name().str(), nameToGraph, rank, expr);
      exprToBuiltPlace[ast] = p;
      return p;
    }

    auto outPlace =
        getOrCreatePlace(placeNameForExpr(expr), nameToGraph, rank, expr);
    exprToBuiltPlace[ast] = outPlace;

    auto kind = expr.decl().decl_kind();

    if (kind == Z3_OP_NOT) {
      assert(expr.num_args() == 1 && "NOT must have exactly one operand");

      auto t =
          getOrCreateTransition(transitionKeyForExpr("not", expr), "not", rank);

      auto operandPlace = buildExpr(expr.arg(0), rank + 1);
      addInhibitorArcIfMissing(operandPlace, t, "not_inh");
      addArcIfMissing(t, outPlace, "not_out");
      return outPlace;
    }

    if (kind == Z3_OP_AND) {
      auto t =
          getOrCreateTransition(transitionKeyForExpr("and", expr), "and", rank);

      for (unsigned i = 0; i < expr.num_args(); ++i)
        connectChildToTransition(expr.arg(i), t, rank);

      addArcIfMissing(t, outPlace, "and_out");
      return outPlace;
    }

    if (kind == Z3_OP_OR) {
      for (unsigned i = 0; i < expr.num_args(); ++i) {
        auto t = getOrCreateTransition(transitionKeyForExpr("or", expr, i),
                                       "or", rank);
        connectChildToTransition(expr.arg(i), t, rank);
        addArcIfMissing(t, outPlace, "or_out");
      }
      return outPlace;
    }

    if (expr.num_args() > 0) {
      auto t =
          getOrCreateTransition(transitionKeyForExpr("op", expr), "op", rank);
      for (unsigned i = 0; i < expr.num_args(); ++i)
        connectChildToTransition(expr.arg(i), t, rank);
      addArcIfMissing(t, outPlace, "op_out");
    }

    return outPlace;
  };

  buildExpr(e, 0);
}

void Petrinet::writeGraph(llvm::raw_ostream &os) const {
  os << "digraph PetriNet {\n";
  os << "  rankdir=LR;\n";
  os << "graph [ranksep=10];\n";
  os << "  node [fontname=\"Helvetica\"];\n";

  // Emit all nodes (places + transitions)
  for (const auto &n : nodes) {
    if (!n)
      continue;

    if (auto *p = llvm::dyn_cast<Place>(n.get()))
      p->writeGraph(os);
    else if (auto *t = llvm::dyn_cast<Transition>(n.get()))
      t->writeGraph(os);
  }

  for (const auto &n : nodes) {
    if (!n)
      continue;

    if (auto *a = llvm::dyn_cast<Arc>(n.get()))
      a->writeGraph(os);
    else if (auto *ia = llvm::dyn_cast<InhibitorArc>(n.get()))
      ia->writeGraph(os);
  }

  os << "}\n";
}

void Petrinet::Place::writeGraph(llvm::raw_ostream &os) const {
  if (final)
    os << "{rank = max; ";
  os << "  \"" << getName() << "\" "
     << "[shape=circle, label=\"\", xlabel=\"" << getName() << "\""
     << ", width=0.8, height=0.8, fixedsize=true";

  if (final)
    os << ", peripheries=2";

  os << "];\n";
  if (final)
    os << "}\n";
}

void Petrinet::Transition::writeGraph(llvm::raw_ostream &os) const {
  unsigned n = std::max(1u, numIncoming);

  const double fieldHeight = 1.2;

  os << "  \"" << getName() << "\" [";
  os << "shape=record, ";
  os << "margin=\"0.03,0.02\", ";
  os << "width=0.06, ";
  os << "height=" << llvm::formatv("{0:F2}", n * fieldHeight) << ", ";
  os << "fixedsize=true, ";

  os << "label=\"";
  for (unsigned i = 0; i < n; ++i) {
    if (i)
      os << "|";
    os << "<f" << i << "> ";
  }
  os << "\", ";

  os << "xlabel=\"t" << id << "\"";

  os << "];\n";
}

void Petrinet::Arc::writeGraph(llvm::raw_ostream &os) const {
  if (!from || !to)
    return;

  // Incoming arc to a transition
  if (auto *t = llvm::dyn_cast<Petrinet::Transition>(to.get())) {
    os << "  \"" << from->getName() << "\" -> \"" << to->getName() << "\":f"
       << incomingIdx << ":w;\n";
    return;
  }

  // Outgoing arc from a transition
  if (llvm::isa<Petrinet::Transition>(from.get())) {
    os << "  \"" << from->getName() << "\":e -> \"" << to->getName() << "\";\n";
    return;
  }

  // Default
  os << "  \"" << from->getName() << "\" -> \"" << to->getName() << "\";\n";
}

void Petrinet::InhibitorArc::writeGraph(llvm::raw_ostream &os) const {
  if (!from || !to)
    return;

  if (auto *t = llvm::dyn_cast<Petrinet::Transition>(to.get())) {
    os << "  \"" << from->getName() << "\" -> \"" << to->getName() << "\":f"
       << incomingIdx << ":w [arrowhead=odot];\n";
    return;
  }

  if (llvm::isa<Petrinet::Transition>(from.get())) {
    os << "  \"" << from->getName() << "\":e -> \"" << to->getName()
       << "\" [arrowhead=odot];\n";
    return;
  }

  os << "  \"" << from->getName() << "\" -> \"" << to->getName()
     << "\" [arrowhead=odot];\n";
}

std::shared_ptr<Petrinet::Place> Petrinet::getOrCreatePlace(
    const std::string &placeName,
    std::unordered_map<std::string, std::shared_ptr<Place>> &nameToGraph,
    int rank, z3::expr expr) {
  auto it = nameToGraph.find(placeName);
  if (it != nameToGraph.end()) {
    auto place = it->second;

    // Preserve earliest / shallowest rank if desired.
    if (rank < place->rank)
      place->rank = rank;

    // If the existing place has no valid expr yet, populate it.
    Z3_ast existingAst = place->expr;
    Z3_ast newAst = expr;
    if (!existingAst && newAst)
      place->expr = expr;

    return place;
  }

  auto place = std::make_shared<Place>(placeName, expr);
  place->rank = rank;
  nodes.push_back(place);
  nameToGraph[placeName] = place;
  return place;
}

std::shared_ptr<Petrinet::Transition>
Petrinet::makeTransition(const std::string &prefix, int rank) {
  auto t = std::make_shared<Transition>(prefix + "_" +
                                        std::to_string(transitionID++));
  t->rank = rank;
  nodes.push_back(t);
  return t;
}

void Petrinet::addArc(const std::shared_ptr<Node> &from,
                      const std::shared_ptr<Node> &to,
                      const std::string &prefix) {

  auto a = std::make_shared<Arc>(prefix + "_" + std::to_string(arcID++));
  a->from = from;
  a->to = to;
  a->incomingIdx = a->to->numIncoming;
  a->to->numIncoming++;
  nodes.push_back(a);
  from->outgoingEdges.push_back(a);
  to->incomingEdges.push_back(a);
}

void Petrinet::addInhibitorArc(const std::shared_ptr<Node> &from,
                               const std::shared_ptr<Node> &to,
                               const std::string &prefix) {

  auto a =
      std::make_shared<InhibitorArc>(prefix + "_" + std::to_string(arcID++));
  a->from = from;
  a->to = to;

  a->incomingIdx = a->to->numIncoming;
  a->to->numIncoming++;
  nodes.push_back(a);
  from->outgoingEdges.push_back(a);
  to->incomingEdges.push_back(a);
}

void Petrinet::setDepth() {
  walkBackwardFromFinals<unsigned>(
      [](const std::shared_ptr<Petrinet::Place> &finalPlace) { return 0; },
      [](const std::shared_ptr<Petrinet::Node> &node, const unsigned &depth) {
        node->depth = std::max(node->depth, depth);
      },
      [](const std::shared_ptr<Petrinet::Node> &pred,
         const std::shared_ptr<Petrinet::ArcLike> &arc,
         const int &curDepth) { return curDepth + 1; });
}

static std::vector<z3::expr>
buildIncomingEdgePredicates(const std::shared_ptr<Petrinet::Transition> &t) {
  std::vector<z3::expr> preds;
  preds.reserve(t->incomingEdges.size());

  for (const auto &arc : t->incomingEdges) {
    if (!arc || !arc->from)
      continue;

    if (!llvm::isa<Petrinet::Place>(arc->from.get()))
      continue;

    auto place = std::static_pointer_cast<Petrinet::Place>(arc->from);

    auto toBool = [&](const z3::expr &e) -> std::optional<z3::expr> {
      if (e.is_bool())
        return e;

      if (e.is_bv()) {
        unsigned w = e.get_sort().bv_size();

        if (w == 1)
          return e == e.ctx().bv_val(1, 1);

        // fallback: non-zero check
        z3::expr zero = e.ctx().bv_val(0, w);

        llvm::errs() << "\n\n[PETRINET WARNING] Non-boolean place expr "
                     << "used as predicate (width=" << w << ")\n"
                     << "  expr: " << e.to_string() << "\n"
                     << "  -> converting via (expr != 0)\n\n";

        return e != zero;
      }

      llvm::errs() << "\n\n[PETRINET ERROR] Unsupported predicate type\n"
                   << "  expr: " << e.to_string() << "\n\n";

      return std::nullopt;
    };

    auto maybeBool = toBool(place->expr);
    if (!maybeBool)
      continue;

    z3::expr cond = *maybeBool;

    if (llvm::isa<Petrinet::InhibitorArc>(arc.get()))
      preds.push_back(cond);
    else
      preds.push_back(!cond);
  }

  return preds;
}

static z3::expr mkAnd(const std::vector<z3::expr> &exprs) {
  assert(!exprs.empty() && "expected non-empty expr list");

  z3::expr acc = exprs.front();
  for (size_t i = 1; i < exprs.size(); ++i)
    acc = acc && exprs[i];
  return acc;
}

std::vector<
    std::pair<std::shared_ptr<Petrinet::Transition>, std::vector<z3::expr>>>
Petrinet::getCountersNaive() const {
  std::vector<std::pair<std::shared_ptr<Transition>, std::vector<z3::expr>>>
      result;

  walkBackwardFromFinals<std::monostate>(
      [](const std::shared_ptr<Place> &) { return std::monostate{}; },
      [&](const std::shared_ptr<Node> &node, const std::monostate &) {
        if (!node || !llvm::isa<Transition>(node.get()))
          return;

        auto t = std::static_pointer_cast<Transition>(node);
        result.push_back({t, buildIncomingEdgePredicates(t)});
      },
      [](const std::shared_ptr<Node> &, const std::shared_ptr<ArcLike> &,
         const std::monostate &) { return std::monostate{}; });

  return result;
}

std::vector<
    std::pair<std::shared_ptr<Petrinet::Transition>, std::vector<z3::expr>>>
Petrinet::getCountersPairs() const {
  std::vector<std::pair<std::shared_ptr<Transition>, std::vector<z3::expr>>>
      result;

  walkBackwardFromFinals<std::monostate>(
      [](const std::shared_ptr<Place> &) { return std::monostate{}; },
      [&](const std::shared_ptr<Node> &node, const std::monostate &) {
        if (!node || !llvm::isa<Transition>(node.get()))
          return;

        auto t = std::static_pointer_cast<Transition>(node);
        auto preds = buildIncomingEdgePredicates(t);

        std::vector<z3::expr> pairExprs;
        for (size_t i = 0; i < preds.size(); ++i) {
          for (size_t j = i + 1; j < preds.size(); ++j) {
            pairExprs.push_back(preds[i] && preds[j]);
          }
        }

        result.push_back({t, std::move(pairExprs)});
      },
      [](const std::shared_ptr<Node> &, const std::shared_ptr<ArcLike> &,
         const std::monostate &) { return std::monostate{}; });

  return result;
}
std::vector<
    std::pair<std::shared_ptr<Petrinet::Transition>, std::vector<z3::expr>>>
Petrinet::getCountersAllSubsets() const {
  using CounterResult = std::vector<
      std::pair<std::shared_ptr<Transition>, std::vector<z3::expr>>>;

  static constexpr uint64_t kMaxSubsetCounters = 1000;
  static constexpr unsigned kMaxTupleSize = 6;

  auto boundedBinom = [](uint64_t n, uint64_t k, uint64_t cap) -> uint64_t {
    if (k > n)
      return 0;
    if (k == 0 || k == n)
      return 1;

    k = std::min(k, n - k);
    uint64_t result = 1;

    for (uint64_t i = 1; i <= k; ++i) {
      // result *= (n - k + i); result /= i;
      uint64_t mul = n - k + i;

      if (result > cap / mul)
        return cap + 1;

      result *= mul;
      result /= i;

      if (result > cap)
        return cap + 1;
    }

    return result;
  };

  auto countUpToKForTransition = [&](size_t n, unsigned k,
                                     uint64_t cap) -> uint64_t {
    uint64_t total = 0;
    unsigned lim = std::min<unsigned>(k, n);

    for (unsigned i = 1; i <= lim; ++i) {
      uint64_t c = boundedBinom(n, i, cap - total);
      if (c > cap - total)
        return cap + 1;
      total += c;
    }
    return total;
  };

  auto estimateTotalCounters = [&](unsigned k, uint64_t cap) -> uint64_t {
    uint64_t total = 0;
    bool overflowed = false;

    walkBackwardFromFinals<std::monostate>(
        [](const std::shared_ptr<Place> &) { return std::monostate{}; },
        [&](const std::shared_ptr<Node> &node, const std::monostate &) {
          if (overflowed || !node || !llvm::isa<Transition>(node.get()))
            return;

          auto t = std::static_pointer_cast<Transition>(node);
          auto preds = buildIncomingEdgePredicates(t);
          size_t n = preds.size();

          if (n == 0)
            return;

          uint64_t here = countUpToKForTransition(n, k, cap - total);
          if (here > cap - total) {
            overflowed = true;
            total = cap + 1;
            return;
          }

          total += here;
          if (total > cap) {
            overflowed = true;
            total = cap + 1;
          }
        },
        [](const std::shared_ptr<Node> &, const std::shared_ptr<ArcLike> &,
           const std::monostate &) { return std::monostate{}; });

    return total;
  };

  auto chooseTupleSize = [&]() -> unsigned {
    for (unsigned k = kMaxTupleSize; k >= 2; --k) {
      uint64_t estimate = estimateTotalCounters(k, kMaxSubsetCounters);
      if (estimate <= kMaxSubsetCounters)
        return k;
    }
    return 1;
  };

  auto buildKTuples = [&](const std::vector<z3::expr> &preds,
                          unsigned maxK) -> std::vector<z3::expr> {
    std::vector<z3::expr> out;
    const size_t n = preds.size();
    if (n == 0 || maxK == 0)
      return out;

    std::vector<z3::expr> cur;
    cur.reserve(maxK);

    std::function<void(size_t, unsigned)> dfs = [&](size_t start,
                                                    unsigned need) {
      if (need == 0) {
        out.push_back(mkAnd(cur));
        return;
      }

      for (size_t i = start; i + need <= n; ++i) {
        cur.push_back(preds[i]);
        dfs(i + 1, need - 1);
        cur.pop_back();
      }
    };

    for (unsigned sz = 1; sz <= std::min<unsigned>(maxK, n); ++sz)
      dfs(0, sz);

    return out;
  };

  unsigned chosenK = chooseTupleSize();

  llvm::errs() << "[PETRINET] getCountersAllSubsets: using tuple size "
               << chosenK << " under max counter budget " << kMaxSubsetCounters
               << "\n";

  CounterResult result;

  walkBackwardFromFinals<std::monostate>(
      [](const std::shared_ptr<Place> &) { return std::monostate{}; },
      [&](const std::shared_ptr<Node> &node, const std::monostate &) {
        if (!node || !llvm::isa<Transition>(node.get()))
          return;

        auto t = std::static_pointer_cast<Transition>(node);
        auto preds = buildIncomingEdgePredicates(t);

        auto tupleExprs = buildKTuples(preds, chosenK);
        result.push_back({t, std::move(tupleExprs)});
      },
      [](const std::shared_ptr<Node> &, const std::shared_ptr<ArcLike> &,
         const std::monostate &) { return std::monostate{}; });

  return result;
}

void Petrinet::writeTransitionToPlaceMatrixCSV(llvm::raw_ostream &os) const {
  std::vector<std::shared_ptr<Place>> places;
  std::vector<std::shared_ptr<Transition>> transitions;

  for (const auto &node : nodes) {
    if (!node)
      continue;

    if (auto *p = llvm::dyn_cast<Place>(node.get()))
      places.push_back(std::static_pointer_cast<Place>(node));
    else if (auto *t = llvm::dyn_cast<Transition>(node.get()))
      transitions.push_back(std::static_pointer_cast<Transition>(node));
  }

  std::sort(places.begin(), places.end(),
            [](const auto &a, const auto &b) { return a->id < b->id; });
  std::sort(transitions.begin(), transitions.end(),
            [](const auto &a, const auto &b) { return a->id < b->id; });

  std::unordered_map<unsigned, size_t> placeCol;
  std::unordered_map<unsigned, size_t> transitionRow;

  for (size_t i = 0; i < places.size(); ++i)
    placeCol[places[i]->id] = i;
  for (size_t i = 0; i < transitions.size(); ++i)
    transitionRow[transitions[i]->id] = i;

  std::vector<std::vector<int>> mat(transitions.size(),
                                    std::vector<int>(places.size(), 0));

  for (const auto &node : nodes) {
    if (!node)
      continue;

    if (auto *arc = llvm::dyn_cast<Arc>(node.get())) {
      auto *fromT = llvm::dyn_cast<Transition>(arc->from.get());
      auto *toP = llvm::dyn_cast<Place>(arc->to.get());
      if (!fromT || !toP)
        continue;

      auto rIt = transitionRow.find(fromT->id);
      auto cIt = placeCol.find(toP->id);
      if (rIt == transitionRow.end() || cIt == placeCol.end())
        continue;

      mat[rIt->second][cIt->second] = 1;
      continue;
    }

    if (auto *inh = llvm::dyn_cast<InhibitorArc>(node.get())) {
      auto *fromT = llvm::dyn_cast<Transition>(inh->from.get());
      auto *toP = llvm::dyn_cast<Place>(inh->to.get());
      if (!fromT || !toP)
        continue;

      auto rIt = transitionRow.find(fromT->id);
      auto cIt = placeCol.find(toP->id);
      if (rIt == transitionRow.end() || cIt == placeCol.end())
        continue;

      mat[rIt->second][cIt->second] = -1;
      continue;
    }
  }

  os << "transition_id";
  for (const auto &p : places)
    os << ",p" << p->id;
  os << "\n";

  for (size_t r = 0; r < transitions.size(); ++r) {
    os << "t" << transitions[r]->id;
    for (size_t c = 0; c < places.size(); ++c)
      os << "," << mat[r][c];
    os << "\n";
  }
}

void Petrinet::writePlaceToTransitionSlotMatrixCSV(
    llvm::raw_ostream &os) const {
  std::vector<std::shared_ptr<Place>> places;
  std::vector<std::shared_ptr<Transition>> transitions;

  for (const auto &node : nodes) {
    if (!node)
      continue;

    if (auto *p = llvm::dyn_cast<Place>(node.get()))
      places.push_back(std::static_pointer_cast<Place>(node));
    else if (auto *t = llvm::dyn_cast<Transition>(node.get()))
      transitions.push_back(std::static_pointer_cast<Transition>(node));
  }

  std::sort(places.begin(), places.end(),
            [](const auto &a, const auto &b) { return a->id < b->id; });
  std::sort(transitions.begin(), transitions.end(),
            [](const auto &a, const auto &b) { return a->id < b->id; });

  std::unordered_map<unsigned, size_t> placeRow;
  for (size_t i = 0; i < places.size(); ++i)
    placeRow[places[i]->id] = i;

  struct TransitionSlot {
    unsigned transitionId;
    unsigned slot;
  };

  std::vector<TransitionSlot> slots;
  std::unordered_map<unsigned, size_t> slotBaseCol;

  for (const auto &t : transitions) {
    slotBaseCol[t->id] = slots.size();
    for (unsigned s = 0; s < t->numIncoming; ++s)
      slots.push_back({t->id, s});
  }

  std::vector<std::vector<int>> mat(places.size(),
                                    std::vector<int>(slots.size(), 0));

  for (const auto &node : nodes) {
    if (!node)
      continue;

    int value = 0;
    Place *fromP = nullptr;
    Transition *toT = nullptr;
    unsigned slot = 0;

    if (auto *arc = llvm::dyn_cast<Arc>(node.get())) {
      fromP = llvm::dyn_cast<Place>(arc->from.get());
      toT = llvm::dyn_cast<Transition>(arc->to.get());
      if (!fromP || !toT)
        continue;
      value = 1;
      slot = arc->incomingIdx;
    } else if (auto *inh = llvm::dyn_cast<InhibitorArc>(node.get())) {
      fromP = llvm::dyn_cast<Place>(inh->from.get());
      toT = llvm::dyn_cast<Transition>(inh->to.get());
      if (!fromP || !toT)
        continue;
      value = -1;
      slot = inh->incomingIdx;
    } else {
      continue;
    }

    auto rIt = placeRow.find(fromP->id);
    auto baseIt = slotBaseCol.find(toT->id);
    if (rIt == placeRow.end() || baseIt == slotBaseCol.end())
      continue;

    size_t col = baseIt->second + slot;
    if (col >= slots.size())
      continue;

    mat[rIt->second][col] = value;
  }

  os << "place_id";
  for (const auto &slotInfo : slots)
    os << ",t" << slotInfo.transitionId << "_" << slotInfo.slot;
  os << "\n";

  for (size_t r = 0; r < places.size(); ++r) {
    os << "p" << places[r]->id;
    for (size_t c = 0; c < slots.size(); ++c)
      os << "," << mat[r][c];
    os << "\n";
  }
}

void Petrinet::writePlaceIdNameCSV(llvm::raw_ostream &os) const {
  std::vector<std::shared_ptr<Place>> places;

  for (const auto &node : nodes) {
    if (!node)
      continue;

    if (auto *p = llvm::dyn_cast<Place>(node.get()))
      places.push_back(std::static_pointer_cast<Place>(node));
  }

  std::sort(places.begin(), places.end(),
            [](const auto &a, const auto &b) { return a->id < b->id; });

  bool first = true;
  for (const auto &p : places) {
    if (!first)
      os << ",";
    first = false;
    os << "p" << p->id;
  }
  os << "\n";

  first = true;
  for (const auto &p : places) {
    if (!first)
      os << ",";
    first = false;
    os << p->getName();
  }
  os << "\n";
}