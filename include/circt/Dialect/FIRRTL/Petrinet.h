#ifndef CIRCT_DIALECT_FIRRTL_Petrinet_H
#define CIRCT_DIALECT_FIRRTL_Petrinet_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <z3++.h>

class Petrinet {
public:
  class ArcLike;
  class Node;
  Petrinet() = default;
  ~Petrinet() = default;

  static unsigned transitionID;
  static unsigned arcID;
  static unsigned nodeID;

  class GraphElement {
  public:
    enum Kind {
      GE_GraphElement,
      GE_Node,
      GE_ArcLike,
      GE_Arc,
      GE_InhibitorArc,
      GE_Place,
      GE_Transition
    };

    std::string name;

  private:
    Kind kind;

  public:
    explicit GraphElement(Kind k, std::string name)
        : kind(k), name(std::move(name)) {}
    virtual ~GraphElement() = default;

    virtual void writeGraph(llvm::raw_ostream &os) const = 0;
    Kind getKind() const { return kind; }
    const std::string &getName() const { return name; }
    void setName(std::string newName) { name = std::move(newName); }

    static bool classof(const GraphElement *ge) { return ge != nullptr; }
  };

  class ArcLike : public GraphElement {
  public:
    explicit ArcLike(Kind k, std::string name)
        : GraphElement(k, std::move(name)) {}
    ~ArcLike() override = default;

    static bool classof(const GraphElement *ge) {
      if (!ge)
        return false;
      switch (ge->getKind()) {
      case GE_ArcLike:
      case GE_Arc:
      case GE_InhibitorArc:
        return true;
      default:
        return false;
      }
    }

    std::shared_ptr<Node> from;
    std::shared_ptr<Node> to;
    unsigned incomingIdx;
  };

  class Node : public GraphElement {
  public:
    explicit Node(Kind k, std::string name)
        : GraphElement(k, std::move(name)), id(nodeID++) {}
    ~Node() override = default;

    static bool classof(const GraphElement *ge) {
      if (!ge)
        return false;
      switch (ge->getKind()) {
      case GE_Node:
      case GE_Place:
      case GE_Transition:
        return true;
      default:
        return false;
      }
    }

    unsigned getId() const { return id; }

    unsigned numIncoming = 0;
    int rank = 0;
    unsigned depth = 0;
    const unsigned id;

    std::vector<std::shared_ptr<ArcLike>> outgoingEdges;
    std::vector<std::shared_ptr<ArcLike>> incomingEdges;
  };

  class Arc : public ArcLike {
  public:
    explicit Arc(std::string name) : ArcLike(GE_Arc, std::move(name)) {}
    ~Arc() override = default;

    static bool classof(const GraphElement *ge) {
      return ge && ge->getKind() == GE_Arc;
    }
    void writeGraph(llvm::raw_ostream &os) const;
  };

  class InhibitorArc : public ArcLike {
  public:
    explicit InhibitorArc(std::string name)
        : ArcLike(GE_InhibitorArc, std::move(name)) {}
    ~InhibitorArc() override = default;

    static bool classof(const GraphElement *ge) {
      return ge && ge->getKind() == GE_InhibitorArc;
    }
    void writeGraph(llvm::raw_ostream &os) const;
  };

  class Place : public Node {
  public:
    Place(std::string name, z3::expr expr)
        : Node(GE_Place, std::move(name)), expr(expr) {}

    ~Place() override = default;

    static bool classof(const GraphElement *ge) {
      return ge && ge->getKind() == GE_Place;
    }

    void writeGraph(llvm::raw_ostream &os) const;
    
    z3::expr expr;
    bool final = false;
  };

  class Transition : public Node {
  public:
    explicit Transition(std::string name)
        : Node(GE_Transition, std::move(name)) {}
    ~Transition() override = default;

    static bool classof(const GraphElement *ge) {
      return ge && ge->getKind() == GE_Transition;
    }
    void writeGraph(llvm::raw_ostream &os) const;
  };
  template <typename T>
  auto getFilteredAs() {
    auto range = llvm::make_filter_range(
        nodes, [](const std::shared_ptr<GraphElement> &n) {
          return llvm::isa<T>(n.get());
        });

    return llvm::map_range(range,
                           [](const std::shared_ptr<GraphElement> &n) -> T * {
                             return llvm::cast<T>(n.get());
                           });
  }

  std::shared_ptr<GraphElement> getNodeByName(const std::string &name) const {
    for (const auto &node : nodes) {
      if (node && node->getName() == name)
        return node;
    }
    return nullptr;
  }

  template <typename T>
  std::shared_ptr<T> getNodeByNameAs(const std::string &name) const {
    auto node = getNodeByName(name);
    if (!node)
      return nullptr;
    return llvm::dyn_cast<T>(node);
  }

  auto getPlaces() { return getFilteredAs<Place>(); }
  auto getTransitions() { return getFilteredAs<Transition>(); }
  auto getArcs() { return getFilteredAs<Arc>(); }
  auto getInhibitorArcs() { return getFilteredAs<InhibitorArc>(); }
  auto getNodes() { return getFilteredAs<Node>(); }

  static std::unique_ptr<Petrinet> createFromZ3(const z3::expr &e,
                                                const std::string &name);

  void extendGraphWithExpression(const z3::expr &e, const std::string &name);

  void writeGraph(llvm::raw_ostream &os) const;

  void writeTransitionToPlaceMatrixCSV(llvm::raw_ostream &os) const;

  void writePlaceToTransitionSlotMatrixCSV(llvm::raw_ostream &os) const;

  void writePlaceIdNameCSV(llvm::raw_ostream &os) const;

  void setDepth();

  template <typename StateT, typename MakeInitialStateFn, typename VisitFn,
            typename PropagateFn>
  void walkBackwardFromFinals(MakeInitialStateFn &&makeInitialState,
                              VisitFn &&visit, PropagateFn &&propagate) const {
    struct WorkItem {
      std::shared_ptr<Node> node;
      StateT state;
    };

    llvm::SmallPtrSet<const Node *, 32> visited;
    std::vector<WorkItem> stack;

    for (const auto &ge : nodes) {
      if (!ge || !llvm::isa<Place>(ge.get()))
        continue;

      auto place = std::static_pointer_cast<Place>(ge);
      if (!place->final)
        continue;

      if (!visited.insert(place.get()).second)
        continue;

      stack.push_back({place, makeInitialState(place)});
    }

    while (!stack.empty()) {
      WorkItem item = std::move(stack.back());
      stack.pop_back();

      visit(item.node, item.state);

      for (const auto &arc : item.node->incomingEdges) {
        if (!arc || !arc->from)
          continue;

        auto pred = arc->from;

        if (!visited.insert(pred.get()).second)
          continue;

        stack.push_back({pred, propagate(pred, arc, item.state)});
      }
    }
  }

  std::vector<
      std::pair<std::shared_ptr<Petrinet::Transition>, std::vector<z3::expr>>>
  getCountersNaive() const;

  std::vector<
      std::pair<std::shared_ptr<Petrinet::Transition>, std::vector<z3::expr>>>
  getCountersAllSubsets() const;

  std::vector<
      std::pair<std::shared_ptr<Petrinet::Transition>, std::vector<z3::expr>>>
  getCountersPairs() const;

private:
  void integrateExpression(const z3::expr &e, const std::string &name,
                           bool markRootFinal);

  std::shared_ptr<Petrinet::Place> getOrCreatePlace(
      const std::string &placeName,
      std::unordered_map<std::string, std::shared_ptr<Place>> &nameToGraph,
      int rank, z3::expr expr);

  std::shared_ptr<Transition> makeTransition(const std::string &prefix,
                                             int rank);

  void addArc(const std::shared_ptr<Node> &from,
              const std::shared_ptr<Node> &to, const std::string &prefix);

  void addInhibitorArc(const std::shared_ptr<Node> &from,
                       const std::shared_ptr<Node> &to,
                       const std::string &prefix);

  std::vector<std::shared_ptr<GraphElement>> nodes;
  std::unordered_map<std::string, std::shared_ptr<Place>> nameToGraph;
};

#endif // CIRCT_DIALECT_FIRRTL_Petrinet_H