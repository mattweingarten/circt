//===- PetriNet.h - Petri Net -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_PETRINET_H
#define CIRCT_PETRINET_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DOTGraphTraits.h"

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Operation.h"

#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>

namespace circt {
namespace pnet {

// template <typename PlaceT, typename TransitionT> //TODO: Templateize?
class PetriNet {
public:
  using NodeID = uint64_t;
  static constexpr NodeID NULL_ID = 0;

  class PetriNetElement {
  public:
    virtual ~PetriNetElement() = default;
    virtual std::string shortName() const = 0;
    virtual void writeGraph(llvm::raw_ostream &os) const = 0;
  };

  class Node : public PetriNetElement {
  public:
    virtual ~Node() = default;
    enum NodeType { Place, Transition };
    Node(NodeType kind) : kind(kind), id(++idCounter) {}
    NodeType getKind() const { return kind; }
    static bool classof(const Node *node) {
      return node->getKind() == NodeType::Transition;
    }
    NodeID getId() const { return id; }

  protected:
    NodeID id;

  private:
    static NodeID idCounter;
    NodeType kind;
  };

  class Place final : public Node {
  public:
    Place() : Node(PetriNet::Node::NodeType::Place) {}
    std::string shortName() const override { return "p" + id; }
    void writeGraph(llvm::raw_ostream &os) const override;

    static bool classof(const Node *node) {
      return node->getKind() == PetriNet::Node::NodeType::Place;
    }

  private:
    // PlaceT value; //TODO: Templateize?
  };

  class Transition final : public Node {
  public:
    Transition() : Node(PetriNet::Node::NodeType::Transition) {}
    std::string shortName() const override { return "t" + id; }
    void writeGraph(llvm::raw_ostream &os) const override;

    static bool classof(const Node *node) {
      return node->getKind() == PetriNet::Node::NodeType::Transition;
    }

  private:
    // TransitionT value; //TODO: Templateize?
  };

  class Arc : virtual public PetriNetElement {
  public:
    enum ArcType { Inhibitor, Activator };
    Arc(ArcType kind, NodeID from, NodeID to)
        : kind(kind), from(from), to(to) {}
    virtual ~Arc() = default;
    ArcType getKind() const { return kind; };
    NodeID getFrom() const { return from; }
    NodeID getTo() const { return to; }

  private:
    ArcType kind;
    NodeID from;
    NodeID to;
  };

  class ActivatorArc : public Arc {
  public:
    ActivatorArc(NodeID from, NodeID to)
        : Arc(Arc::ArcType::Activator, from, to) {}

    // TODO: FIX
    std::string shortName() const override {
      return "act_" + std::to_string(getFrom()) + "_to_" +
             std::to_string(getTo());
    }
    static bool classof(const Arc *node) {
      return node->getKind() == Arc::ArcType::Activator;
    }
    void writeGraph(llvm::raw_ostream &os) const override;
  };

  class InhibitorArc : public Arc {
  public:
    InhibitorArc(NodeID from, NodeID to)
        : Arc(Arc::ArcType::Inhibitor, from, to) {}
    // TODO: FIX
    std::string shortName() const override {
      return "act_" + std::to_string(getFrom()) + "_to_" +
             std::to_string(getTo());
    }

    static bool classof(const Arc *node) {
      return node->getKind() == Arc::ArcType::Inhibitor;
    }
    void writeGraph(llvm::raw_ostream &os) const override;
  };

  const Node *getNode(NodeID id) { return nodes.at(id).get(); }

  // Get nodes of a specific type, i.e
  //  for (auto &e : nodesOf<Place>())..
  template <typename T>
  auto nodesOf() const {
    return llvm::make_filter_range(
        nodes, [](auto &kv) { return llvm::isa<T>(kv.second.get()); });
  }

  // For now this does not play well with inhibitory arcs. That is fine for now.
  template <typename T>
  auto arcsOf() {
    return llvm::make_filter_range(nodes, [](const std::unique_ptr<Arc> &e) {
      return llvm::isa<T>(e.get());
    });
  }

  // Get a list of all arcs comfing from a specific node.
  auto arcsFrom(NodeID fromID) const {
    return llvm::make_filter_range(arcs,
                                   [fromID](const std::unique_ptr<Arc> &a) {
                                     return a->getFrom() == fromID;
                                   });
  }

  // Gets a list of all arcs going to a specific node.
  auto arcsTo(NodeID toID) const {
    return llvm::make_filter_range(arcs, [toID](const std::unique_ptr<Arc> &a) {
      return a->getTo() == toID;
    });
  }

  NodeID createAndAddPlace(mlir::Operation *Op);
  NodeID createAndAddTransition(mlir::Operation *Op);
  void activate(NodeID from, NodeID to);
  void inhibit(NodeID from, NodeID to);
  void writeGraph(llvm::raw_ostream &os) const;

  // TODO: merge node?

private:
  std::vector<std::unique_ptr<Arc>> arcs;
  std::map<NodeID, std::unique_ptr<Node>> nodes;
  std::map<NodeID, std::vector<mlir::Operation *>> nodeToOps;
  std::map<mlir::Operation *, NodeID> OpToNode;

  void handleInsertOpMapping(NodeID id, mlir::Operation *op);
};
} // namespace pnet
} // namespace circt

#endif // CIRCT_PETRINET_H