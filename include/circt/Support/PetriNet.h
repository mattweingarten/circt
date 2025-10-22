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
#include <string>
#include <set>  

namespace circt {
namespace pnet {

// class PetriNet::Node;
// class PetriNet::Arc;

// Naive implementation: Just hold a list of places, transitions, and arcs
class PetriNet {

public:
  class Arc;
  class Node;
  class NetElement {
  public:
    virtual std::string shortName() const = 0;
    virtual std::string
    toString() const = 0; // For now just equal to shortName();
    virtual void writeGraph(llvm::raw_ostream &os) const = 0;
  };

  // At some point, we need to hook this into the LLVM DotGraphTratis API, for
  // now we just do this ourselves because it is easier?
  class Node : public NetElement {
    using Arcs = llvm::SmallVector<std::shared_ptr<PetriNet::Arc>, 4>;

  public:
    // using ArcsRange = llvm::iterator_range<
    //     llvm::SmallVectorImpl<std::shared_ptr<Arc>>::iterator>;
    enum NodeType { Place, Transition };
    Node(NodeType kind) : kind(kind), node_id(++idCounter) {}
    NodeType getKind() const { return kind; }
    virtual uint64_t getId() const = 0;
    static bool classof(const Node *node) {
      return node->getKind() == NodeType::Transition;
    }
    uint64_t getNodeId() const { return node_id; }
    llvm::SmallVector<std::shared_ptr<PetriNet::Arc>, 4>
    getOutgoingArcs() const {
      return outgoingArcs;
    }
    llvm::SmallVector<std::shared_ptr<PetriNet::Arc>, 4>
    getIncomingArcs() const {
      return incomingArcs;
    }
    void addIncomingArc(std::shared_ptr<PetriNet::Arc> arc) {
      incomingArcs.push_back(arc);
    }
    void addOutgoingArc(std::shared_ptr<PetriNet::Arc> arc) {
      outgoingArcs.push_back(arc);
    }

    void addOp(Operation *op) { operations.insert(op); }

  llvm::SmallVector<std::shared_ptr<PetriNet::Arc>, 4> incomingArcs;
  llvm::SmallVector<std::shared_ptr<PetriNet::Arc>, 4> outgoingArcs;

  private:
    static uint64_t idCounter;
    NodeType kind;
    uint64_t node_id;
    llvm::SmallPtrSet<Operation *, 4>
        operations; // Debug data to track where this Node is coming from
  };

  // A place encodes a state of particular register or buffer. For now we only
  // model registers.

  // TODO: Splitting a single register into multiple states? For example for a
  // processor with CoreWidth > 1?

  class Place final : public Node {

    // TODO: How do we encode the state? Z3 API? Custom API? --> this might be
    // better, since we want to maybe point this to specific operations in the
    // MLIR.
  public:
    Place(Operation *op, std::string name, bool sink = false)
        : Node(NodeType::Place), id(++idCounter), op(op), name(name),
          sink(sink) {}
    static std::shared_ptr<Place> createRegisterPlace(Operation *op);
    static std::shared_ptr<Place> createSinkPlace(Operation *op);
    std::string shortName() const override { return "p" + id; }
    std::string toString() const override;
    void writeGraph(llvm::raw_ostream &os) const override;

    uint64_t getId() const override { return id; }

    class Expression {};
    static bool classof(const Node *node) {
      return node->getKind() == NodeType::Place;
    }

  private:
    static uint64_t idCounter;
    uint64_t id;      // the id of the register
    Operation *op;    // Points to register in the MLIR
    std::string name; // the name of the register
    bool sink;
    std::unique_ptr<Expression>
        state; // if this register has a token, this means this state is true
  };

  //
  class Transition final : public Node {
  public:
    Transition(Operation *op) : Node(NodeType::Transition), id(++idCounter) {
      name = std::move(shortName() );
      operations.insert(op);
    }
    Transition(hw::PortInfo portInfo) : Node(NodeType::Transition), id(++idCounter) {
      name = std::move(shortName());
    }
    std::string toString() const override;
    std::string shortName() const override { return "t" + id; }

    void writeGraph(llvm::raw_ostream &os) const override;
    uint64_t getId() const override { return id; }

    static bool classof(const Node *node) {
      return node->getKind() == NodeType::Transition;
    }

    llvm::DenseSet<mlir::Operation *>
        operations; // points to operations in the MLIR?
  private:
    static uint64_t idCounter;
    uint64_t id;      // the id of the transition
    std::string name; // the name of the transition?
    class Guard {};
  };

  // An arc connects a place to a transition, and a transition to a place.
  class Arc : virtual public NetElement {
  public:
    enum ArcType { Inhibitor, Normal };
    Arc(ArcType kind, std::shared_ptr<Node> source,
        std::shared_ptr<Node> target)
        : kind(kind), source(source), target(target) {}
    std::shared_ptr<Node> getSource() const { return source; };
    std::shared_ptr<Node> getTarget() const { return target; };
    void addOp(Operation *op) { operations.insert(op); }
    void setSource(std::shared_ptr<Node> source) { this->source = source; }
    void setTarget(std::shared_ptr<Node> target) { this->target = target; };
    // std::string shortName() const override { return "a" }
    // std::string toString() const override { return shortName(); }
    ArcType getKind() const { return kind; };
    std::string getShortString() const {
      if(operations.size() == 0){
        return "";
      }
      std::string buf;
      llvm::raw_string_ostream ss(buf);
    
      if (operations.size() > 0) {
        // llvm::raw_os_ostream ros(ss);
        (*(operations.begin()))->print(ss);
      }
      ss.flush();
      std::replace(buf.begin(), buf.end(), '"', '\'');
    
      return buf;
    }
    
    size_t incomingNumber;

  private:
    const ArcType kind;

  protected:
    std::shared_ptr<Node> source;
    std::shared_ptr<Node> target;
    llvm::SmallPtrSet<Operation *, 4>
        operations; // Debug data to track where this Arc is coming from
  };

  class InhibitorArc final : virtual public Arc {
  public:
    InhibitorArc() = delete;
    InhibitorArc(std::shared_ptr<Node> source, std::shared_ptr<Node> target)
        : Arc(ArcType::Inhibitor, source, target) {}
    std::string shortName() const override {
      return source->shortName() + " --o " + target->shortName();
    }
    std::string toString() const override { return shortName(); }
    void writeGraph(llvm::raw_ostream &os) const override;
    static bool classof(const Arc *arc) {
      return arc->getKind() == ArcType::Inhibitor;
    }
  };

  class NormalArc final : virtual public Arc {

  public:
    NormalArc() = delete;
    NormalArc(std::shared_ptr<Node> source, std::shared_ptr<Node> target)
        : Arc(ArcType::Normal, source, target) {}
    std::string shortName() const override {
      return source->shortName() + " --> " + target->shortName();
    }
    std::string toString() const override { return shortName(); }
    void writeGraph(llvm::raw_ostream &os) const override;
    static bool classof(const Arc *arc) {
      return arc->getKind() == ArcType::Normal;
    }
  };

public:
  static std::unique_ptr<PetriNet> buildGraphFromModule(
      circt::hw::HWModuleOp module,
      const std::string &progressSignal /*Make multiple signals possible?*/);
  void
  mergeInto(std::unique_ptr<PetriNet> other); // Todo: determine merge point?
  LogicalResult addPlaceFromRegister(Operation *op);
  void addPlace(std::shared_ptr<Place> place, Operation *op) {
    places.push_back(place);
    place->addOp(op);
    opToPlace.insert({op, place});
  }
  std::shared_ptr<Place> createAndAddPlace(Operation *op) {
    auto place = Place::createRegisterPlace(op);
    addPlace(place, op);
    return place;
  }
  void addTransition(std::shared_ptr<Transition> transition, Operation *op) {
    transitions.push_back(transition);
    if (op){
      transition->addOp(op);
      opToTransition.insert({op, transition});
    }
  }

  void addTransition(std::shared_ptr<Transition> transition, Value value) {
    transitions.push_back(transition);
    valueToTransition.insert({value, transition});
    
  }

  void removeArc(const std::shared_ptr<Arc> &arc) {
    auto src = arc->getSource();
    auto dst = arc->getTarget();
  
    // ---- source: outgoing ----
    {
      llvm::SmallVector<unsigned, 4> idx;
      for (unsigned i = 0, e = src->outgoingArcs.size(); i < e; ++i)
        if (src->outgoingArcs[i] == arc) idx.push_back(i);
      for (unsigned n = idx.size(); n > 0; --n)
        src->outgoingArcs.erase(src->outgoingArcs.begin() + idx[n - 1]);
    }
  
    // ---- target: incoming ----
    {
      llvm::SmallVector<unsigned, 4> idx;
      for (unsigned i = 0, e = dst->incomingArcs.size(); i < e; ++i)
        if (dst->incomingArcs[i] == arc) idx.push_back(i);
      for (unsigned n = idx.size(); n > 0; --n)
        dst->incomingArcs.erase(dst->incomingArcs.begin() + idx[n - 1]);
    }
  
    // ---- global list ----
    {
      llvm::SmallVector<unsigned, 4> idx;
      for (unsigned i = 0, e = arcs.size(); i < e; ++i)
        if (arcs[i] == arc) idx.push_back(i);
      for (unsigned n = idx.size(); n > 0; --n)
        arcs.erase(arcs.begin() + idx[n - 1]);
    }
  }
  void removeTransition(std::shared_ptr<Transition> transition) {
    for (auto t : transitions){
      if (t == transition){
        transitions.erase(std::remove(transitions.begin(), transitions.end(), t), transitions.end());
        return;
      }
    }
    // transitions.erase(std::remove(transitions.begin(), transitions.end(), transition), transitions.begin());
  }

  std::shared_ptr<Transition> createAndAddTransition(Operation *op) {
    auto transition = std::make_shared<Transition>(op);
    addTransition(transition, op);
    return transition;
  }

  std::shared_ptr<Transition> createAndAddTransition(circt::hw::HWModuleOp module, hw::PortInfo portInfo) {
    auto transition = std::make_shared<Transition>(portInfo);
    auto block = module.getBodyBlock();
    auto blockArgument = block->getArgument(portInfo.argNum);
    addTransition(transition, blockArgument);
    if (portInfo.isInput()){
      inNodes.insert(transition);
    } else {
      outNodes.insert(transition);
    }
    return transition;
  }

  void addNormalArc(std::shared_ptr<Node> src, std::shared_ptr<Node> target,
                    std::optional<Operation *> op = std::nullopt) {
    ArcKeyT key = {src->getNodeId(), target->getNodeId()};
    auto it = arcMap.find(key);
    if (it != arcMap.end()) {
      if (op)
        it->second->addOp(*op);
      return;
    }
    auto arc = std::make_shared<NormalArc>(src, target);
    arc->incomingNumber = target->getIncomingArcs().size();
    src->addOutgoingArc(arc);
    target->addIncomingArc(arc);
    arcs.push_back(arc);
    if (op)
      arc->addOp(*op);
    arcMap.insert({key, arc});
  }
  void initializeGraph();

  std::string toString() const;
  std::string shortName() const;
  void writeGraph(llvm::raw_ostream &os,
                  int levels = -1) const; // TODO: implement this

  llvm::DenseMap<Operation *, std::shared_ptr<Place>> opToPlace;
  llvm::DenseMap<Operation *, std::shared_ptr<Transition>> opToTransition;
  llvm::DenseMap<Value, std::shared_ptr<Place>> valueToPlace;
  llvm::DenseMap<Value, std::shared_ptr<Transition>> valueToTransition;

  bool isTransition(Operation *op) { return opToTransition.count(op) > 0; }
  bool isPlace(Operation *op) { return opToPlace.count(op) > 0; }
  std::optional<std::shared_ptr<Node>> getNode(Operation *op) {
    // Transition has priority over place?
    if (!op){
      return std::nullopt;
    }
    if (isTransition(op))
      return opToTransition[op];
    if (isPlace(op))
      return opToPlace[op];
    return std::nullopt;
  }

  std::optional<std::shared_ptr<Node>> getNode(Value op) {
    if (valueToTransition.count(op) > 0)
      return valueToTransition[op];
    if (valueToPlace.count(op) > 0)
      return valueToPlace[op];
    
    return getNode(op.getDefiningOp());
  }

  using ArcKeyT = std::pair<uint64_t, uint64_t>;
  std::vector<std::shared_ptr<Place>> sinks;
  std::vector<std::shared_ptr<Place>> places;
  std::vector<std::shared_ptr<Transition>> transitions;
  std::vector<std::shared_ptr<Arc>> arcs;
  llvm::DenseMap<ArcKeyT, std::shared_ptr<Arc>> arcMap;
  
  std::set<std::shared_ptr<Node>> inNodes;
  std::set<std::shared_ptr<Node>> outNodes;

  class Hierarchy {
    struct Node {
      std::vector<const NetElement *> elements;
      llvm::SmallVector<std::unique_ptr<Node>, 4> children;
    };
  };
};

} // namespace pnet
} // namespace circt

#endif // CIRCT_PETRINET_H