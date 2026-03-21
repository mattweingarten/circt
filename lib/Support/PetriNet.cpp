//===- PetriNet.cpp - Petri Net -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/PetriNet.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "petri"

using namespace circt;
using namespace pnet;

using NodeType = circt::pnet::PetriNet::Node::NodeType;
using NodeID = PetriNet::NodeID;

NodeID PetriNet::Node::idCounter = PetriNet::NULL_ID;

// TODO: write graphs

void PetriNet::handleInsertOpMapping(NodeID id, mlir::Operation *op) {
  OpToNode[op] = id;
  auto &vec = nodeToOps.try_emplace(id).first->second;
  vec.push_back(op);
}

NodeID PetriNet::createAndAddPlace(mlir::Operation *Op) {
  std::unique_ptr<Node> node = std::make_unique<Place>();
  NodeID id = node->getId();
  nodes[id] = std::move(node);
  handleInsertOpMapping(id, Op);
  return id;
}
NodeID PetriNet::createAndAddTransition(mlir::Operation *Op) {
  std::unique_ptr<Node> node = std::make_unique<Transition>();
  NodeID id = node->getId();
  nodes[id] = std::move(node);
  handleInsertOpMapping(id, Op);
  return id;
}

void PetriNet::activate(NodeID from, NodeID to) {
  arcs.push_back(std::make_unique<ActivatorArc>(from, to));
}
void PetriNet::inhibit(NodeID from, NodeID to) {
  arcs.push_back(std::make_unique<InhibitorArc>(from, to));
}

// TODO:
//  - handle IO of modules
void PetriNet::writeGraph(llvm::raw_ostream &os) const {
  os << "digraph PetriNet {\n  rankdir = LR;\n";

  for (const auto &kv : nodes) {
    kv.second->writeGraph(os);
  }

  // If you want to keep the filtered loops, that's fine too:
  for (const auto &p : nodesOf<Place>()) {
    llvm::cast<Place>(p.second.get())->writeGraph(os);
  }

  for (const auto &t : nodesOf<Transition>()) {
    llvm::cast<Transition>(t.second.get())->writeGraph(os);
  }

  for (const auto &a : arcs) {
    a->writeGraph(os);
  }

  os << "}\n";
}

void PetriNet::Place::writeGraph(llvm::raw_ostream &os) const {
  os << "  p" << getId() << " [shape=circle, width=1.25, label=\"\", xlabel=\""
     << "p" + std::to_string(getId()) << "];\n";
  // os << "//" << *op;
  os << "\n";
}

void PetriNet::Transition::writeGraph(llvm::raw_ostream &os) const {
  os << "  t" << getId() << " [shape=record, label=\"";
  // for (size_t i = 0; i < n; ++i) {
  //   os << "<f" << i << ">";
  //   if (i != n - 1) {
  //     os << "|";
  //   }
  // }
  os << "\", xlabel=\""
     << "t" << getId() << "\", width=0, height=" << 1 << "];\n";
  // for (auto *const op : operations) {
  //   os << "//";
  //   os << *op;
  //   os << "\n";
  // }
}

void PetriNet::ActivatorArc::writeGraph(llvm::raw_ostream &os) const { return; }

void PetriNet::InhibitorArc::writeGraph(llvm::raw_ostream &os) const { return; }
// TODO: destructors