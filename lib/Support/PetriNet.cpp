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
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace hw;
using namespace pnet;

uint64_t PetriNet::Place::idCounter = 0;
uint64_t PetriNet::Transition::idCounter = 0;
uint64_t PetriNet::Node::idCounter = 0;

// firmem? fifo?
static bool isPlaceOp(Operation *op) {
  return isa<::circt::seq::CompRegOp>(op) ||
         isa<::circt::seq::CompRegClockEnabledOp>(op) ||
         isa<::circt::seq::FirRegOp>(op) || isa<::circt::seq::ShiftRegOp>(op);
}

static bool isTransitionOp(Operation *op) {
  return isa<::circt::comb::AddOp>(op) || isa<::circt::comb::AndOp>(op) ||
         isa<::circt::comb::ConcatOp>(op) || isa<::circt::comb::DivSOp>(op) ||
         isa<::circt::comb::DivUOp>(op) || isa<::circt::comb::ExtractOp>(op) ||
         isa<::circt::comb::ICmpOp>(op) || isa<::circt::comb::ModSOp>(op) ||
         isa<::circt::comb::MulOp>(op) || isa<::circt::comb::MuxOp>(op) ||
         isa<::circt::comb::OrOp>(op) || isa<::circt::comb::ParityOp>(op) ||
         isa<::circt::comb::ReplicateOp>(op) || isa<::circt::comb::ShlOp>(op) ||
         isa<::circt::comb::ShrSOp>(op) || isa<::circt::comb::ShrUOp>(op) ||
         isa<::circt::comb::SubOp>(op) ||
         isa<::circt::comb::TruthTableOp>(op) || isa<::circt::comb::XorOp>(op);
}

// HWOp --> InstanceOp
// We can safely ingore

// Remove this --> Deprecated?
static std::string getNameFromPlaceOp(Operation *op) {
  assert(isPlaceOp(op) && "Operation is not a place operation");
  if (auto reg = dyn_cast<::circt::seq::CompRegOp>(op)) {
    return reg.getName() ? reg.getName()->str() : "";
  } else if (auto reg = dyn_cast<::circt::seq::CompRegClockEnabledOp>(op)) {
    return reg.getName() ? reg.getName()->str() : "";
  } else if (auto reg = dyn_cast<::circt::seq::FirRegOp>(op)) {
    return reg.getName().str();
  } else if (auto reg = dyn_cast<::circt::seq::ShiftRegOp>(op)) {
    return reg.getName() ? reg.getName()->str() : "";
  } else {
    std::string s;
    llvm::raw_string_ostream ss(s);
    op->print(ss);
    return s;
  }
}

void PetriNet::initializeGraph() {
  // llvm::errs() << "initializing graph" << "\n";
}

// TODO: might not need this?
void PetriNet::mergeInto(std::unique_ptr<PetriNet> other) {
  llvm::errs() << "merging graph" << "\n";
}

static std::optional<std::string> getNameFromOp(Operation *op) {
  auto name_hint = op->getAttr("sv.namehint");
  if (name_hint) {
    return mlir::cast<StringAttr>(name_hint).str();
  }

  auto name_attr = op->getAttr("name");
  if (name_attr) {
    return mlir::cast<StringAttr>(name_attr).str();
  }
  return std::nullopt;
}

static bool doesNameMatch(Operation *op, const std::string &name) {
  std::optional<std::string> op_name = getNameFromOp(op);
  if (!op_name) {
    return false;
  } else if (*op_name == name) {
    return true;
  } else {
    return false;
  }
}

static std::unique_ptr<PetriNet>
backwardsGraphBuildPass(const llvm::SmallPtrSet<Operation *, 4> &progressOps) {
  auto petriNet = std::make_unique<PetriNet>();
  petriNet->initializeGraph();
  // llvm::errs() << "\nbackwards graph build pass" << "\n";
  for (Operation *op : progressOps) {
    auto sink = PetriNet::Place::createSinkPlace(op);
    petriNet->addPlace(sink, op);
    petriNet->sinks.push_back(sink);
    llvm::SmallPtrSet<Operation *, 16> visited;
    llvm::SmallVector<Operation *, 16> worklist;
    worklist.push_back(op);

    while (!worklist.empty()) {
      // llvm::errs() << "-------------------------" << "\n";
      Operation *curr = worklist.pop_back_val();
      visited.insert(curr);
      auto curr_node = petriNet->getNode(curr);
      bool curr_is_transition = petriNet->isTransition(curr);
      bool curr_is_place = petriNet->isPlace(curr);
      if (curr_is_transition && curr_is_place) {
        // llvm::errs() << "Node cannot be place and transition "
        //              << curr->getName() << "\n";
        // curr->print(llvm::errs());
        // llvm::errs() << "\n";
        // assert(false && "Node cannot be place and transition!");
      }
      if (!curr_node) {
        // llvm::errs() << "Node not found for operation: " << curr->getName()
        //              << "\n";
        // curr->print(llvm::errs());
        // llvm::errs() << "\n";
        continue;
      }
      llvm::SmallVector<Value, 4> operands;

      // only take the first operand if is register.
      if (isPlaceOp(curr) && curr->getNumOperands() > 0) {
        operands.push_back(curr->getOperand(0));
      } else {
        for (Value operand : curr->getOperands()) {
          operands.push_back(operand);
        }
      }
      for (Value operand : operands) {
        if (Operation *p = operand.getDefiningOp()) {
          if (visited.count(p) == 0) {
            worklist.push_back(p);
          }

          llvm::errs() << "Curr: " << curr->getName();
          curr->print(llvm::errs());
          llvm::errs() << " | ";
          llvm::errs() << "P: ";
          p->print(llvm::errs());
          llvm::errs() << "\n";
          auto existing_node = petriNet->getNode(p);
          if (existing_node && existing_node != curr_node) {
            petriNet->addNormalArc(existing_node, curr_node);
            // llvm::errs() << "Node already exists" << "\n";
            // curr->print(llvm::errs());
            // llvm::errs() << "\n";
            continue;
          }
          if (!isPlaceOp(p) && !isTransitionOp(p)) {
            continue;
          }
          // Main handling and building of graph:
          if (curr_is_transition &&
              isPlaceOp(p)) { // P is place, create new tranisiton and add arc
                              // to curr place
            auto place = PetriNet::Place::createRegisterPlace(p);
            petriNet->addPlace(place, p);
            petriNet->addNormalArc(place, curr_node);
            // llvm::errs() << "Adding place, case 1 " << "\n";
            // p->print(llvm::errs());
            // llvm::errs() << "\n";

          } else if (curr_is_transition &&
                     !isPlaceOp(p)) { // P is transition, add the operation to
                                      // the exisiting transition. We do not
                                      // need to create a new transition
            std::shared_ptr<PetriNet::Transition> current_transition =
                petriNet->opToTransition[curr];
            current_transition->operations.insert(p);
            petriNet->opToTransition.insert({p, current_transition});
            // llvm::errs() << "Adding to exisiting transition, case2 " << "\n";
            // p->print(llvm::errs());
            // llvm::errs() << "\n";
          } else if (curr_is_place &&
                     !isPlaceOp(p)) { // P is transition, curr is plac --> add
            auto transition = std::make_shared<PetriNet::Transition>(p);
            petriNet->transitions.push_back(transition);
            petriNet->opToTransition.insert({p, transition});
            petriNet->addNormalArc(transition, curr_node);
            // llvm::errs() << "Adding transition, case 3 " << "\n";
            // llvm::errs() << "Adding transition" << "\n";
            // llvm::errs() << "Adding transition: " << transition->shortName()
            //              << "\n";
          } else if (curr_is_place &&
                     isPlaceOp(p)) { // P is transition, so is current. We need
                                     // to add a new dummy transition in between
            auto place = PetriNet::Place::createRegisterPlace(p);
            auto transition = std::make_shared<PetriNet::Transition>(p);
            petriNet->addNormalArc(transition, curr_node);
            petriNet->addNormalArc(place, transition);
            petriNet->opToTransition.insert({p, transition});
            petriNet->addPlace(place, p);
            petriNet->transitions.push_back(
                transition); // do not add mapping of p
            // llvm::errs() << "Adding place, case 4 " << "\n";
            // p->print(llvm::errs());
            // llvm::errs() << "\n";
          } else {
            llvm::errs() << "Panic! " << "\n";
            p->print(llvm::errs());
            llvm::errs() << "\n";
          }
        } else {
          auto blockArg = operand.cast<BlockArgument>();
          llvm::outs() << "  - Operand produced by Block argument, number "
                       << blockArg.getArgNumber()
                       << "stopping analysis for now!\n";
        }
      }
    }
  }

  // llvm::errs() << "Finished backwards graph build pass" << "\n";
  // llvm::errs() << "Number of places: " << petriNet->places.size() << "\n";
  // llvm::errs() << "Number of transitions" << petriNet->transitions.size()
  //              << "\n";
  // llvm::errs() << "Number of arcs: " << petriNet->arcs.size() << "\n";
  return petriNet;
}

std::unique_ptr<PetriNet>
PetriNet::buildGraphFromModule(circt::hw::HWModuleOp module,
                               const std::string &progressSignal) {
  auto petriNet = std::make_unique<PetriNet>();
  // llvm::errs() << "building graph from module" << module.getName() << "\n";
  petriNet->initializeGraph();
  llvm::SmallPtrSet<Operation *, 4> progressOps;
  // Step 1: Walk module and find signal to start analysis and building graph.
  module.walk([&](Operation *op) {
    // if (isPlaceOp(op)) {
    //   // This is adding every register into the petrinet, we do not want this
    //   // if (failed(petriNet->addPlaceFromRegister(op))) {
    //   //   llvm::errs() << "Failed to add place from register";
    //   // }

    // }
    auto n = getNameFromOp(op);
    // llvm::errs() << "Name: " << (n ? *n : "UNKOWN") << "\n";
    if (doesNameMatch(op, progressSignal)) {
      progressOps.insert(op);
      // llvm::errs() << "Found progress signal op: ";
      // op->print(llvm::errs());
      // Make place for this op.
    }
  });
  assert(!progressOps.empty() &&
         "No progress signal found in module, cannot build graph");
  return backwardsGraphBuildPass(progressOps);
}

std::shared_ptr<PetriNet::Place>
PetriNet::Place::createRegisterPlace(Operation *op) {
  return std::make_shared<PetriNet::Place>(op,
                                           std::move(getNameFromPlaceOp(op)));
}
std::shared_ptr<PetriNet::Place>
PetriNet::Place::createSinkPlace(Operation *op) {
  std::optional<std::string> name = getNameFromOp(op);
  return std::make_shared<PetriNet::Place>(op, name ? *name : "UNKOWN",
                                           /*sink=*/true);
}
LogicalResult PetriNet::addPlaceFromRegister(Operation *op) {
  assert(isPlaceOp(op) && "Operation is not a place operation");
  places.push_back(PetriNet::Place::createRegisterPlace(op));
  return success();
}

std::string PetriNet::Transition::toString() const {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << shortName() << ": ";
  ss << "\n";
  ss.flush();
  return s;
}

std::string PetriNet::Place::toString() const {
  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << shortName() << ": ";
  op->print(ss);
  ss << "\n";
  ss.flush();
  return s;
}

std::string PetriNet::toString() const {
  std::string s;
  llvm::raw_string_ostream ss(s);
  for (auto &place : places) {
    ss << place->toString();
  }
  for (auto &transition : transitions) {
    ss << transition->toString();
  }
  for (auto &arc : arcs) {
    ss << arc->toString();
  }
  ss.flush();
  return s;
}

void PetriNet::Transition::writeGraph(llvm::raw_ostream &os) const {
  size_t n = incomingArcs.size();
  os << "  t" << getId() << " [shape=record, label=\"";
  for (size_t i = 0; i < n; ++i) {
    os << "<f" << i << ">";
    if (i != n - 1) {
      os << "|";
    }
  }
  os << "\", xlabel=\"" << "t" << getId() << "\", width=0, height=" << n
     << "];\n";
  for (auto &op : operations) {
    os << "//";
    os << *op;
    os << "\n";
  }
}

void PetriNet::Place::writeGraph(llvm::raw_ostream &os) const {
  os << "  p" << getId() << " [shape=circle, width=1.25, label=\"\", xlabel=\""
     << (name.empty() ? "p" + std::to_string(getId()) : name) << "\""
     << (sink ? ", peripheries=3 " : "") << "];\n";
  os << "//" << *op;
  os << "\n";
}
void PetriNet::NormalArc::writeGraph(llvm::raw_ostream &os) const {
  if (source->getKind() == PetriNet::Node::Place) {
    os << "  p" << source->getId() << ":e" << " -> ";
  } else {
    os << "  t" << source->getId() << ":e -> ";
  }
  if (target->getKind() == PetriNet::Node::Place) {
    os << "p" << target->getId() << ":w";
  } else {
    os << "t" << target->getId() << ":f" << incomingNumber << ":w";
  }
  os << ";\n";
}
void PetriNet::InhibitorArc::writeGraph(llvm::raw_ostream &os) const {
  if (source->getKind() == PetriNet::Node::Place) {
    os << "  p" << source->getId() << " -> ";
  } else {
    os << "  t" << source->getId() << ":e -> ";
  }
  if (target->getKind() == PetriNet::Node::Place) {
    os << "p" << target->getId() << ";\n";
  } else {
    os << "t" << target->getId() << ":f" << incomingNumber << ":w" << ";\n";
  }
  os << "[arrowhead=odot];\n";
}
void PetriNet::writeGraph(llvm::raw_ostream &os, int levels) const {
  os << "digraph PetriNet{\n  rankdir = RL;\n";
  llvm::SmallVector<std::shared_ptr<Node>, 16> worklist;
  llvm::DenseSet<uint64_t> visited;
  llvm::DenseMap<uint64_t, int> levelMap;
  for (auto sink : sinks) {
    worklist.push_back(sink);
    visited.insert(sink->getNodeId());
    levelMap.insert({sink->getNodeId(), 0});
  }

  while (!worklist.empty()) {
    auto node = worklist.pop_back_val();
    node->writeGraph(os);
    for (auto arc : node->getIncomingArcs()) {
      arc->writeGraph(os);
      if (visited.count(arc->getSource()->getNodeId()) == 0) {
        auto level = levelMap[arc->getTarget()->getNodeId()];
        if (levels != -1 && level >= levels) {
          continue;
        }
        visited.insert(arc->getSource()->getNodeId());
        worklist.push_back(arc->getSource());
        levelMap.insert({arc->getSource()->getNodeId(),
                         levelMap[arc->getTarget()->getNodeId()] + 1});
      }
    }
  }
  os << "}\n";
}