//===- R2RGraph.cpp - Petri Net -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/R2RGraph.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "r2r-graph"

using namespace circt;
using namespace r2r;

uint32_t R2RGraph::Register::idCounter = 0;

// Name clashes here for sv namehint
// std::string R2RGraph::Register::getName() const {
//   if (auto nameAttr = op->getAttrOfType<mlir::StringAttr>("sv.namehint"))
//     return nameAttr.getValue().str();

//   std::string s;
//   llvm::raw_string_ostream os(s);
//   op->print(os);
//   os.flush();
//   return s;
// }

std::string R2RGraph::Register::getName() const {
  std::string s;
  llvm::raw_string_ostream os(s);
  op->print(os);
  os.flush();

  auto percentPos = s.find('%');
  auto equalPos = s.find('=', percentPos);
  if (percentPos != std::string::npos && equalPos != std::string::npos &&
      equalPos > percentPos + 1) {
    auto name = s.substr(percentPos + 1, equalPos - percentPos - 2);
    while (!name.empty() && isspace(name.back()))
      name.pop_back();
    return name;
  }
  return "<unnamed_reg>";
}
R2RGraph::R2RGraph(circt::hw::HWModuleOp module) {
  LLVM_DEBUG(llvm::outs() << "[R2RGraph] Building R2RGraph for module: "
                          << module.getName() << "\n";);
  build(module);
}

void R2RGraph::build(hw::HWModuleOp module) {
  module.walk([&](seq::FirRegOp regOp) {
    Register reg(regOp.getOperation());
    LLVM_DEBUG(llvm::outs() << "  [R2RGraph] reg: " << reg.getName() << "\n";);
    registers.emplace_back(std::move(reg));
  });
}

void R2RGraph::writeGraph(llvm::raw_ostream &os) const {
  os << "digraph PetriNet{\n  rankdir = LR;\n";
  for (const auto &reg : registers) {
    reg.writeGraph(os);
  }
  os << "}\n";
}

void R2RGraph::Register::writeGraph(llvm::raw_ostream &os) const {
  os << "r" << getId() << "[shape=circle, width=1.25, label=\"\", xlabel=\""
     << getName() << "\"];\n"
     << "//" << *op << "\n";
}