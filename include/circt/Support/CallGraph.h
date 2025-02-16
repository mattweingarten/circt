//===- CallGraph.h - Call graph -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a generic call graph for whole program analysis
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/SmallVector.h"

#ifndef CIRCT_SUPPORT_CALLGRAPH_H
#define CIRCT_SUPPORT_CALLGRAPH_H

namespace circt {
namespace cgraph {

// Do we need to be able to handle regions as nodes? Or do we treat them as is?
class CallGraphNode {

public:
  CallGraphNode(ModuleOp *module);
  bool hasChildren();
  void addChild(std::unique_ptr<CallGraphNode> child);

private:
  // TODO: more useful fields, such as size of this node, some prelimiary
  // analysis caches, etc...
  ModuleOp *module;
  // Maybe we want to represent the CallGraph with an Edge the same way LLVM
  // CallGraph does? This would allow us to represent edges to unkown areas of
  // the code? --> I think this is a good idea for the future.
  llvm::SmallVector<std::unique_ptr<CallGraphNode>> children;
};

class CallGraph {
public:
  CallGraph() : root(nullptr) {}

private:
  std::unique_ptr<CallGraphNode> root;
};
} // namespace cgraph
} // namespace circt
#endif // CIRCT_SUPPORT_CALLGRAPH_H
