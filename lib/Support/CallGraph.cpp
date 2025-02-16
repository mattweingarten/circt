//===- InstanceGraph.cpp - Call Graph -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/CallGraph.h"

using namespace circt;
using namespace cgraph;

CallGraphNode::CallGraphNode(ModuleOp *module) : module(module) {}

bool CallGraphNode::hasChildren() { return children.empty(); }

void CallGraphNode::addChild(std::unique_ptr<CallGraphNode> child) {
  children.push_back(std::move(child));
}
