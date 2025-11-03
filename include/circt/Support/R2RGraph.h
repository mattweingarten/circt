//===- R2RGraph.h - Petri Net -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

#ifndef CIRCT_R2RGRAPH_H
#define CIRCT_R2RGRAPH_H

namespace circt {
namespace r2r {

class R2RGraph {
public:
  class Register {
  public:
    explicit Register(mlir::Operation *op) : op(op), id(++idCounter) {
      assert(op && "Register constructed with null operation");
    };

    std::string getName() const;
    void writeGraph(llvm::raw_ostream &os) const;
    mlir::Operation *getOp() const { return op; }
    uint32_t getId() const { return id; }

  private:
    static uint32_t idCounter;
    mlir::Operation *op;
    uint32_t id;
  };

  explicit R2RGraph(circt::hw::HWModuleOp module);
  const std::vector<Register> &getRegisters() const { return registers; }
  void writeGraph(llvm::raw_ostream &os) const;

private:
  void build(hw::HWModuleOp module);
  std::vector<Register> registers;
};

} // namespace r2r
} // namespace circt

#endif // CIRCT_R2RGRAPH_H