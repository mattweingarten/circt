//===- PerfOps.td - `perf` dialect ops ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for `perf`.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PERF_PERFOPS_TD
#define CIRCT_DIALECT_PERF_PERFOPS_TD

include "circt/Dialect/Perf/PerfDialect.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "circt/Dialect/Seq/SeqOpInterfaces.td"

include "mlir/IR/OpAsmInterface.td"

class PerfOp<string mnemonic, list<Trait> traits = []>
    : Op<PerfDialect, mnemonic, traits>;

// TODO: Generalize this to all other Dialects? We want this to interoperate
// with any dialect essentially. Is there a nicer way to do this?
def AnyBoolType : Type<CPred<"::llvm::isa<::mlir::IntegerType>($_self) && "
                             "::llvm::cast<::mlir::IntegerType>($_self)."
                             "isSignlessInteger(1) || "
                             "::llvm::isa<firrtl::UIntType>($_self) && "
                             "::llvm::cast<firrtl::UIntType>($_self)."
                             "getWidthOrSentinel() == 1">,
                       "bool-like condition (i1 or firrtl.uint<1>)">;

def AnyResetType : Type<Or<[
                           AnyBoolType.predicate,
                           CPred<"$_self.isa<circt::firrtl::ResetType>()">
                         ]>,
                         "any reset type (bool-like or firrtl.reset)">;

def AnyClockType : Type<Or<[
                          ClockType.predicate,
                          CPred<"$_self.isa<circt::firrtl::ClockType>()">
                        ]>,
                        "Any clock type">;




def PerfCounterOp : PerfOp<"counter", [
  Clocked,
]> {
  let summary = "conditional performance counter increment";
  let description = [{}];

  let arguments = (ins
    AnyBoolType:$cond,
    AnyClockType:$clk,
    Optional<AnyResetType>:$reset,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<StrAttr>:$desc,
    OptionalAttr<FlatSymbolRefAttr>:$context
  );

  let results = (outs);

  let builders = [
    OpBuilder<(ins
      "mlir::Value":$cond,
      "mlir::Value":$clk,
      "mlir::Value":$reset,
      "mlir::StringAttr":$name,
      "mlir::StringAttr":$desc
    ), [{
      build($_builder, $_state,
            cond, clk, reset, name, desc,
            /*context=*/mlir::FlatSymbolRefAttr());
    }]>
  ];

  let assemblyFormat = [{
    $cond `:` type($cond)
    (`,` $name^ (`,` $desc^)? )?
    `,` $clk `:` type($clk)
    (`,` $reset^ `:` type($reset) )?
    (`context` $context^)?
    attr-dict
  }];
}

def PerfSamplerOp : PerfOp<"sampler", [Clocked, AttrSizedOperandSegments]> {
  let summary = "conditional value sampling event";
  let description = [{}];

  let arguments = (ins
    AnyType:$value,
    Optional<AnyBoolType>:$cond,
    AnyClockType:$clk,
    Optional<AnyResetType>:$reset,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<StrAttr>:$desc
  );

  let results = (outs);

  let assemblyFormat = [{
    $value `:` type($value)
    (`,` $cond^ `:` type($cond))?
    `,` $clk `:` type($clk)
    (`,` $reset^ `:` type($reset))?
    (`,` $name^  (`,` $desc^)?)?
    attr-dict
  }];
}

def PerfTraceOp : PerfOp<"trace", [
  Clocked,
]> {
  let summary = "conditional trace event";
  let description = [{}];

  let arguments = (ins
    AnyBoolType:$cond,
    AnyClockType:$clk,
    Optional<AnyResetType>:$reset,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<StrAttr>:$desc,
    OptionalAttr<FlatSymbolRefAttr>:$context
  );

  let results = (outs);

  let builders = [
    OpBuilder<(ins
      "mlir::Value":$cond,
      "mlir::Value":$clk,
      "mlir::Value":$reset,
      "mlir::StringAttr":$name,
      "mlir::StringAttr":$desc
    ), [{
      build($_builder, $_state,
            cond, clk, reset, name, desc,
            /*context=*/mlir::FlatSymbolRefAttr());
    }]>
  ];

  let assemblyFormat = [{
    $cond `:` type($cond)
    (`,` $name^ (`,` $desc^)? )?
    `,` $clk `:` type($clk)
    (`,` $reset^ `:` type($reset) )?
    (`context` $context^)?
    attr-dict
  }];

}

#endif // CIRCT_DIALECT_PERF_PERFOPS_TD
