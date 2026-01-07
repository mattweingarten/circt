//===- PerfOpInterfaces.td - Operation Interfaces ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the Perf operation interfaces and traits.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PERF_PERFOPINTERFACES
#define CIRCT_DIALECT_PERF_PERFOPINTERFACES

include "mlir/IR/OpBase.td"

// def Clocked : OpInterface<"Clocked"> {
//   let cppNamespace = "circt::perf";
//   let description = "Trait for clocked elements";

//   let methods = [
//     InterfaceMethod<"Get the clock signal", "Value", "getClk", (ins)>
//   ];
// }

// def Resettable : OpInterface<"Resettable"> {
//   let cppNamespace = "circt::seq";
//   let description = "Trait for Resettable elements";

//   let methods = [
//     InterfaceMethod<"Get the reset signal", "Value", "getReset", (ins)>,
//     InterfaceMethod<"Get the reset value", "Value", "getResetValue", (ins)>
//   ];
// }


#endif // CIRCT_DIALECT_PERF_PERFOPINTERFACES
