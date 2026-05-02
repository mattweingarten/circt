// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(perf-insert-trace{targets=Top:x,Top:y})))' | FileCheck %s --check-prefix=INSERT
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(perf-insert-trace{targets=Top:x,Top:y}), firrtl.module(perf-emit-firesim-annotations)))' | FileCheck %s --check-prefix=EMIT

firrtl.circuit "Top" {
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
    %c0 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.regreset interesting_name %clock, %reset, %c0 : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
    %x = firrtl.node interesting_name %0 : !firrtl.uint<1>
    %y = firrtl.wire interesting_name : !firrtl.uint<1>
    firrtl.connect %y, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// INSERT: module {
// INSERT:   firrtl.circuit "Top" {
// INSERT:     firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
// INSERT:       [[C0:%.*]] = firrtl.constant 0 : !firrtl.uint<1>
// INSERT:       %0 = firrtl.regreset interesting_name %clock, %reset, [[C0]] : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
// INSERT:       %x = firrtl.node interesting_name %0 : !firrtl.uint<1>
// INSERT:       %x_perf = firrtl.wire : !firrtl.uint<1>
// INSERT:       perf.trace %x_perf : !firrtl.uint<1>, "x", %clock : !firrtl.clock, %reset : !firrtl.reset
// INSERT:       firrtl.connect %x_perf, %x : !firrtl.uint<1>, !firrtl.uint<1>
// INSERT:       %y = firrtl.wire interesting_name : !firrtl.uint<1>
// INSERT:       perf.trace %y : !firrtl.uint<1>, "y", %clock : !firrtl.clock, %reset : !firrtl.reset
// INSERT:       firrtl.connect %y, %0 : !firrtl.uint<1>, !firrtl.uint<1>
// INSERT:     }
// INSERT:   }
// INSERT: }

// EMIT: module {
// EMIT:   firrtl.circuit "Top" {
// EMIT:     firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
// EMIT:       [[C0:%.*]] = firrtl.constant 0 : !firrtl.uint<1>
// EMIT:       %0 = firrtl.regreset interesting_name %clock, %reset, [[C0]] : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
// EMIT:       %x = firrtl.node interesting_name %0 : !firrtl.uint<1>
// EMIT:       %x_perf = firrtl.wire
// EMIT-SAME:  {annotations =
// EMIT-SAME:  class = "midas.targetutils.TraceDoctorFirrtlAnnotation"
// EMIT-SAME:  clock = "~Top|Top>clock"
// EMIT-SAME:  coverGenerated = false
// EMIT-SAME:  description = "x"
// EMIT-SAME:  label = "x"
// EMIT-SAME:  reset = "~Top|Top>reset"
// EMIT-SAME:  class = "firrtl.transforms.DontTouchAnnotation"
// EMIT-SAME:  : !firrtl.uint<1>
// EMIT:       firrtl.connect %x_perf, %x : !firrtl.uint<1>, !firrtl.uint<1>
// EMIT:       %y = firrtl.wire interesting_name
// EMIT-SAME:  {annotations =
// EMIT-SAME:  class = "midas.targetutils.TraceDoctorFirrtlAnnotation"
// EMIT-SAME:  clock = "~Top|Top>clock"
// EMIT-SAME:  coverGenerated = false
// EMIT-SAME:  description = "y"
// EMIT-SAME:  label = "y"
// EMIT-SAME:  reset = "~Top|Top>reset"
// EMIT-SAME:  class = "firrtl.transforms.DontTouchAnnotation"
// EMIT-SAME:  : !firrtl.uint<1>
// EMIT:       firrtl.connect %y, %0 : !firrtl.uint<1>, !firrtl.uint<1>
// EMIT-NOT:   perf.trace
// EMIT:     }
// EMIT:   }
// EMIT: }