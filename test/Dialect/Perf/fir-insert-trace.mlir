// RUN: circt-opt %s \
// RUN:   -pass-pipeline='builtin.module(firrtl.circuit(perf-insert-trace{targets=~Top|Top>x,~Top|Top>y}))' \
// RUN:   | FileCheck %s --check-prefix=INSERT

// RUN: circt-opt %s \
// RUN:   -pass-pipeline='builtin.module(firrtl.circuit(perf-insert-trace{targets=~Top|Top>x,~Top|Top>y}, firrtl.module(perf-emit-firesim-annotations)))' \
// RUN:   | FileCheck %s --check-prefix=EMIT

// RUN: circt-opt %s \
// RUN:   -pass-pipeline='builtin.module(firrtl.circuit(perf-insert-trace{targets=~Top|Top/u_mid:Mid/u_leaf:Leaf>z}))' \
// RUN:   | FileCheck %s --check-prefix=CTX-INSERT

// RUN: circt-opt %s \
// RUN:   -pass-pipeline='builtin.module(firrtl.circuit(perf-insert-trace{targets=~Top|Top/u_mid:Mid/u_leaf:Leaf>z}, firrtl.module(perf-emit-firesim-annotations)))' \
// RUN:   | FileCheck %s --check-prefix=CTX-EMIT

firrtl.circuit "Top" {
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
    %c0 = firrtl.constant 0 : !firrtl.uint<1>

    %0 = firrtl.regreset interesting_name %clock, %reset, %c0 : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
    %x = firrtl.node interesting_name %0 : !firrtl.uint<1>

    %y = firrtl.wire interesting_name : !firrtl.uint<1>
    firrtl.connect %y, %0 : !firrtl.uint<1>, !firrtl.uint<1>

    %u_mid_clock, %u_mid_reset = firrtl.instance u_mid @Mid(in clock: !firrtl.clock, in reset: !firrtl.reset)
    firrtl.connect %u_mid_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %u_mid_reset, %reset : !firrtl.reset, !firrtl.reset
  }

  firrtl.module @Mid(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
    %u_leaf_clock, %u_leaf_reset = firrtl.instance u_leaf @Leaf(in clock: !firrtl.clock, in reset: !firrtl.reset)
    firrtl.connect %u_leaf_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %u_leaf_reset, %reset : !firrtl.reset, !firrtl.reset
  }

  firrtl.module @Leaf(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
    %c0 = firrtl.constant 0 : !firrtl.uint<1>
    %leaf_reg = firrtl.regreset %clock, %reset, %c0 : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
    %z = firrtl.node interesting_name %leaf_reg : !firrtl.uint<1>

    %w = firrtl.wire : !firrtl.uint<1>
    firrtl.connect %w, %z : !firrtl.uint<1>, !firrtl.uint<1>
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
// INSERT:       %u_mid_clock, %u_mid_reset = firrtl.instance u_mid @Mid(in clock: !firrtl.clock, in reset: !firrtl.reset)
// INSERT:       firrtl.connect %u_mid_clock, %clock : !firrtl.clock, !firrtl.clock
// INSERT:       firrtl.connect %u_mid_reset, %reset : !firrtl.reset, !firrtl.reset
// INSERT:     }
// INSERT:     firrtl.module @Mid(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
// INSERT:       %u_leaf_clock, %u_leaf_reset = firrtl.instance u_leaf @Leaf(in clock: !firrtl.clock, in reset: !firrtl.reset)
// INSERT:       firrtl.connect %u_leaf_clock, %clock : !firrtl.clock, !firrtl.clock
// INSERT:       firrtl.connect %u_leaf_reset, %reset : !firrtl.reset, !firrtl.reset
// INSERT:     }
// INSERT:     firrtl.module @Leaf(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
// INSERT:       [[LC0:%.*]] = firrtl.constant 0 : !firrtl.uint<1>
// INSERT:       %leaf_reg = firrtl.regreset %clock, %reset, [[LC0]] : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
// INSERT:       %z = firrtl.node interesting_name %leaf_reg : !firrtl.uint<1>
// INSERT:       %w = firrtl.wire : !firrtl.uint<1>
// INSERT:       firrtl.connect %w, %z : !firrtl.uint<1>, !firrtl.uint<1>
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
// EMIT-SAME:  description = ""
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
// EMIT-SAME:  description = ""
// EMIT-SAME:  label = "y"
// EMIT-SAME:  reset = "~Top|Top>reset"
// EMIT-SAME:  class = "firrtl.transforms.DontTouchAnnotation"
// EMIT-SAME:  : !firrtl.uint<1>
// EMIT:       firrtl.connect %y, %0 : !firrtl.uint<1>, !firrtl.uint<1>
// EMIT-NOT:   perf.trace
// EMIT:     }
// EMIT:   }
// EMIT: }

// CTX-INSERT: module {
// CTX-INSERT:   firrtl.circuit "Top" {
// CTX-INSERT:     hw.hierpath @[[HPATH:[A-Za-z0-9_.$]+]]
// CTX-INSERT-SAME: [
// CTX-INSERT-SAME: @Top::@u_mid
// CTX-INSERT-SAME: @Mid::@u_leaf
// CTX-INSERT-SAME: ]

// CTX-INSERT:     firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
// CTX-INSERT:       %u_mid_clock, %u_mid_reset = firrtl.instance u_mid
// CTX-INSERT-SAME:  sym @u_mid
// CTX-INSERT-SAME:  @Mid
// CTX-INSERT:       firrtl.connect %u_mid_clock, %clock : !firrtl.clock, !firrtl.clock
// CTX-INSERT:       firrtl.connect %u_mid_reset, %reset : !firrtl.reset, !firrtl.reset
// CTX-INSERT:     }

// CTX-INSERT:     firrtl.module @Mid(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
// CTX-INSERT:       %u_leaf_clock, %u_leaf_reset = firrtl.instance u_leaf
// CTX-INSERT-SAME:  sym @u_leaf
// CTX-INSERT-SAME:  @Leaf
// CTX-INSERT:       firrtl.connect %u_leaf_clock, %clock : !firrtl.clock, !firrtl.clock
// CTX-INSERT:       firrtl.connect %u_leaf_reset, %reset : !firrtl.reset, !firrtl.reset
// CTX-INSERT:     }

// CTX-INSERT:     firrtl.module @Leaf(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
// CTX-INSERT:       [[LC0:%.*]] = firrtl.constant 0 : !firrtl.uint<1>
// CTX-INSERT:       %leaf_reg = firrtl.regreset %clock, %reset, [[LC0]] : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<1>
// CTX-INSERT:       %z = firrtl.node interesting_name %leaf_reg : !firrtl.uint<1>
// CTX-INSERT:       %z_perf = firrtl.wire : !firrtl.uint<1>
// CTX-INSERT:       perf.trace %z_perf : !firrtl.uint<1>, "z", %clock : !firrtl.clock, %reset : !firrtl.reset context @[[HPATH]]
// CTX-INSERT:       firrtl.connect %z_perf, %z : !firrtl.uint<1>, !firrtl.uint<1>
// CTX-INSERT:       %w = firrtl.wire : !firrtl.uint<1>
// CTX-INSERT:       firrtl.connect %w, %z : !firrtl.uint<1>, !firrtl.uint<1>
// CTX-INSERT:     }
// CTX-INSERT:   }
// CTX-INSERT: }

// CTX-EMIT: module {
// CTX-EMIT:   firrtl.circuit "Top" {
// CTX-EMIT:     hw.hierpath @[[HPATH:[A-Za-z0-9_.$]+]]
// CTX-EMIT-SAME: [
// CTX-EMIT-SAME: @Top::@u_mid
// CTX-EMIT-SAME: @Mid::@u_leaf
// CTX-EMIT-SAME: ]

// CTX-EMIT:     firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
// CTX-EMIT:       %u_mid_clock, %u_mid_reset = firrtl.instance u_mid
// CTX-EMIT-SAME:  sym @u_mid
// CTX-EMIT-SAME:  @Mid
// CTX-EMIT:       firrtl.connect %u_mid_clock, %clock : !firrtl.clock, !firrtl.clock
// CTX-EMIT:       firrtl.connect %u_mid_reset, %reset : !firrtl.reset, !firrtl.reset
// CTX-EMIT:     }

// CTX-EMIT:     firrtl.module @Mid(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
// CTX-EMIT:       %u_leaf_clock, %u_leaf_reset = firrtl.instance u_leaf
// CTX-EMIT-SAME:  sym @u_leaf
// CTX-EMIT-SAME:  @Leaf
// CTX-EMIT:       firrtl.connect %u_leaf_clock, %clock : !firrtl.clock, !firrtl.clock
// CTX-EMIT:       firrtl.connect %u_leaf_reset, %reset : !firrtl.reset, !firrtl.reset
// CTX-EMIT:     }

// CTX-EMIT:     firrtl.module @Leaf(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
// CTX-EMIT:       %z = firrtl.node interesting_name %leaf_reg : !firrtl.uint<1>
// CTX-EMIT:       %z_perf = firrtl.wire
// CTX-EMIT-SAME:  {annotations =
// CTX-EMIT-SAME:  class = "midas.targetutils.TraceDoctorFirrtlAnnotation"
// CTX-EMIT-SAME:  clock = "~Top|Top/u_mid:Mid/u_leaf:Leaf>clock"
// CTX-EMIT-SAME:  coverGenerated = false
// CTX-EMIT-SAME:  description = ""
// CTX-EMIT-SAME:  label = "z"
// CTX-EMIT-SAME:  reset = "~Top|Top/u_mid:Mid/u_leaf:Leaf>reset"
// CTX-EMIT-SAME:  class = "firrtl.transforms.DontTouchAnnotation"
// CTX-EMIT-SAME:  : !firrtl.uint<1>
// CTX-EMIT:       firrtl.connect %z_perf, %z : !firrtl.uint<1>, !firrtl.uint<1>
// CTX-EMIT-NOT:   perf.trace
// CTX-EMIT:     }
// CTX-EMIT:   }
// CTX-EMIT: }