// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(perf-insert-counter{targets=Top:x})))' | FileCheck %s --check-prefix=INSERT
// RUN: circt-opt %s -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(perf-insert-counter{targets=Top:x})),perf-emit-autocounter{file=%t.json})' >/dev/null && FileCheck %s --check-prefix=JSON < %t.json

firrtl.circuit "Top" {
  firrtl.module @Top(in %clock: !firrtl.clock) {
    %0 = firrtl.reg interesting_name %clock : !firrtl.clock, !firrtl.uint<1>
    %x = firrtl.node interesting_name %0 : !firrtl.uint<1>
  }
}

// INSERT: module {
// INSERT:   firrtl.circuit "Top" {
// INSERT:     firrtl.module @Top(in %clock: !firrtl.clock) {
// INSERT:       %0 = firrtl.reg interesting_name %clock : !firrtl.clock, !firrtl.uint<1>
// INSERT:       %x = firrtl.node interesting_name %0 : !firrtl.uint<1>
// INSERT:       perf.counter %x : !firrtl.uint<1>, "x", %clock : !firrtl.clock
// INSERT:     }
// INSERT:   }
// INSERT: }

// JSON: "class":"midas.targetutils.AutoCounterFirrtlAnnotation"
// JSON: "target":"~Top|Top>x"
// JSON: "clock":"~Top|Top>clock"
// JSON: "reset":"~Top|Top>reset"
// JSON: "label":"x"
// JSON: "description":"x<--(circt autogen)"
// JSON: "opType":{
// JSON: "class":"midas.targetutils.PerfCounterOps$Accumulate$"
// JSON: }
// JSON: "coverGenerated":false