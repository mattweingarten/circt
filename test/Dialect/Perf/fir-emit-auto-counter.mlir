// RUN: circt-opt --perf-emit-autocounter='file=%t.json' %s >/dev/null
// RUN: FileCheck %s --check-prefix=JSON < %t.json

module {
  firrtl.circuit "Top" {
    firrtl.module @Top(
        in %clk:   !firrtl.clock,
        in %reset: !firrtl.uint<1>,
        in %cond:  !firrtl.uint<1>
      ) attributes {convention = #firrtl<convention scalarized>} {

      %wb_valid = firrtl.node interesting_name %cond : !firrtl.uint<1>

      perf.counter %wb_valid : !firrtl.uint<1>,
                   "wb_valid", "WB valid perf counter",
                   %clk : !firrtl.clock,
                   %reset : !firrtl.uint<1>

      firrtl.skip
    }
  }
}

// JSON: "class":"midas.targetutils.AutoCounterFirrtlAnnotation"
// JSON: "target":"~Top|Top>wb_valid"
// JSON: "clock":"~Top|Top>clk"
// JSON: "reset":"~Top|Top>reset"
// JSON: "label":"wb_valid"
// JSON: "description":"wb_valid<--(circt autogen)"
// JSON: "opType":{
// JSON: "class":"midas.targetutils.PerfCounterOps$Accumulate$"
// JSON: }
// JSON: "coverGenerated":false