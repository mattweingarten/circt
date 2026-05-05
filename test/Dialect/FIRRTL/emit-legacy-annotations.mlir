// RUN: rm -f %t.json
// RUN: circt-opt %s \
// RUN:   -pass-pipeline='builtin.module(firrtl.circuit(firrtl-emit-legacy-annotations{file=%t.json}))' \
// RUN:   >/dev/null
// RUN: FileCheck %s --input-file=%t.json --check-prefix=JSON

// RUN: not circt-opt %s \
// RUN:   -pass-pipeline='builtin.module(firrtl.circuit(firrtl-emit-legacy-annotations))' \
// RUN:   2>&1 | FileCheck %s --check-prefix=MISSING

firrtl.circuit "Top" attributes {
  annotations = [
    {
      class = "test.CircuitAnno",
      str = "hello",
      flag = true,
      int = 42 : i64
    }
  ]
} {
  hw.hierpath @fooNLA [@Top::@bar, @Bar::@baz, @Baz::@foo]
  hw.hierpath @bazPortNLA [@Top::@bar, @Bar::@baz, @Baz::@clock]
  hw.hierpath @differentSymNLA [@Top::@differentSym, @DifferentSymLeaf::@leafWire]

  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) attributes {
    annotations = [
      {
        class = "test.ModuleAnno"
      }
    ],
    portAnnotations = [
      [
        {
          class = "test.ClockPortAnno"
        }
      ],
      [
        {
          class = "test.ResetPortAnno"
        }
      ]
    ]
  } {
    %c0 = firrtl.constant 0 : !firrtl.uint<1>

    %foo = firrtl.wire sym @foo {annotations = [
      {
        class = "test.NamedWireAnno",
        target = "this.should.be.ignored"
      }
    ]} : !firrtl.uint<1>

    firrtl.strictconnect %foo, %c0 : !firrtl.uint<1>, !firrtl.uint<1>

    firrtl.instance child @Child()

    %bar_clock = firrtl.instance bar sym @bar @Bar(in clock: !firrtl.clock)
    firrtl.strictconnect %bar_clock, %clock : !firrtl.clock, !firrtl.clock

    firrtl.instance real_instance_name sym @differentSym @DifferentSymLeaf()
  }

  firrtl.module @Child() attributes {
    annotations = [
      {
        class = "test.ChildModuleAnno"
      },
      {
        class = "firrtl.transforms.BlackBoxInlineAnno"
      }
    ]
  } {
  }

  firrtl.module @Bar(in %clock: !firrtl.clock) {
    %baz_clock = firrtl.instance baz sym @baz @Baz(in clock: !firrtl.clock)
    firrtl.strictconnect %baz_clock, %clock : !firrtl.clock, !firrtl.clock
  }

  firrtl.module @Baz(in %clock: !firrtl.clock sym @clock) attributes {
    portAnnotations = [
      [
        {
          class = "test.NLAPortAnnotation",
          circt.nonlocal = @bazPortNLA
        }
      ]
    ]
  } {
    %c0 = firrtl.constant 0 : !firrtl.uint<1>

    %foo = firrtl.wire sym @foo {annotations = [
      {
        class = "test.NonLocalWireAnno",
        circt.nonlocal = @fooNLA
      }
    ]} : !firrtl.uint<1>

    firrtl.strictconnect %foo, %c0 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  firrtl.module @DifferentSymLeaf() {
  %leaf = firrtl.wire sym @leafWire {annotations = [
    {
      class = "test.DifferentSymbolNameAnno",
      circt.nonlocal = @differentSymNLA
    }
  ]} : !firrtl.uint<1>
  }
}

// JSON-DAG: "class": "test.CircuitAnno"
// JSON-DAG: "flag": true
// JSON-DAG: "int": 42
// JSON-DAG: "str": "hello"
// JSON-DAG: "target": "~Top"

// JSON-DAG: "class": "test.ModuleAnno"
// JSON-DAG: "target": "~Top|Top"

// JSON-DAG: "class": "test.ClockPortAnno"
// JSON-DAG: "target": "~Top|Top>clock"

// JSON-DAG: "class": "test.ResetPortAnno"
// JSON-DAG: "target": "~Top|Top>reset"

// JSON-DAG: "class": "test.NamedWireAnno"
// JSON-DAG: "target": "~Top|Top>foo"

// JSON-DAG: "class": "test.ChildModuleAnno"
// JSON-DAG: "target": "~Top|Child"

// JSON-DAG: "class": "firrtl.transforms.BlackBoxInlineAnno"
// JSON-DAG: "target": "Top.Child"

// JSON-DAG: "class": "test.NonLocalWireAnno"
// JSON-DAG: "target": "~Top|Top/bar:Bar/baz:Baz>foo"

// JSON-DAG: "class": "test.NLAPortAnnotation"
// JSON-DAG: "target": "~Top|Top/bar:Bar/baz:Baz>clock"

// JSON-DAG: "class": "test.DifferentSymbolNameAnno"
// JSON-DAG: "target": "~Top|Top/real_instance_name:DifferentSymLeaf>leaf"

// JSON-NOT: "circt.nonlocal"

// MISSING: error: missing output filename; use --firrtl-emit-legacy-annotations=file=<path>