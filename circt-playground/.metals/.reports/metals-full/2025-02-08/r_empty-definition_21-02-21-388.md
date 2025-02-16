error id: example/
file://<WORKSPACE>/chisel_test/m1.scala
empty definition using pc, found symbol in pc: example/
semanticdb not found
|empty definition using fallback
non-local guesses:
	 -

Document text:

```scala
import chisel3._
import chisel3.util._

// Module 1
class Module1 extends Module {
  val io = IO(new Bundle {
    val data_in  = Input(UInt(4.W))
    val data_out = Output(UInt(4.W))
    val rst      = Input(Bool())
    val clk      = Input(Clock())
  })

  val reg = RegInit(0.U(4.W))
  when(io.rst) {
    reg := 0.U
  }.otherwise {
    reg := (io.data_in + 1.U) ^ 3.U(4.W)
  }
  io.data_out := reg
}

// Module 2
class Module2 extends Module {
  val io = IO(new Bundle {
    val data_in  = Input(UInt(4.W))
    val data_out = Output(UInt(4.W))
    val rst      = Input(Bool())
    val clk      = Input(Clock())
  })

  val reg = RegInit(0.U(4.W))
  when(io.rst) {
    reg := 0.U
  }.otherwise {
    reg := (io.data_in << 1) | 1.U(4.W)
  }
  io.data_out := reg
}

// Module 3
class Module3 extends Module {
  val io = IO(new Bundle {
    val data_in  = Input(UInt(4.W))
    val data_out = Output(UInt(4.W))
    val rst      = Input(Bool())
    val clk      = Input(Clock())
  })

  val reg = RegInit(0.U(4.W))
  when(io.rst) {
    reg := 0.U
  }.otherwise {
    reg := (io.data_in ^ 10.U(4.W)) + 3.U(4.W)
  }
  io.data_out := reg
}

// Top-level Module
class TopModule extends Module {
  val io = IO(new Bundle {
    val clk      = Input(Clock())
    val rst      = Input(Bool())
    val in_data  = Input(UInt(4.W))
    val out_data = Output(UInt(4.W))
  })

  val u1 = Module(new Module1)
  val u2 = Module(new Module2)
  val u3 = Module(new Module3)

  u1.io.data_in := io.in_data
  u1.io.rst := io.rst
  u1.io.clk := io.clk
  
  u2.io.data_in := u1.io.data_out
  u2.io.rst := io.rst
  u2.io.clk := io.clk
  
  u3.io.data_in := u2.io.data_out
  u3.io.rst := io.rst
  u3.io.clk := io.clk
  
  io.out_data := u3.io.data_out
}


object TopTest extends App {
  chisel3.Driver.execute(args, () => new TopModule)
}
```

#### Short summary: 

empty definition using pc, found symbol in pc: example/