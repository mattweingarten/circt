//===- PrintPetriNet.cpp - Print Petri Net ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Print Petri Net of Module.
//
//===----------------------------------------------------------------------===//
#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"

#include "circt/Dialect/Perf/PerfDialect.h"

#include "circt/Dialect/FIRRTL/CounterInserter.h"
#include "circt/Dialect/FIRRTL/Petrinet.h"
#include "circt/Dialect/FIRRTL/TransformZ3Graph.h"

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/PetriNet.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <z3++.h>

#include "circt/Support/PetriNet.h"
#include <optional>

#include <algorithm>
#include <cassert>
#include <map>
#include <string>
#include <unordered_set>

#define DEBUG_TYPE "petri"

#define ASSERT_STATE_DEBUG(cond)                                               \
  do {                                                                         \
    if (!(cond)) {                                                             \
      llvm::dbgs() << "Assertion failed: " << #cond << "\n";                   \
      state.dump(llvm::dbgs());                                                \
      assert(cond);                                                            \
    }                                                                          \
  } while (0)

using namespace circt;
using namespace firrtl;

namespace {
struct FIRRTLPrintPetriNetPass
    : public circt::firrtl::PrintPetriNetBase<FIRRTLPrintPetriNetPass> {
  FIRRTLPrintPetriNetPass(std::string moduleName, std::string progressSignal,
                          std::string topModule, std::string petriFile,
                          std::string debugDirectory, int levels,
                          std::string counterMode) {
    moduleName = moduleName;
    progressSignal = progressSignal;
    topModule = topModule;
    petriFile = petriFile;
    debugDirectory = debugDirectory;
    levels = levels;
    counterMode = counterMode;
  }
  void runOnOperation() override;
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<circt::perf::PerfDialect>();
    registry.insert<circt::firrtl::FIRRTLDialect>();
  }

private:
  struct FIRRTLPrintPetriNetPassState {
    struct LocalState {
      // context (i.e for handling bundles)
      std::vector<std::string> fields;
      std::vector<mlir::Operation *> operationPath;

      struct InterproceduralState {
        struct CallInfo {
          mlir::Operation *caller;
          mlir::Operation *callsite;
          mlir::Operation *callee;
          unsigned injectPortIdx;
        };
        std::vector<CallInfo> callerContext;

        InterproceduralState() = default;
        InterproceduralState(const InterproceduralState &) = default;
        InterproceduralState &operator=(const InterproceduralState &) = default;
      };

      InterproceduralState interproceduralState;

      // Deep copy to store and freeze LocalState
      // LocalState(const LocalState &other)
      //     : fields(other.fields), operationPath(other.operationPath),
      //       interproceduralState(other.interproceduralState) {

      //   interproceduralState.callerContext.reserve(
      //       other.interproceduralState.callerContext.size());

      //   for (const auto &ci : other.interproceduralState.callerContext) {
      //     interproceduralState.callerContext.push_back(
      //         typename InterproceduralState::CallInfo{
      //             ci.caller, ci.callsite, ci.callee, ci.injectPortIdx});
      //   }
      // }

      LocalState() = default;
      LocalState(const LocalState &) = default;
      LocalState &operator=(const LocalState &) = default;
    };

    struct StageState {
      struct ActivationPoint {
        LocalState localState;
        mlir::Operation *op;

        ActivationPoint(const LocalState &localState, mlir::Operation *op)
            : localState(localState), op(op) {}
        static ActivationPoint create(const FIRRTLPrintPetriNetPassState &state,
                                      mlir::Operation *op) {
          return ActivationPoint(state.localState, op);
        }

        bool operator<(const ActivationPoint &rhs) const { return op < rhs.op; }
      };

      bool done = false;
      int stageNumber = 0;

      std::set<ActivationPoint> activePoints;
      std::set<ActivationPoint> nextActivePoints;
      std::vector<AnnoPathValue> visited;
      std::unique_ptr<circt::pnet::PetriNet> petriNet;
    };

    // Always stays the same, regardless of which stage we are at or where in
    // the program we are at
    struct GlobalState {
      // Is this generalizable enough? is 1 to 1 mapping enough?
      llvm::DenseMap<mlir::Operation *, mlir::Operation *> connectToRegister;
      std::set<std::string> registers;
      std::unique_ptr<z3::context> z3ctx;
      // TODO: better place for this
      std::vector<z3::expr> finalExprs;

      std::map<int, std::vector<z3::expr>> perStageExprs;
      std::map<int, std::vector<std::string>>
          perStageNames; // strongly recommended

      std::vector<std::string> nodeNames;
      std::vector<AnnoPathValue> targets;

      CircuitOp circuit;
      FModuleOp top;

      std::map<std::string, AnnoPathValue> targetCache;

      int nameCounter = 0;

      GlobalState() = default;
      GlobalState(CircuitOp circuit, FModuleOp top)
          : circuit(circuit), top(top), z3ctx(std::make_unique<z3::context>()) {
      }
    };

    LocalState localState;
    StageState stageState;
    GlobalState globalState;

    unsigned getCurrInjectedPort() {
      assert(!callerContext().empty() && "No caller context available");
      return callerContext().back().injectPortIdx;
    }

    // InstanceGraph &instanceGraph;

    void mapConnectToReg(mlir::Operation *connectOp, mlir::Operation *regOp) {
      assert(regOp && "regOp is null");
      assert(isRegisterLike(regOp));
      assert(connectOp && "connectOp is null");
      assert(isConnectLike(connectOp));
      assert(regOp && !regOp->getName().getStringRef().empty());
      connectToRegister()[connectOp] = regOp;
    }

    // Global
    llvm::DenseMap<mlir::Operation *, mlir::Operation *> &connectToRegister() {
      return globalState.connectToRegister;
    }
    z3::context &ctx() const {
      assert(globalState.z3ctx && "z3::context not set");
      return *globalState.z3ctx;
    }
    CircuitOp &circuit() { return globalState.circuit; }
    FModuleOp &top() { return globalState.top; }
    std::vector<z3::expr> &finalExprs() { return globalState.finalExprs; }
    std::vector<std::string> &nodeNames() { return globalState.nodeNames; }
    const circt::pnet::PetriNet *getPetriNet() {
      return stageState.petriNet.get();
    }

    using CallInfo = LocalState::InterproceduralState::CallInfo;
    using ActivationPoint = StageState::ActivationPoint;

    // Local
    std::vector<std::string> &fields() { return localState.fields; }
    std::vector<mlir::Operation *> &operationPath() {
      return localState.operationPath;
    }
    std::vector<CallInfo> &callerContext() {
      return localState.interproceduralState.callerContext;
    }

    // Stage state
    std::set<ActivationPoint> &activePoints() {
      return stageState.activePoints;
    }
    std::set<ActivationPoint> &nextActivePoints() {
      return stageState.nextActivePoints;
    }
    std::vector<AnnoPathValue> visited() { return stageState.visited; }

    bool isDone() const { return stageState.done; }
    void setDone() { stageState.done = true; }
    int getStageNumber() const { return stageState.stageNumber; }
    void incrementStageNumber() { stageState.stageNumber++; }

    // InstanceGraph &getInstanceGraph() { return instanceGraph; }

    FIRRTLPrintPetriNetPassState(CircuitOp circuit, FModuleOp top) {
      globalState = GlobalState(circuit, top);
      stageState.petriNet = std::make_unique<circt::pnet::PetriNet>();
    }

    FIRRTLPrintPetriNetPassState(const FIRRTLPrintPetriNetPassState &) = delete;
    FIRRTLPrintPetriNetPassState &
    operator=(const FIRRTLPrintPetriNetPassState &) = delete;
    FIRRTLPrintPetriNetPassState(FIRRTLPrintPetriNetPassState &&) = default;
    FIRRTLPrintPetriNetPassState &
    operator=(FIRRTLPrintPetriNetPassState &&) = default;

    void dump(llvm::raw_ostream &os, unsigned depth = 0) const {
      os << indent(depth) << "=== FIRRTLPrintPetriNetPassState ===\n";

      // -------------------------
      // LocalState
      // -------------------------
      os << indent(depth + 1) << "LocalState\n";

      os << indent(depth + 2) << "fields:\n";
      for (const auto &f : localState.fields)
        os << indent(depth + 3) << "- " << f << "\n";

      os << indent(depth + 2) << "operationPath:\n";
      for (auto *op : localState.operationPath)
        os << indent(depth + 3) << "- " << *op << "\n";

      os << indent(depth + 2) << "callerContext:\n";
      for (const auto &ci : localState.interproceduralState.callerContext) {
        os << indent(depth + 3) << "CallInfo\n";
        os << indent(depth + 4) << "caller    : " << ci.caller->getName()
           << "\n";
        os << indent(depth + 4) << "callsite  : " << *ci.callsite << "\n";
        os << indent(depth + 4) << "callee    : " << ci.callee->getName()
           << "\n";
        os << indent(depth + 4) << "injectIdx : " << ci.injectPortIdx << "\n";
      }

      // // -------------------------
      // // StageState
      // // -------------------------
      // os << indent(depth + 1) << "StageState\n";
      // os << indent(depth + 2) << "stageNumber : " << stageState.stageNumber
      //    << "\n";
      // os << indent(depth + 2)
      //    << "done        : " << (stageState.done ? "true" : "false") << "\n";

      // os << indent(depth + 2) << "activePoints ("
      //    << stageState.activePoints.size() << ")\n";
      // for (const auto &ap : stageState.activePoints) {
      //   os << indent(depth + 3) << "- op: " << *ap.op << "\n";
      // }

      // os << indent(depth + 2) << "nextActivePoints ("
      //    << stageState.nextActivePoints.size() << ")\n";
      // for (const auto &ap : stageState.nextActivePoints) {
      //   os << indent(depth + 3) << "- op: " << *ap.op << "\n";
      // }

      // os << indent(depth + 2) << "visited (" << stageState.visited.size()
      //    << ")\n";
      // for (auto anno : stageState.visited)
      //   os << indent(depth + 3) << "- " << anno << "\n";

      // os << indent(depth + 2)
      //    << "petriNet: " << (stageState.petriNet ? "present" : "null") <<
      //    "\n";

      //   // -------------------------
      //   // GlobalState
      //   // -------------------------
      //   os << indent(depth + 1) << "GlobalState\n";

      //   os << indent(depth + 2)
      //      << "connectToRegister: " << globalState.connectToRegister.size()
      //      << " entries\n";

      //   os << indent(depth + 2)
      //      << "z3ctx: " << (globalState.z3ctx ? "initialized" : "null") <<
      //      "\n";

      //   os << indent(depth + 2) << "finalExprs: " <<
      //   globalState.finalExprs.size()
      //      << "\n";

      //   os << indent(depth + 2) << "nodeNames: " <<
      //   globalState.nodeNames.size()
      //      << "\n";

      //   os << indent(depth + 2)
      //      << "circuit: " << (globalState.circuit ? "set" : "null") << "\n";

      //   os << indent(depth + 2) << "top: " << (globalState.top ? "set" :
      //   "null")
      //      << "\n";

      //   os << indent(depth) << "====================================\n";

      //   // -------------------------
      //   // Current Final Expressions
      //   // -------------------------
      //   for (unsigned long i = 0; i < globalState.nodeNames.size(); i++) {
      //     auto name = globalState.nodeNames[i];
      //     auto expr = globalState.finalExprs[i];
      //     os << "" << name << ":\n";
      //     os << expr.to_string() << "\n\n";
      //   }
    }
  };

  struct RegisterArtifacts {
    std::shared_ptr<Z3Graph> subgraph;
    std::shared_ptr<Z3Graph> quasiSubgraph;
    z3::expr finalExpr;
    z3::expr normalized_expr;
    std::unique_ptr<Petrinet> petri;
    std::vector<
        std::pair<std::shared_ptr<Petrinet::Transition>, std::vector<z3::expr>>>
        counterExpressions;

    RegisterArtifacts(z3::context &ctx) : finalExpr(ctx), normalized_expr(ctx) {}
  };

  static bool ensureDir(StringRef dir, mlir::Operation *op, mlir::Pass *pass) {
    if (std::error_code ec = llvm::sys::fs::create_directories(dir)) {
      llvm::errs() << "failed to create directory '" << dir
                   << "': " << ec.message();
      return false;
    }
    return true;
  }

  static std::string normalizeRegisterSubgraphName(const std::string &s) {
    auto pos = s.find('~');
    std::string out = (pos == std::string::npos) ? s : s.substr(pos + 1);

    for (char &c : out) {
      if (c == '/' || c == ':' || c == '>' || c == '|')
        c = '.';
    }

    return out;
  }

  static std::string normalizeAfterArrow(const std::string &s) {
    auto pos = s.rfind('>');
    std::string out = (pos == std::string::npos) ? s : s.substr(pos + 1);

    for (char &c : out) {
      if (c == '/' || c == ':' || c == '>')
        c = '.';
    }

    return out;
  }

  static bool
  withOutputFile(StringRef filename,
                 llvm::function_ref<void(llvm::raw_ostream &)> emitFile,
                 mlir::Operation *op, mlir::Pass *pass) {
    std::error_code ec;
    llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "failed to open '" << filename << "': " << ec.message();
      return false;
    }
    emitFile(os);
    return true;
  }

  static void insertPlaceCounters(
      FIRRTLPrintPetriNetPassState &state,
      std::unordered_map<std::string, RegisterArtifacts> &artifactsMap,
      CounterInserter &counterInserter) {

    int totalPlanned = 0;

    for (auto &[reg, art] : artifactsMap) {
      for (const auto &place : art.petri->getPlaces()) {
        auto expr = place->expr;
        std::string name = "p" + std::to_string(place->id);
        llvm::errs() << "Inserting counter for place " << place->name
                     << " of register " << reg << "\n";

        counterInserter.insertPerfCounters(
            expr, name, state.globalState.targetCache, place->name, false);
        totalPlanned++;
      }
    }

    llvm::errs() << "Finished inserting counters for places: "
                 << counterInserter.numCountersInserted << "/" << totalPlanned
                 << "\n";
  }

  static void insertIncomingTransitionEdgeCounters(
      FIRRTLPrintPetriNetPassState &state,
      std::unordered_map<std::string, RegisterArtifacts> &artifactsMap,
      CounterInserter &counterInserter) {
    int totalPlanned = 0;

    for (auto &[reg, art] : artifactsMap) {
      for (auto &[transition, expressions] : art.counterExpressions) {
        const auto &incomingEdges = transition->incomingEdges;

        if (incomingEdges.size() != expressions.size()) {
          llvm::errs() << "[WARN] transition " << transition->id
                       << " has mismatched incomingEdges ("
                       << incomingEdges.size() << ") and expressions ("
                       << expressions.size() << ")\n";
        }

        size_t n = std::min(incomingEdges.size(), expressions.size());
        for (size_t i = 0; i < n; ++i) {
          auto &expr = expressions[i];
          auto &edge = incomingEdges[i];
          auto &place = edge->from;

          (void)place;

          std::string name =
              "t" + std::to_string(transition->id) + "_" + std::to_string(i);

          counterInserter.insertPerfCounters(expr, name,
                                             state.globalState.targetCache);
          totalPlanned++;
        }
      }
    }

    llvm::errs()
        << "Finished inserting counters for incoming transition edges, "
           "with total of "
        << counterInserter.numCountersInserted << "/" << totalPlanned
        << " counters inserted.\n";
  }

  static bool computeRegisterArtifacts(
      Z3Graph &z3graph, z3::context &ctx,
      llvm::function_ref<void(const Twine &)> emitErrorFn,
      llvm::function_ref<void()> signalFailure,
      std::unordered_map<std::string, RegisterArtifacts> &artifactsMap) {

    for (const auto &node : z3graph.nodes) {
      if (!node || !node->isRegister)
        continue;

      auto subgraph = z3graph.createSubgraph(node->name);
      if (!subgraph)
        continue;
      // llvm::errs() << "Is subgraph: " << node->name
      //              << " with size: " << subgraph->size() << "\n";

      if (subgraph->size() <= 1)
        continue;

      // llvm::errs() << "Is subgraph: " << node->name << "1"
      //              << "\n";

      if (!subgraph->valid) {
        emitErrorFn("failed to create subgraph for register '" + node->name +
                    "'");
        signalFailure();
        return false;
      }

      // if (!subgraph->domTree) {
      //   emitErrorFn("failed to build dominator tree for register subgraph '"
      //   +
      //               node->name + "'");
      //   signalFailure();
      //   return false;
      // }

      auto quasiSubgraph = subgraph->createQuasiRegisterGraph();
      if (!quasiSubgraph || !quasiSubgraph->valid) {
        emitErrorFn("failed to create quasi register graph for register '" +
                    node->name + "'");
        signalFailure();
        return false;
      }

      auto finalExpr = quasiSubgraph->buildInlinedExpr();
      finalExpr = finalExpr.simplify();

      // z3::set_param("pp.max_depth", 1000000);
      // z3::set_param("pp.max_num_lines", 1000000);
      // z3::set_param("pp.min_alias_size", 1000000000);

      auto normalized_expr = Z3Graph::Canonicalizer::toNNF(finalExpr);
      auto can = Z3Graph::Canonicalizer(ctx);
      // auto normalized_expr = finalExpr.simplify();
      normalized_expr = can.canonicalize(normalized_expr);
      // llvm::errs() << "Running toDNF for " << node->name << " with original
      // size: " << normalized_expr.to_string().size() << "\n";
      normalized_expr = Z3Graph::Canonicalizer::toNNF(normalized_expr);

      normalized_expr = can.canonicalize(normalized_expr);
      normalized_expr = normalized_expr.simplify();
      // llvm::errs() << "Finished toDNF for " << node->name << " with CNF size:
      // " << normalized_expr.to_string().size() << "\n";
      // TODO: Use CNF or not?

      auto countExprs = [](const auto &counters) {
        size_t total = 0;
        for (const auto &p : counters)
          total += p.second.size();
        return total;
      };

      auto petri = Petrinet::createFromZ3(normalized_expr, node->name);

      // --- Naive ---
      // auto counters = petri->getCountersNaive();
      // llvm::errs() << "Naive counter exprs for: " << node->name << " "
      //              << countExprs(counters) << "\n";

      // // --- Pairs ---
      // counters = petri->getCountersPairs();
      // llvm::errs() << "Pair counter exprs for: " << node->name << " "
      //             << countExprs(counters) << "\n";

      // // --- All subsets ---
      // counters = petri->getCountersAllSubsets();
      // llvm::errs() << "All-subset counter exprs for: " << node->name << " "
      //              << countExprs(counters) << "\n";
      // for(auto c : counters){
      // llvm::errs() << "Counter for " << c.first << " with " <<
      // c.second.size() << " expressions\n"; for(auto e : c.second){
      //   llvm::errs() << "Counter expression" << e.to_string() << "\n";
      // }
      // }

      petri->setDepth();
      if (!petri) {
        emitErrorFn("failed to build petrinet for register '" + node->name +
                    "'");
        signalFailure();
        return false;
      }

      RegisterArtifacts artifacts(ctx);
      artifacts.subgraph = std::move(subgraph);
      artifacts.quasiSubgraph = std::move(quasiSubgraph);
      artifacts.finalExpr = finalExpr;
      artifacts.normalized_expr = normalized_expr;
      artifacts.petri = std::move(petri);
      // artifacts.counterExpressions = std::move(counters);

      llvm::dbgs() << "Inserting rootgraph with name: " << node->name << "\n";
      artifactsMap.emplace(node->name, std::move(artifacts));
    }

    return true;
  }

  static bool emitRegisterArtifacts(
      const std::string &debugDirectory,
      const std::unordered_map<std::string, RegisterArtifacts> &artifactsMap,
      mlir::Operation *op, mlir::Pass *pass) {

    const std::string regDir = debugDirectory + "/register_subgraphs";
    const std::string domDir = debugDirectory + "/dominator_register_subgraphs";

    if (!ensureDir(regDir, op, pass))
      return false;
    if (!ensureDir(domDir, op, pass))
      return false;

    for (const auto &[rootName, artifacts] : artifactsMap) {
      if (!artifacts.subgraph || !artifacts.quasiSubgraph || !artifacts.petri)
        continue;

      // std::string subgraphFile =
      //     regDir + "/" + normalizeRegisterSubgraphName(rootName) + ".dot";
      // if (!withOutputFile(
      //         subgraphFile,
      //         [&](llvm::raw_ostream &os) {
      //           artifacts.subgraph->printFullGraphDot(os);
      //         },
      //         op, pass))
      //   return false;

      std::string quasiFile =
          regDir + "/" + normalizeRegisterSubgraphName(rootName) + "_quasi.dot";
      if (!withOutputFile(
              quasiFile,
              [&](llvm::raw_ostream &os) {
                artifacts.quasiSubgraph->printFullGraphDot(os);
              },
              op, pass))
        return false;

      std::string petriFile =
          regDir + "/" + normalizeRegisterSubgraphName(rootName) + ".petri.dot";
      if (!withOutputFile(
              petriFile,
              [&](llvm::raw_ostream &os) { artifacts.petri->writeGraph(os); },
              op, pass))
        return false;

      std::string t2pFile = regDir + "/" +
                            normalizeRegisterSubgraphName(rootName) +
                            ".petri.t2p.csv";
      if (!withOutputFile(
              t2pFile,
              [&](llvm::raw_ostream &os) {
                artifacts.petri->writeTransitionToPlaceMatrixCSV(os);
              },
              op, pass))
        return false;

      std::string p2tSlotFile = regDir + "/" +
                                normalizeRegisterSubgraphName(rootName) +
                                ".petri.p2t_slots.csv";
      if (!withOutputFile(
              p2tSlotFile,
              [&](llvm::raw_ostream &os) {
                artifacts.petri->writePlaceToTransitionSlotMatrixCSV(os);
              },
              op, pass))
        return false;

      std::string placeMapFile = regDir + "/" +
                                 normalizeRegisterSubgraphName(rootName) +
                                 ".petri.place_map.csv";

      if (!withOutputFile(
              placeMapFile,
              [&](llvm::raw_ostream &os) {
                artifacts.petri->writePlaceIdNameCSV(os);
              },
              op, pass))
        return false;

      z3::set_param("pp.max_depth", 1000000);
      z3::set_param("pp.max_num_lines", 1000000);
      z3::set_param("pp.min_alias_size", 1000000000);

      std::string exprFilename =
          regDir + "/" + normalizeRegisterSubgraphName(rootName) + ".expr";
      if (!withOutputFile(
              exprFilename,
              [&](llvm::raw_ostream &os) {
                os << artifacts.normalized_expr.to_string();
              },
              op, pass))
        return false;

      std::string countersFilename =
          regDir + "/" + normalizeRegisterSubgraphName(rootName) + ".counters";
      if (!withOutputFile(
              countersFilename,
              [&](llvm::raw_ostream &os) {
                for (const auto &entry : artifacts.counterExpressions) {
                  const auto &transition = entry.first;
                  const auto &exprs = entry.second;

                  for (const auto &expr : exprs) {
                    os << "t" << transition->getId() << ":";
                    os << expr.to_string() << "\n";
                  }
                }
              },
              op, pass))
        return false;

      // if (artifacts.subgraph->domTree) {
      //   std::string domFile = domDir + "/" + rootName + "_dom.dot";
      //   if (!withOutputFile(
      //           domFile,
      //           [&](llvm::raw_ostream &os) {
      //             artifacts.subgraph->domTree->printFullGraphDot(os);
      //           },
      //           op, pass))
      //     return false;
      // }

      // llvm::errs() << "Subgraph create for " << rootName << " with size: "
      //              << (artifacts.quasiSubgraph
      //                      ? artifacts.quasiSubgraph->nodes.size()
      //                      : 0)
      //              << " and root node: "
      //              << (artifacts.quasiSubgraph &&
      //              artifacts.quasiSubgraph->root
      //                      ? artifacts.quasiSubgraph->root->name
      //                      : "null")
      //              << "\n";
    }

    return true;
  }

  using ActivationPoint =
      FIRRTLPrintPetriNetPassState::StageState::ActivationPoint;
  using CallInfo =
      FIRRTLPrintPetriNetPassState::LocalState::InterproceduralState::CallInfo;

  FIRRTLPrintPetriNetPassState
  initializeState(CircuitOp circuit, FModuleOp top, mlir::Operation *op,
                  std::vector<CallInfo> &startingCallerContext) {
    FIRRTLPrintPetriNetPassState state(circuit, top);
    state.localState.interproceduralState.callerContext = startingCallerContext;
    auto starting_activation = ActivationPoint::create(state, op);

    state.stageState.activePoints.insert(starting_activation);
    return state;
  }

  static std::string
  getQualifiedPortNameFromBlockArg(mlir::BlockArgument barg,
                                   FIRRTLPrintPetriNetPassState &state) {
    auto target = createAnnoPathForCallerContext(state, barg);
    auto name = annoToString(target);
    state.globalState.targetCache[name] = target;
    return name;
  }

  static std::string indent(unsigned depth) {
    return std::string(depth * 2, ' ');
  }

  static std::string atomWithType(const z3::expr &e) {
    return "(" + e.to_string() + " : " + e.get_sort().to_string() + ")";
  }

  static std::string prettyZ3Inline(const z3::expr &e) {
    // typed leaf for ANY sort
    if (e.is_const() && e.num_args() == 0)
      return atomWithType(e);

    // Only pretty-print logical structure for Bool apps; otherwise fallback
    if (!e.is_bool())
      return e.to_string();

    if (e.is_app()) {
      auto kind = e.decl().decl_kind();

      if (kind == Z3_OP_NOT && e.num_args() == 1)
        return "¬" + prettyZ3Inline(e.arg(0));

      if (e.num_args() == 2) {
        std::string a = prettyZ3Inline(e.arg(0));
        std::string b = prettyZ3Inline(e.arg(1));
        switch (kind) {
        case Z3_OP_EQ:
          return a + " = " + b;
        case Z3_OP_AND:
          return a + " ∧ " + b;
        case Z3_OP_OR:
          return a + " ∨ " + b;
        case Z3_OP_IMPLIES:
          return a + " → " + b;
        default:
          break;
        }
      }
    }
    return e.to_string();
  }

  static bool isBool(const z3::expr &e) { return e.is_bool(); }

  static bool isBV(const z3::expr &e) { return e.is_bv(); }

  static unsigned bvWidth(const z3::expr &e) {
    assert(e.is_bv());
    return e.get_sort().bv_size();
  }

  std::pair<z3::expr, z3::expr>
  typeConflictResolution(const z3::expr &lhs, const z3::expr &rhs,
                         FIRRTLPrintPetriNetPassState &state,
                         bool bv_priority = true) {
    z3::context &ctx = lhs.ctx();
    // Simple cases
    if ((lhs.is_bool() && rhs.is_bool()) || (lhs.is_int() && rhs.is_int()) ||
        (lhs.is_bv() && rhs.is_bv() &&
         lhs.get_sort().bv_size() == rhs.get_sort().bv_size())) {
      return {lhs, rhs};
    }

    // Special case: BV1 <-> BV1 → treat as Bool
    if (lhs.is_bv() && rhs.is_bv() && lhs.get_sort().bv_size() == 1 &&
        rhs.get_sort().bv_size() == 1) {
      return {lhs == ctx.bv_val(1, 1), rhs == ctx.bv_val(1, 1)};
    }

    if (lhs.is_bv() && rhs.is_bv() &&
        lhs.get_sort().bv_size() == rhs.get_sort().bv_size()) {
      return {lhs, rhs};
    }

    // Matching int to bool: cast the boolean to an integer
    if (lhs.is_bool() && rhs.is_int()) {
      return {asInt(lhs, state), rhs};
    }

    if (lhs.is_int() && rhs.is_bool()) {
      return {lhs, asInt(rhs, state)};
    }

    // Concatenate to match bit widths (only if both are bitvectors)
    if (lhs.is_bv() && rhs.is_bv() &&
        lhs.get_sort().bv_size() != rhs.get_sort().bv_size()) {
      unsigned maxWidth =
          std::max(lhs.get_sort().bv_size(), rhs.get_sort().bv_size());
      return {asBV(lhs, maxWidth, state), asBV(rhs, maxWidth, state)};
    }

    // Bool and bitvector -> cast bool to bitvector
    if (lhs.is_bool() && rhs.is_bv()) {
      return {asBV(lhs, rhs.get_sort().bv_size(), state), rhs};
    }
    if (lhs.is_bv() && rhs.is_bool()) {
      return {lhs, asBV(rhs, lhs.get_sort().bv_size(), state)};
    }

    // Bitvector and Int --> depends on the operation.
    if (bv_priority) {
      if (lhs.is_bv() && rhs.is_int()) {
        return {lhs, asBV(rhs, lhs.get_sort().bv_size(), state)};
      }
      if (lhs.is_int() && rhs.is_bv()) {
        return {asBV(lhs, rhs.get_sort().bv_size(), state), rhs};
      }
    } else {
      if (lhs.is_bv() && rhs.is_int()) {
        return {asInt(lhs, state), rhs};
      }
      if (lhs.is_int() && rhs.is_bv()) {
        return {lhs, asInt(rhs, state)};
      }
    }

    llvm::dbgs() << "Type conflict resolution failed for:\n  lhs: "
                 << lhs.to_string() << " of sort " << lhs.get_sort().to_string()
                 << "\n  rhs: " << rhs.to_string() << " of sort "
                 << rhs.get_sort().to_string() << "\n";
    state.dump(llvm::dbgs());
    assert(false && "Unsupported type combination in typeConflictResolution");
  }

  std::pair<z3::expr, z3::expr>
  typeConflictResolutionForceBV(const z3::expr &lhs, const z3::expr &rhs,
                                FIRRTLPrintPetriNetPassState &state,
                                bool bv_priority = true) {
    auto [new_lhs, new_rhs] =
        typeConflictResolution(lhs, rhs, state, bv_priority);

    if (new_lhs.is_bool() && new_rhs.is_bool()) {
      unsigned bvSize = 1;
      return {asBV(new_lhs, bvSize, state), asBV(new_rhs, bvSize, state)};
    }

    return {new_lhs, new_rhs};
  }

  static std::string reindentNewlines(std::string s, unsigned depth) {
    const std::string pad = "\n" + indent(depth);
    // Replace every '\n' with '\n' + current indent
    size_t pos = 0;
    while ((pos = s.find('\n', pos)) != std::string::npos) {
      s.replace(pos, 1, pad);
      pos += pad.size();
    }
    return s;
  }

  static std::string z3ToStringIndented(const z3::expr &e, unsigned depth) {
    return indent(depth) + reindentNewlines(e.to_string(), depth);
  }

  // Hacky just thrown together for faster debugging. Nothing special
  static std::string prettyZ3(const z3::expr &e, unsigned depth = 0) {
    // typed leaf for ANY sort
    if (e.is_const() && e.num_args() == 0)
      return indent(depth) + atomWithType(e);

    // Only format logical structure for Bool; otherwise just print (still
    // typed leaves handled above).  IMPORTANT: reindent any newlines Z3
    // emits.
    if (!e.is_bool())
      return z3ToStringIndented(e, depth);

    if (!e.is_app())
      return z3ToStringIndented(e, depth);

    auto kind = e.decl().decl_kind();

    // NOT
    if (kind == Z3_OP_NOT && e.num_args() == 1) {
      if (isAtom(e))
        return indent(depth) + "(" + prettyZ3Inline(e) + ")";
      return indent(depth) + "(\n" + indent(depth + 1) + "¬\n" +
             prettyZ3(e.arg(0), depth + 1) + "\n" + indent(depth) + ")";
    }

    // Binary ops
    if (e.num_args() == 2) {
      const z3::expr lhsE = e.arg(0);
      const z3::expr rhsE = e.arg(1);

      std::string op;
      switch (kind) {
      case Z3_OP_EQ:
        op = "=";
        break;
      case Z3_OP_AND:
        op = "∧";
        break;
      case Z3_OP_OR:
        op = "∨";
        break;
      case Z3_OP_IMPLIES:
        op = "→";
        break;
      default:
        return z3ToStringIndented(e, depth);
      }

      if (isAtom(lhsE) && isAtom(rhsE)) {
        return indent(depth) + "(" + prettyZ3Inline(lhsE) + " " + op + " " +
               prettyZ3Inline(rhsE) + ")";
      }

      return indent(depth) + "(\n" + prettyZ3(lhsE, depth + 1) + "\n" +
             indent(depth + 1) + op + "\n" + prettyZ3(rhsE, depth + 1) + "\n" +
             indent(depth) + ")";
    }

    return z3ToStringIndented(e, depth);
  }

  z3::goal to_tseitin_cnf(const z3::expr &formula,
                          FIRRTLPrintPetriNetPassState &state) {
    // assert(formula.get_sort().is_bool() && "Expected Bool formula");

    z3::context &ctx = state.ctx();

    z3::goal g(ctx);
    g.add(formula);

    z3::tactic pipeline =
        // z3::tactic(ctx, "simplify") & ---> this breaks things
        z3::tactic(ctx, "elim-term-ite");
    // z3::tactic(ctx, "propagate-values") &
    // z3::tactic(ctx, "solve-eqs");
    // z3::tactic(ctx, "tseitin-cnf");

    z3::apply_result result = pipeline(g);

    assert(result.size() == 1 && "Unexpected multiple goals");

    return result[0];
  }

  void init();
  mlir::Operation *findProgressSignal(
      circt::firrtl::CircuitOp circuitOp, std::string progressSignal,
      std::string moduleName, std::string topModule,
      std::vector<FIRRTLPrintPetriNetPassState::LocalState::
                      InterproceduralState::CallInfo> &callerContext);

  static bool isEnd(mlir::Operation *op) {
    return llvm::isa<firrtl::RegOp>(op) || llvm::isa<firrtl::RegResetOp>(op);
  }

  // TOOD: Fix this so that in each iteration we run.
  static bool isRegisterLike(mlir::Operation *op) {
    return llvm::isa<firrtl::RegOp>(op) || llvm::isa<firrtl::RegResetOp>(op);
  }

  static bool isDefinition(mlir::Operation *op) {
    return llvm::isa<firrtl::RegOp>(op) || llvm::isa<firrtl::RegResetOp>(op) ||
           llvm::isa<firrtl::NodeOp>(op);
  }

  static bool isWireLike(mlir::Operation *op) {
    return llvm::isa<firrtl::WireOp>(op);
  }

  static bool isConnectLike(mlir::Operation *op) {
    return llvm::isa<firrtl::FConnectLike>(op);
  }

  static bool isInstance(mlir::Operation *op) {
    return llvm::isa<firrtl::FInstanceLike>(op);
  }

  static bool isNodeOp(mlir::Operation *op) {
    return llvm::isa<firrtl::NodeOp>(op);
  }

  static bool isWhenOp(mlir::Operation *op) {
    return llvm::isa<firrtl::WhenOp>(op);
  }

  static std::optional<unsigned> getFIRRTLBitWidth(mlir::Value v) {
    if (auto firTy =
            mlir::dyn_cast<circt::firrtl::FIRRTLBaseType>(v.getType())) {
      int64_t w = firTy.getBitWidthOrSentinel();
      if (w > 0)
        return static_cast<unsigned>(w);
    }

    return std::nullopt;
  }

  static std::optional<uint64_t>
  getFIRRTLBitWidth(circt::firrtl::FIRRTLType ty) {
    using namespace circt::firrtl;

    // Helper to combine optionals safely.
    auto addOpt = [](std::optional<uint64_t> a,
                     std::optional<uint64_t> b) -> std::optional<uint64_t> {
      if (!a || !b)
        return std::nullopt;
      return *a + *b;
    };

    // Base cases + recursion via TypeSwitch.
    return llvm::TypeSwitch<FIRRTLType, std::optional<uint64_t>>(ty)
        // Ground integer types
        .Case<UIntType>([](UIntType t) -> std::optional<uint64_t> {
          auto w = t.getWidth();
          if (!w)
            return std::nullopt;
          return static_cast<uint64_t>(*w);
        })
        .Case<SIntType>([](SIntType t) -> std::optional<uint64_t> {
          auto w = t.getWidth();
          if (!w)
            return std::nullopt;
          return static_cast<uint64_t>(*w);
        })

        // 1-bit ground types
        .Case<ClockType>([](ClockType) -> std::optional<uint64_t> { return 1; })
        .Case<ResetType>([](ResetType) -> std::optional<uint64_t> { return 1; })
        .Case<AsyncResetType>(
            [](AsyncResetType) -> std::optional<uint64_t> { return 1; })

        // Analog is still "width" in FIRRTL (though semantically different)
        .Case<AnalogType>([](AnalogType t) -> std::optional<uint64_t> {
          auto w = t.getWidth();
          if (!w)
            return std::nullopt;
          return static_cast<uint64_t>(*w);
        })

        // Vectors: size * element width
        .Case<FVectorType>([&](FVectorType t) -> std::optional<uint64_t> {
          auto elemW = getFIRRTLBitWidth(t.getElementType());
          if (!elemW)
            return std::nullopt;
          return (*elemW) * static_cast<uint64_t>(t.getNumElements());
        })

        // Bundles: sum element widths (packed)
        .Case<BundleType>([&](BundleType t) -> std::optional<uint64_t> {
          std::optional<uint64_t> total = 0;
          for (auto e : t.getElements()) {
            // e.type is a FIRRTLType
            auto w = getFIRRTLBitWidth(e.type);
            total = addOpt(total, w);
            if (!total)
              return std::nullopt;
          }
          return total;
        })

        // Open bundles (if you encounter them): same rule as BundleType
        .Case<OpenBundleType>([&](OpenBundleType t) -> std::optional<uint64_t> {
          std::optional<uint64_t> total = 0;
          for (auto e : t.getElements()) {
            auto w = getFIRRTLBitWidth(e.type);
            total = addOpt(total, w);
            if (!total)
              return std::nullopt;
          }
          return total;
        })

        // Fallback: unknown / unsupported FIRRTL types
        .Default(
            [](FIRRTLType) -> std::optional<uint64_t> { return std::nullopt; });
  }

  static llvm::StringRef getModuleName(mlir::Operation *op) {
    auto module = op->getParentOfType<firrtl::FModuleOp>();
    assert(module && "register-like op must be inside a module");
    return module.getName();
  }

  static AnnoPathValue getTargetAnno(mlir::Operation *op,
                                     FIRRTLPrintPetriNetPassState &state) {
    assert(isDefinition(op) && "op is not register like");

    llvm::StringRef regName;
    if (auto regOp = llvm::dyn_cast<firrtl::RegOp>(op)) {
      regName = regOp.getName();
    } else if (auto regResetOp = llvm::dyn_cast<firrtl::RegResetOp>(op)) {
      regName = regResetOp.getName();
    } else if (auto nodeOp = llvm::dyn_cast<firrtl::NodeOp>(op)) {
      regName = nodeOp.getName();
    } else {
      llvm_unreachable("not register-like");
    }

    llvm::StringRef moduleName = getModuleName(op);
    return createAnnoPathForCallerContext(state, op);
  }
  static AnnoPathValue
  createAnnoPathForCallerContext(FIRRTLPrintPetriNetPassState &state,
                                 mlir::BlockArgument barg) {
    auto *owner = barg.getOwner();
    assert(owner && "BlockArgument must have an owner block");

    auto *moduleOp = owner->getParentOp();
    assert(moduleOp && "BlockArgument expected to be inside an op");

    auto fModule = llvm::dyn_cast<circt::firrtl::FModuleOp>(moduleOp);
    assert(fModule &&
           "BlockArgument expected to be inside a firrtl::FModuleOp");

    unsigned portNo = barg.getArgNumber();
    auto ports = fModule.getPorts();
    assert(portNo < ports.size() &&
           "BlockArgument index out of range for module ports");

    llvm::SmallVector<circt::firrtl::InstanceOp, 8> path;
    path.reserve(state.callerContext().size());

    for (const auto &ci : state.callerContext()) {
      auto inst = llvm::dyn_cast<circt::firrtl::InstanceOp>(ci.callsite);
      assert(inst && "callerContext.callsite must be an InstanceOp");
      path.push_back(inst);
    }

    return AnnoPathValue(path, PortAnnoTarget(fModule, portNo), 0);
  }

  static AnnoPathValue
  createAnnoPathForCallerContext(FIRRTLPrintPetriNetPassState &state,
                                 mlir::Operation *op) {

    if (state.callerContext().empty()) {
      return AnnoPathValue(op);
    }

    llvm::SmallVector<circt::firrtl::InstanceOp, 8> path;
    path.reserve(state.callerContext().size());

    for (const auto &ci : state.callerContext()) {
      auto inst = llvm::dyn_cast<circt::firrtl::InstanceOp>(ci.callsite);
      assert(inst && "callerContext.callsite must be an InstanceOp");
      path.push_back(inst);
    }

    const auto &leaf = state.callerContext().back();
    return AnnoPathValue(path, OpAnnoTarget(op), 0);
  }

  static std::string annoToString(const AnnoPathValue &annoPath) {
    // TODO: Add option to print out full annoPath values for debugging
    std::string s;
    llvm::raw_string_ostream os(s);
    // os << annoPath.ref;
    os << annoPath;
    return os.str();
  }

  // TODO: Is this correct?
  static bool isExpressionLike(mlir::Operation *op) {
    if (!op)
      return false;
    if (op->getName().getDialectNamespace() !=
        circt::firrtl::FIRRTLDialect::getDialectNamespace())
      return false;
    if (op->getNumResults() == 0)
      return false;
    if (llvm::isa<circt::firrtl::InvalidValueOp>(op))
      return true;
    if (isPure(op))
      return true;
    if (llvm::isa<mlir::InferTypeOpInterface>(op))
      return true;

    return false;
  }

  static bool isConstantLike(mlir::Operation *op) {
    return llvm::isa<firrtl::ConstantOp>(op);
  }

  static bool isNullExpr(const z3::expr &e) {
    Z3_ast ast = e; // z3::expr converts to Z3_ast
    return ast == nullptr;
  }

  z3::expr handleConstantLike(mlir::Operation *op,
                              FIRRTLPrintPetriNetPassState &state) {
    if (auto cst = llvm::dyn_cast<circt::firrtl::ConstantOp>(op)) {
      z3::context &ctx = state.ctx();

      llvm::APInt value = cst.getValue();
      mlir::Value result = cst.getResult();

      unsigned width = getFIRRTLBitWidth(result).value_or(value.getBitWidth());

      // TODO: Sometimes there are zero width values. In theory we should
      // ignore this, but we need to create a valid z3::expr.
      if (width == 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[PETRINET] Zero-width ConstantOp mapped to placeholder: "
                   << *op << "\n");

        return ctx.bool_val(false);
      }

      // If width is one, just make a boolean here
      if (width == 1) {
        bool bit = value.getBoolValue();

        LLVM_DEBUG(llvm::dbgs()
                   << "[PETRINET] ConstantOp BV1 -> Bool: "
                   << (bit ? "true" : "false") << " for op " << *op << "\n");

        return ctx.bool_val(bit);
      }

      auto firTy = llvm::cast<circt::firrtl::FIRRTLType>(result.getType());
      bool isSigned = firTy.isa<circt::firrtl::SIntType>();

      // --- Make a decimal string for Z3 (arbitrary precision safe) ---
      llvm::SmallString<64> tmp;
      value.toString(tmp, /*Radix=*/10, /*Signed=*/isSigned,
                     /*FormatAsCLiteral=*/false);
      std::string decStr = tmp.str().str();

      // Build BV literal
      z3::expr bv = ctx.bv_val(decStr.c_str(), width);

      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling ConstantOp, value: " << decStr
                 << ", width: " << width << ", signed: "
                 << (isSigned ? "true" : "false") << " for op " << *op << "\n");

      return bv;
    }
  }

  z3::expr handleValue(mlir::Value value, FIRRTLPrintPetriNetPassState &state) {
    if (auto defOp = value.getDefiningOp()) {
      return handleOp(defOp, value, state);
    }

    if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
      // Special case where we don't go through handleOp, we setup the correct
      // state here.
      FIRRTLPrintPetriNetPassState::LocalState ls =
          state.localState; // copy current local state for potential use in
                            // recursive calls
      auto res = handleBlockArgument(arg, state);
      state.localState = ls;
      return res;
    }
    ASSERT_STATE_DEBUG(false &&
                       "Value has no defining op and is not a block argument");
  }

  // TODO: handle entire FIRRTL specification
  z3::expr handleOp(mlir::Operation *op, mlir::Value value,
                    FIRRTLPrintPetriNetPassState &state) {
    LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Handling op: " << *op << "\n");
    state.operationPath().push_back(op);
    FIRRTLPrintPetriNetPassState::LocalState ls =
        state.localState; // copy current local state for potential use in
                          // recursive calls
    z3::expr e = handleOpMatch(op, value, state);
    if (isNullExpr(e)) {
      llvm::errs() << "\n[PETRINET ERROR] handleOpMatch returned null expr\n";
      llvm::errs() << "Expr: " << e.to_string()
                   << " sort:" << e.get_sort().to_string() << "\n";
      llvm::errs() << "  for op: " << *op << "\n";
      state.dump(llvm::errs());
      assert(false);
    }
    state.localState = ls; // restore local state to before recursive calls
    state.operationPath().pop_back();
    return e;
  }

  bool isMemoryLike(mlir::Operation *op) {
    return llvm::isa<firrtl::MemOp>(op);
  }

  z3::expr handleMemoryLike(mlir::Operation *op,
                            FIRRTLPrintPetriNetPassState &state) {
    using namespace circt::firrtl;

    auto mem = llvm::dyn_cast<MemOp>(op);
    if (!mem)
      ASSERT_STATE_DEBUG(false && "Expected MemOp in handleMemoryLike");

    FIRRTLType dataTy = mem.getDataType();

    unsigned dataW = 0;
    if (auto base = mlir::dyn_cast<FIRRTLBaseType>(dataTy)) {
      int64_t w = base.getBitWidthOrSentinel();
      // Dummy fallback if unknown:
      dataW = (w > 0) ? (unsigned)w : 1;
    } else {
      dataW = 1;
    }

    std::string memName = mem.getName().str();
    if (memName.empty())
      memName = "mem";

    std::string sym = "mem$" + memName;
    z3::context &ctx = state.ctx();

    z3::expr m = ctx.bv_const(sym.c_str(), dataW);

    return m;
  }

  z3::expr handleOpMatch(mlir::Operation *op, mlir::Value value,
                         FIRRTLPrintPetriNetPassState &state) {
    if (isDefinition(op)) {
      // llvm::errs() << "Handle register-like op: " << *op << "\n";
      return handleRegisterLike(op, state);
    } else if (isMemoryLike(op)) {
      return handleMemoryLike(op, state);
    } else if (isConnectLike(op)) {
      return handleConnectLike(op, value, state);
    } else if (isConstantLike(op)) {
      return handleConstantLike(op, state);
    } else if (isWireLike(op)) {
      return handleWireLike(op, state);
    } else if (isNodeOp(op)) {
      return handleNodeOp(op, state);
    }
    if (isWhenOp(op)) {
      assert(false && "Unhandled when OP");
      // return handleWhenOp(op, state);
    } else if (isExpressionLike(op)) {
      return handleExpressionLike(op, state);
    } else if (isInstance(op)) {
      return handleInstanceLike(op, value, state);
    } else {
      // LLVM_DEBUG(llvm::dbgs()
      //            << "[PETRINET] Unhandled op in handleOp: " << *op <<
      //            "\n");
      llvm::errs() << "\n[PETRINET ERROR] Unhandled op in handleOp: " << *op
                   << "\n";
      assert(false && "unsupported op");
    }
  }

  std::optional<circt::firrtl::FModuleOp>
  getModuleOfInstanceNode(circt::firrtl::InstanceOp &inst) {

    // Get InstanceGraph analysis
    auto &ig = getAnalysis<InstanceGraph>();

    auto parentMod = inst->getParentOfType<FModuleOp>();
    assert(parentMod && "InstanceOp must be inside an FModuleOp");

    InstanceGraphNode *parentNode =
        ig.lookup(parentMod); // if this fails, see notes below
    assert(parentNode && "No InstanceGraphNode for parent module");

    InstanceRecord *rec = nullptr;
    for (InstanceRecord *r : *parentNode) {
      if (r->getInstance() == inst) {
        rec = r;
        break;
      }
    }
    assert(rec && "Could not find InstanceRecord for this InstanceOp");
    auto *targetNode = rec->getTarget();
    assert(targetNode && "InstanceRecord has null target");

    // Get the referenced FIRRTL module op.
    auto targetFMod = targetNode->getModule<FModuleOp>();
    if (!targetFMod) {
      LLVM_DEBUG(llvm::dbgs() << "Instance target is not an FModuleOp\n");
      return std::nullopt;
    }
    return targetFMod;
  }

  z3::expr stepIntoModule(FModuleOp &moduleOp,
                          FIRRTLPrintPetriNetPassState &state) {
    // TODO: Input port vs output port?
    // Lets assume for now that we only step into when we have output port? Is
    // this a correct assumption?

    // Look for connect that has this block argument as a destination
    LLVM_DEBUG({
      llvm::dbgs() << "[PETRINET] Stepping into module: " << moduleOp.getName()
                   << " via injected port argument #"
                   << state.getCurrInjectedPort() << " ("
                   << moduleOp.getPortName(state.getCurrInjectedPort()).str()
                   << ")\n";
    });
    auto *body = moduleOp.getBodyBlock();
    assert(body && "FModuleOp must have a body block");
    assert(state.getCurrInjectedPort() < body->getNumArguments() &&
           "injectPortIdx out of range for module arguments");
    mlir::Value portArg = body->getArgument(state.getCurrInjectedPort());
    mlir::Operation *op = findDrivingOp(portArg, state);
    assert(op &&
           "Expected to find an operation driving the injected port argument");
    auto conn = llvm::dyn_cast<circt::firrtl::FConnectLike>(op);
    assert(conn && "Expected driving operation to be a connect");
    return handleOp(conn, conn.getDest(), state);
  }

  // We add a value here because isntanceOp can have multiple returns. In this
  // case we know that.

  z3::expr handleInstanceLike(mlir::Operation *op, mlir::Value value,
                              FIRRTLPrintPetriNetPassState &state) {
    circt::firrtl::InstanceOp inst =
        mlir::dyn_cast<circt::firrtl::InstanceOp>(op);
    assert(inst && "Expected current op to be InstanceOp");

    auto getInstResultIndex = [&](mlir::Value v) -> std::optional<unsigned> {
      if (auto r = v.dyn_cast<mlir::OpResult>()) {
        if (r.getOwner() == inst.getOperation())
          return r.getResultNumber();
      }
    };

    std::optional<unsigned> idxOpt = getInstResultIndex(value);
    bool usedSrc = true;
    assert(idxOpt &&
           "The value passed to instance is not a result of the instance");

    assert(idxOpt && "Connect does not reference an instance port directly");
    unsigned matchingResultIdx = *idxOpt;

    auto portName = inst.getPortName(matchingResultIdx);
    auto portDir = inst.getPortDirection(matchingResultIdx);

    // Input ports are handled specially, we should not have to traverse into
    // the module (I think?)
    if (portDir == circt::firrtl::Direction::In) {
      usedSrc = false;
      auto *driver = findDrivingOp(value, state);
      ASSERT_STATE_DEBUG(driver && "Expected to find a driving drivereration "
                                   "for the instance input port");
      return handleOp(driver, value, state);
    }

    auto portTy = mlir::cast<circt::firrtl::FIRRTLType>(
        inst.getResult(matchingResultIdx).getType());
    LLVM_DEBUG(llvm::dbgs()
               << "[PETRINET] Connect references instance port result #"
               << matchingResultIdx << " (" << (usedSrc ? "src" : "dest")
               << "): " << portName.str() << " dir=" << (unsigned)portDir
               << " type=" << portTy << "\n");
    auto FModuleOpt = getModuleOfInstanceNode(inst);

    if (!FModuleOpt) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Could not resolve target module for instance\n");
      // TODO: Create a new primitive value
      assert(false && "Could not resolve target module for instance");
    }

    // state.callerContext().push_back(FModuleOpt->getOperation());
    auto parentOp = op->getParentOfType<FModuleOp>();
    assert(parentOp && "Stepping into function requires but parent is unkown");
    FIRRTLPrintPetriNetPassState::CallInfo ci{
        parentOp, op, FModuleOpt->getOperation(), matchingResultIdx};

    state.callerContext().push_back(ci);

    return stepIntoModule(*FModuleOpt, state);
  }

  z3::expr handleRegisterLike(mlir::Operation *op,
                              FIRRTLPrintPetriNetPassState &state) {
    LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Handling register op: " << *op
                            << ", done recursing in this stage.\n");

    auto anno = getTargetAnno(op, state);
    LLVM_DEBUG(llvm::dbgs()
               << "[PETRINET] Generated anno for register: " << anno << "\n");
    auto name = annoToString(anno);
    state.globalState.targetCache[name] = anno;
    LLVM_DEBUG(llvm::dbgs()
               << "[PETRINET] Generated anno for register: " << name << "\n");

    bool alreadyVisited = std::find(state.stageState.visited.begin(),
                                    state.stageState.visited.end(),
                                    anno) != state.stageState.visited.end();

    if (llvm::isa<circt::firrtl::RegResetOp>(op) ||
        llvm::isa<circt::firrtl::RegOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "[PETRINET] This is a reg or regreset op, "
                                 "adding to initial state\n");

      // llvm::errs() << "[PETRINET] Adding register to initial state: " << name
      //              << "\n";
      state.globalState.registers.insert(name);
    }

    if (!alreadyVisited) {
      auto ap =
          FIRRTLPrintPetriNetPassState::StageState::ActivationPoint::create(
              state, op);
      auto [it, inserted] = state.stageState.nextActivePoints.insert(ap);
      state.globalState.targets.push_back(anno);
    }

    std::optional<unsigned> wOpt = getFIRRTLBitWidth(op->getResult(0));
    LLVM_DEBUG(llvm::dbgs()
               << "[PETRINET] Register name: " << name << ", bit width: "
               << (wOpt ? std::to_string(*wOpt) : "unknown") << "\n");
    if (wOpt && *wOpt > 1) {
      return state.ctx().bv_const(name.c_str(), *wOpt);
    } else {
      return state.ctx().bool_const(name.c_str());
    }
  }

  z3::expr handleWireLike(mlir::Operation *op,
                          FIRRTLPrintPetriNetPassState &state) {
    // Add this to expression?
    LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Handling wire op: " << *op << "\n");
    auto *driver = findDrivingOp(op->getResult(0), state);
    ASSERT_STATE_DEBUG(driver &&
                       "Expected to find a driving drivereration for the wire");
    return handleOp(driver, op->getResult(0), state);

    llvm_unreachable("Unhandled wire with no connects");
  }

  z3::expr asBool(const z3::expr &e, FIRRTLPrintPetriNetPassState &state) {
    z3::context &ctx = state.ctx();
    z3::sort s = e.get_sort();

    if (s.is_bool()) {
      return e;
    }

    if (s.is_bv()) {
      unsigned w = s.bv_size();
      if (w == 1)
        return e == ctx.bv_val(1, 1);

      return e != ctx.bv_val(0, w);
    }

    if (s.is_int())
      return e != ctx.int_val(0);

    LLVM_DEBUG(llvm::dbgs() << "Cannot cast to Bool: " << e.to_string()
                            << " of sort " << e.get_sort().to_string() << "\n");

    assert(false && "Expression is neither Bool nor BitVec(1)");
    return e;
  }

  z3::expr asInt(const z3::expr &e, FIRRTLPrintPetriNetPassState &state,
                 bool isSigned = false) {
    if (e.get_sort().is_int())
      return e;
    z3::context &ctx = state.ctx();
    if (e.get_sort().is_bool()) {
      return z3::ite(e, ctx.int_val(1), ctx.int_val(0));
    }

    assert(e.get_sort().is_bv() && "Expected BitVec");
    return z3::bv2int(e, isSigned);
  }

  z3::expr asBV(const z3::expr &e, unsigned width,
                FIRRTLPrintPetriNetPassState &state) {
    z3::context &ctx = state.ctx();

    if (e.is_bv() && e.get_sort().bv_size() == width)
      return e;

    if (e.is_int()) {
      return z3::int2bv(width, e);
    }

    if (e.is_bv()) {
      unsigned w = e.get_sort().bv_size();
      if (w < width) {
        return z3::zext(e, width - w);
      }
      return e.extract(width - 1, 0);
    }

    if (e.is_bool()) {
      return z3::ite(e, ctx.bv_val(1, width), ctx.bv_val(0, width));
    }

    LLVM_DEBUG(llvm::dbgs() << "As BV failed for " << e.to_string()
                            << e.get_sort().to_string()
                            << " with width : " << width << "\n");
    assert(false && "Unsupported type in asBV");
    return e; // unreachable
  }

  static std::optional<circt::firrtl::FConnectLike>
  findConnectByDest(mlir::Value dest) {
    for (mlir::OpOperand &use : dest.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (auto conn = mlir::dyn_cast<circt::firrtl::FConnectLike>(user)) {
        if (conn.getDest() == dest)
          return conn;
      }
    }
    return std::nullopt;
  }

  static std::optional<circt::firrtl::FConnectLike>
  findConnectBySrc(mlir::Value src) {
    for (mlir::OpOperand &use : src.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (auto conn = mlir::dyn_cast<circt::firrtl::FConnectLike>(user)) {
        if (conn.getSrc() == src)
          return conn;
      }
    }
    return std::nullopt;
  }

  circt::firrtl::FIRRTLType
  getFieldOfBundleType(mlir::Type ty, FIRRTLPrintPetriNetPassState &state) {

    LLVM_DEBUG({
      llvm::dbgs() << "[PETRINET] Getting field of bundle type for type: " << ty
                   << " and state.localState.fields:";
      for (int i = state.localState.fields.size() - 1; i >= 0; --i) {
        auto f = state.localState.fields[i];
        llvm::dbgs() << "." << f;
      }
      llvm::dbgs() << "\n";
    });
    auto firTy = llvm::dyn_cast<circt::firrtl::BundleType>(ty);

    // (2) Use state.localState.fields to index into the type.
    circt::firrtl::FIRRTLType cur = firTy;
    for (int i = state.localState.fields.size() - 1; i >= 0; --i) {
      const std::string &field = state.localState.fields[i];
      // Expect bundles when consuming a subfield name.
      if (auto bundleTy = llvm::dyn_cast<circt::firrtl::BundleType>(cur)) {
        bool found = false;
        for (auto elem : bundleTy.getElements()) {
          // elem.name is a StringAttr
          if (elem.name.getValue() == field) {
            cur = elem.type;
            found = true;
            break;
          }
        }
        assert(found && "Subfield name not found in BundleType");
        continue;
      }

      // Some FIRRTL flows also use OpenBundleType (if present in your CIRCT
      // version)
      if (auto openBundleTy =
              llvm::dyn_cast<circt::firrtl::OpenBundleType>(cur)) {
        bool found = false;
        for (auto elem : openBundleTy.getElements()) {
          if (elem.name.getValue() == field) {
            cur = elem.type;
            found = true;
            break;
          }
        }
        assert(found && "Subfield name not found in OpenBundleType");
        continue;
      }

      assert(false &&
             "Tried to apply SubfieldOp field to a non-bundle FIRRTL type");
    }
    return cur;
  }

  z3::expr stepOutofModule(mlir::BlockArgument &barg,
                           FIRRTLPrintPetriNetPassState &state) {
    FIRRTLPrintPetriNetPassState::CallInfo callinfo =
        state.callerContext().back();
    state.callerContext().pop_back();
    // llvm::dbgs() << "Stepping out to: " << callinfo.caller->getName() <<
    // "\n";
    assert(callinfo.callee && "Callee module in callsite context is null");
    auto calleeName =
        callinfo.callsite->getParentOfType<circt::firrtl::FModuleOp>()
            .getName();
    LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Stepping out of module into "
                            << calleeName << "\n");

    unsigned portIdx = barg.getArgNumber();
    auto callsiteResult = callinfo.callsite->getResult(portIdx);
    auto connOpt = findConnectByDest(callsiteResult);
    if (!connOpt) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] No connect found with dest == callsite result #"
                 << portIdx << " (" << callsiteResult << ")\n");

      z3::context &ctx = state.ctx();

      // Generate unique name
      std::string name =
          "unconn_" + std::to_string(state.globalState.nameCounter++);

      // Determine FIRRTL width
      auto widthOpt = getFIRRTLBitWidth(callsiteResult);

      if (!widthOpt || widthOpt.value() == 1) {
        return ctx.bool_const(name.c_str());
      }

      unsigned width = widthOpt.value();

      if (width == 0)
        width = 1; // Z3 does not support width 0

      return ctx.bv_const(name.c_str(), width);
    }

    auto conn = *connOpt;
    mlir::Value src = conn.getSrc();

    LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Found connect for callsite result #"
                            << portIdx << ": " << *conn.getOperation() << "\n");

    return handleValue(src, state);
  }

  z3::expr handleBlockArgumentTopLevel(mlir::BlockArgument &barg,
                                       FIRRTLPrintPetriNetPassState &state) {

    LLVM_DEBUG({
      llvm::dbgs() << "[PETRINET] Handling block argument: "
                   << "\n";

      if (auto firTy =
              llvm::dyn_cast<circt::firrtl::FIRRTLType>(barg.getType())) {
        llvm::dbgs() << "[PETRINET]  FIRRTL type: " << firTy << "\n";
      }

      llvm::dbgs() << "[PETRINET]  state.localState.fields:";
      for (int i = state.localState.fields.size() - 1; i >= 0; --i) {
        auto f = state.localState.fields[i];
        llvm::dbgs() << "." << f;
      }
      llvm::dbgs() << "\n";
    });

    auto ty = barg.getType();
    auto firTy = llvm::dyn_cast<circt::firrtl::FIRRTLType>(ty);
    assert(firTy && "Block argument is not a FIRRTLType");

    circt::firrtl::FIRRTLType fieldType = firTy;

    if (llvm::isa<circt::firrtl::BundleType>(firTy) ||
        llvm::isa<circt::firrtl::OpenBundleType>(firTy)) {

      fieldType = getFieldOfBundleType(firTy, state);

    } else {
      assert(state.localState.fields.empty() &&
             "Non-bundle block argument reached with non-empty "
             "state.localState.fields "
             "(stale subfield context)");
    }

    std::optional<unsigned> wOpt = getFIRRTLBitWidth(fieldType);
    auto name = getQualifiedPortNameFromBlockArg(barg, state);

    LLVM_DEBUG({
      llvm::dbgs() << "[PETRINET] Handling block argument: " << name
                   << ", bit width: "
                   << (wOpt ? std::to_string(*wOpt) : "unknown") << "\n";
      llvm::dbgs() << indent(2) << "callerContext:\n";
      for (const auto &ci :
           state.localState.interproceduralState.callerContext) {
        llvm::dbgs() << indent(3) << "CallInfo\n";
        llvm::dbgs() << indent(4) << "caller    : " << ci.caller->getName()
                     << "\n";
        llvm::dbgs() << indent(4) << "callsite  : " << *ci.callsite << "\n";
        llvm::dbgs() << indent(4) << "callee    : " << ci.callee->getName()
                     << "\n";
        llvm::dbgs() << indent(4) << "injectIdx : " << ci.injectPortIdx << "\n";
      }
      state.dump(llvm::dbgs());
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling block argument: " << name
                 << ", bit width: "
                 << (wOpt ? std::to_string(*wOpt) : "unknown") << "\n");
    });
    if (wOpt && *wOpt > 1) {
      z3::expr e = state.ctx().bv_const(name.c_str(), *wOpt);

      LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Block argument expr: "
                              << e.to_string() << "\n");
      return e;
    } else {
      return state.ctx().bool_const(name.c_str());
    }
  }

  z3::expr handleBlockArgument(mlir::BlockArgument &barg,
                               FIRRTLPrintPetriNetPassState &state) {
    if (state.callerContext().size() == 0) {
      // We are in the top-level module. Nothing to do here but create a
      // signal
      return handleBlockArgumentTopLevel(barg, state);
    } else {
      // We are stepping outside of the module
      return stepOutofModule(barg, state);
    }
  }

  z3::expr handleConnectLike(mlir::Operation *op, mlir::Value v,
                             FIRRTLPrintPetriNetPassState &state) {

    auto connect = llvm::cast<firrtl::FConnectLike>(op);
    if (!connect) {
      assert(false && "COnnect op not connect op? How did we get here?!");
    }
    if (connect.getDest() != v) {
      state.dump(llvm::dbgs());
      llvm::dbgs() << "Is connect like op: " << *connect << "\n";
      llvm::dbgs() << "Destination: " << connect.getDest() << "\n";
      llvm::dbgs() << "Matched value that led us here: " << v << "\n";
      llvm::dbgs() << "[PETRINET] Connect op dest does not match value: "
                   << "dest= " << connect.getDest() << ", \nvalue= " << v
                   << "For op " << *op << "and \nvalue:" << v << "\n";
      assert(false && "Connect op dest does not match value");
    }
    // Add this to expression?
    LLVM_DEBUG(llvm::dbgs()
               << "[PETRINET] Handling connectlike op: " << *connect
               << "with source: " << connect.getSrc() << "\n");
    if (auto ba = llvm::dyn_cast<BlockArgument>(connect.getSrc())) {
      return handleBlockArgument(ba, state);
    }

    assert(connect.getSrc() && "connect must have src");
    assert(connect.getSrc().getDefiningOp() &&
           "connect src must be defined by an op");
    auto src = connect.getSrc().getDefiningOp();
    return handleOp(src, connect.getSrc(), state);
  }

  z3::expr handleNodeOp(mlir::Operation *op,
                        FIRRTLPrintPetriNetPassState &state) {
    auto nodeOp = llvm::cast<circt::firrtl::NodeOp>(op);
    mlir::Value inVal = nodeOp.getInput();
    return handleValue(inVal, state);
  }

  z3::expr handleExpressionLike(mlir::Operation *op,
                                FIRRTLPrintPetriNetPassState &state) {
    // LLVM_DEBUG(llvm::dbgs()
    //            << "[PETRINET] Handling expressionlike op: " << *op <<
    //            "\n");
    if (auto mux = llvm::dyn_cast<circt::firrtl::MuxPrimOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Handling MuxPrimOp\n");

      auto ops = op->getOperands();
      assert(ops.size() == 3 && "mux expects exactly 3 operands");

      mlir::Value selVal = ops[0];
      mlir::Value trueVal = ops[1];
      mlir::Value falseVal = ops[2];

      z3::expr sel = asBool(handleValue(selVal, state), state);
      z3::expr tRaw = handleValue(trueVal, state);
      z3::expr fRaw = handleValue(falseVal, state);
      auto [tval, fval] = typeConflictResolution(tRaw, fRaw, state);

      LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Mux sel: " << sel.to_string()
                              << ", tval: " << tval.to_string()
                              << ", fval: " << fval.to_string() << " for op "
                              << *op << " final expression: "
                              << ite(sel, tval, fval).to_string() << "\n");

      return ite(sel, tval, fval);
    }

    if (auto mb = llvm::dyn_cast<circt::firrtl::MultibitMuxOp>(op)) {
      mlir::Value idxVal = mb.getIndex();

      unsigned idxW = getFIRRTLBitWidth(idxVal).value_or(1);
      z3::expr idx = asBV(handleValue(idxVal, state), idxW, state);

      auto inputs = mb.getInputs();
      ASSERT_STATE_DEBUG(inputs.size() >= 1 &&
                         "multibit_mux must have at least 1 input");

      z3::expr acc = asBV(handleValue(inputs[0], state),
                          getFIRRTLBitWidth(inputs[0]).value_or(1), state);

      // Note: operand order is v_{n-1} ... v_0, but "index==0 selects v_0".
      for (unsigned k = 0, n = inputs.size(); k < n; ++k) {
        mlir::Value v = inputs[n - 1 - k]; // v0 at back
        unsigned w = getFIRRTLBitWidth(v).value_or(1);
        z3::expr vk = asBV(handleValue(v, state), w, state);

        z3::expr selk = (idx == state.ctx().bv_val(k, idxW));
        acc = z3::ite(selk, vk, acc);
      }
      return acc;
    }

    if (auto andOp = llvm::dyn_cast<circt::firrtl::AndPrimOp>(op)) {
      auto ops = andOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "and expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];

      auto lhs = asBool(handleValue(lhsVal, state), state);
      auto rhs = asBool(handleValue(rhsVal, state), state);

      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling AndPrimOp, "
                 << "lhs: " << lhs.to_string()
                 << " (sort: " << lhs.get_sort().to_string() << "), "
                 << "rhs: " << rhs.to_string()
                 << " (sort: " << rhs.get_sort().to_string() << ") "
                 << "for op " << *op << "\n");

      return (lhs && rhs);
    }

    if (auto orOp = llvm::dyn_cast<circt::firrtl::OrPrimOp>(op)) {
      auto ops = orOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "or expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];

      auto lhs = asBool(handleValue(lhsVal, state), state);
      auto rhs = asBool(handleValue(rhsVal, state), state);

      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling OrPrimOp, lhs: " << lhs.to_string()
                 << ", rhs: " << rhs.to_string() << " for op " << *op << "\n");
      return (lhs || rhs);
    }

    if (auto eqOp = llvm::dyn_cast<circt::firrtl::EQPrimOp>(op)) {
      auto ops = eqOp->getOperands();
      assert(ops.size() == 2 && "eq expects exactly 2 operands");

      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];

      auto lhsTy = llvm::cast<circt::firrtl::FIRRTLType>(ops[0].getType());
      auto rhsTy = llvm::cast<circt::firrtl::FIRRTLType>(ops[1].getType());

      auto [lhs, rhs] = typeConflictResolution(
          handleValue(lhsVal, state), handleValue(rhsVal, state), state);

      LLVM_DEBUG({
        llvm::dbgs() << "[PETRINET] Handling EQPrimOp " << *op
                     << "\n  lhs: " << lhs.to_string()
                     << "\n  lhs sort: " << lhs.get_sort().to_string()
                     << "\n  rhs: " << rhs.to_string()
                     << "\n  rhs sort: " << rhs.get_sort().to_string() << "\n";
      });

      return (lhs == rhs); // Bool
    }
    if (auto neqOp = llvm::dyn_cast<circt::firrtl::NEQPrimOp>(op)) {
      auto ops = neqOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "neq expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];

      auto [lhs, rhs] = typeConflictResolution(
          handleValue(lhsVal, state), handleValue(rhsVal, state), state);
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling NEQPrimOp, lhs: " << lhs.to_string()
                 << ", rhs: " << rhs.to_string() << " for op " << *op << "\n");
      return (lhs != rhs);
    }

    if (auto notOp = llvm::dyn_cast<circt::firrtl::NotPrimOp>(op)) {
      auto ops = notOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 1 && "not expects exactly 1 operand");
      mlir::Value inputVal = ops[0];
      z3::expr input = asBool(handleValue(inputVal, state), state);
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling NotPrimOp, input: "
                 << input.to_string() << " for op " << *op << "\n");
      return !input;
    }

    if (auto xorOp = llvm::dyn_cast<circt::firrtl::XorPrimOp>(op)) {
      auto ops = xorOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "xor expects exactly 2 operands");

      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];

      auto [lhs, rhs] = typeConflictResolution(
          handleValue(lhsVal, state), handleValue(rhsVal, state), state);

      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling XorPrimOp, lhs: " << lhs.to_string()
                 << ", rhs: " << rhs.to_string() << " for op " << *op << "\n");

      // Some z3 bullshit to deal with.
      z3::expr result =
          (lhs.is_bool() && rhs.is_bool()) ? (lhs != rhs) : (lhs ^ rhs);
      return result;
    }

    if (auto orOp = llvm::dyn_cast<circt::firrtl::OrPrimOp>(op)) {
      auto ops = orOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "or expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];

      auto lhs = asBool(handleValue(lhsVal, state), state);
      auto rhs = asBool(handleValue(rhsVal, state), state);
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling OrPrimOp, lhs: " << lhs.to_string()
                 << ", rhs: " << rhs.to_string() << " for op " << *op << "\n");
      return (lhs || rhs);
    }

    if (auto addOp = llvm::dyn_cast<circt::firrtl::AddPrimOp>(op)) {
      auto ops = addOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "add expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];
      z3::expr lhs = asInt(handleValue(lhsVal, state), state);
      z3::expr rhs = asInt(handleValue(rhsVal, state), state);
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling AddPrimOp, lhs: " << lhs.to_string()
                 << ", rhs: " << rhs.to_string() << " for op " << *op << "\n");
      return (lhs + rhs);
    }
    if (auto mulOp = llvm::dyn_cast<circt::firrtl::MulPrimOp>(op)) {
      auto ops = mulOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "mul expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];

      // Determine FIRRTL result width (prefer result type)
      unsigned resW = 0;
      if (auto resTy = mlir::dyn_cast<circt::firrtl::FIRRTLBaseType>(
              mulOp.getResult().getType())) {
        int64_t w = resTy.getBitWidthOrSentinel();
        ASSERT_STATE_DEBUG(w > 0 && "mul result width must be known");
        resW = (unsigned)w;
      } else {
        ASSERT_STATE_DEBUG(false && "mul result must be FIRRTLBaseType");
      }

      // Get operand expressions, then coerce to BV of resW
      z3::expr lhsE = handleValue(lhsVal, state);
      z3::expr rhsE = handleValue(rhsVal, state);

      z3::expr lhs = asBV(lhsE, resW, state);
      z3::expr rhs = asBV(rhsE, resW, state);

      // BV multiplication is modulo 2^resW, which matches FIRRTL UInt
      // semantics
      z3::expr prod = lhs * rhs;

      // Defensive: ensure exact width in case asBV returns a different width
      if (prod.is_bv() && prod.get_sort().bv_size() != resW) {
        if (prod.get_sort().bv_size() < resW)
          prod = z3::zext(prod, resW - prod.get_sort().bv_size());
        else
          prod = prod.extract(resW - 1, 0);
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling MulPrimOp, lhs: " << lhs.to_string()
                 << ", rhs: " << rhs.to_string() << ", resW=" << resW
                 << " for op " << *op << "\n");

      return prod;
    }

    if (auto subOp = llvm::dyn_cast<circt::firrtl::SubPrimOp>(op)) {
      auto ops = subOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "sub expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];
      z3::expr lhs = asInt(handleValue(lhsVal, state), state);
      z3::expr rhs = asInt(handleValue(rhsVal, state), state);
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling SubPrimOp, lhs: " << lhs.to_string()
                 << ", rhs: " << rhs.to_string() << " for op " << *op << "\n");
      return (lhs - rhs);
    }

    if (auto leqOp = llvm::dyn_cast<circt::firrtl::LEQPrimOp>(op)) {
      auto ops = leqOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "leq expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];

      auto [lhs, rhs] = typeConflictResolutionForceBV(
          handleValue(lhsVal, state), handleValue(rhsVal, state), state);
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling LEQPrimOp, lhs: " << lhs.to_string()
                 << ", rhs: " << rhs.to_string() << " for op " << *op << "\n");
      return (lhs <= rhs);
    }

    if (auto ltOp = llvm::dyn_cast<circt::firrtl::LTPrimOp>(op)) {
      auto ops = ltOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "lt expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];
      auto [lhs, rhs] = typeConflictResolutionForceBV(
          handleValue(lhsVal, state), handleValue(rhsVal, state), state);
      return (lhs < rhs);
    }

    if (auto geqOp = llvm::dyn_cast<circt::firrtl::GEQPrimOp>(op)) {
      auto ops = geqOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "geq expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];
      auto [lhs, rhs] = typeConflictResolutionForceBV(
          handleValue(lhsVal, state), handleValue(rhsVal, state), state);
      return (lhs >= rhs);
    }

    if (auto gtOp = llvm::dyn_cast<circt::firrtl::GTPrimOp>(op)) {
      auto ops = gtOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "gt expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];
      auto [lhs, rhs] = typeConflictResolutionForceBV(
          handleValue(lhsVal, state), handleValue(rhsVal, state), state);
      return (lhs > rhs);
    }

    if (auto orrOp = llvm::dyn_cast<circt::firrtl::OrRPrimOp>(op)) {
      auto inputVal = orrOp.getOperand();
      z3::expr input = handleValue(inputVal, state);
      // Somtimes in is reduced to bool, so we can just fowards this value
      // here
      if (input.is_bool()) {
        return input;
      }
      unsigned w = input.get_sort().bv_size();
      z3::expr zero = state.ctx().bv_val(0, w);
      z3::expr is_nonzero = (input != zero);
      LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Handling OrRPrimOp, input: "
                              << input.to_string()
                              << ", is_nonzero: " << is_nonzero.to_string()
                              << " for op " << *op << "\n");
      return z3::ite(is_nonzero, state.ctx().bv_val(1, 1),
                     state.ctx().bv_val(0, 1));
    }

    if (auto andrOp = llvm::dyn_cast<circt::firrtl::AndRPrimOp>(op)) {
      auto inputVal = andrOp.getOperand();
      z3::expr input = handleValue(inputVal, state);
      // Somtimes in is reduced to bool, so we can just fowards this value
      // here
      if (input.is_bool()) {
        return input;
      }
      unsigned w = input.get_sort().bv_size();
      ASSERT_STATE_DEBUG(w > 0 && "andr input width must be > 0");

      z3::expr all_ones = state.ctx().bv_val(-1, w);
      z3::expr is_all_ones = (input == all_ones);
      return z3::ite(is_all_ones, state.ctx().bv_val(1, 1),
                     state.ctx().bv_val(0, 1));
    }

    if (auto xorrOp = llvm::dyn_cast<circt::firrtl::XorRPrimOp>(op)) {
      auto inputVal = xorrOp.getOperand();
      z3::expr input = handleValue(inputVal, state);

      if (input.is_bool()) {
        return input;
      }

      unsigned w = input.get_sort().bv_size();
      z3::expr parity = state.ctx().bv_val(0, 1);

      for (unsigned i = 0; i < w; ++i) {
        z3::expr bit = input.extract(i, i);
        parity = parity ^ bit;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling XorRPrimOp, input: "
                 << input.to_string() << ", parity: " << parity.to_string()
                 << " for op " << *op << "\n");

      return parity;
    }

    if (auto shrOp = llvm::dyn_cast<circt::firrtl::ShrPrimOp>(op)) {
      auto ops = shrOp->getOperands();
      ASSERT_STATE_DEBUG(
          ops.size() == 1 &&
          "shr expects exactly 1 operand (amount is an attribute)");
      mlir::Value inputVal = ops[0];
      uint64_t amount = shrOp.getAmount();
      unsigned inW = getFIRRTLBitWidth(inputVal).value_or(1);
      z3::expr input = handleValue(inputVal, state);

      z3::expr shiftAmount = state.ctx().bv_val(amount, inW);
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling ShrPrimOp, input: "
                 << input.to_string()
                 << " (sort: " << input.get_sort().to_string() << ")"
                 << ", amount: " << amount << " for op " << *op << "\n");
      return z3::lshr(asBV(input, inW, state), shiftAmount);
    }

    if (auto shlOp = llvm::dyn_cast<circt::firrtl::ShlPrimOp>(op)) {
      auto ops = shlOp->getOperands();
      ASSERT_STATE_DEBUG(
          ops.size() == 1 &&
          "shl expects exactly 1 operand (amount is an attribute)");
      mlir::Value inputVal = ops[0];
      uint64_t amount = shlOp.getAmount();
      unsigned inW = getFIRRTLBitWidth(inputVal).value_or(1);
      z3::expr input = handleValue(inputVal, state);

      z3::expr shiftAmount = state.ctx().bv_val(amount, inW);
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling ShrPrimOp, input: "
                 << input.to_string()
                 << " (sort: " << input.get_sort().to_string() << ")"
                 << ", amount: " << amount << " for op " << *op << "\n");
      return z3::shl(asBV(input, inW, state), shiftAmount);
    }

    if (auto dhsrOp = llvm::dyn_cast<circt::firrtl::DShrPrimOp>(op)) {
      auto ops = dhsrOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "and expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];

      unsigned lhsW = getFIRRTLBitWidth(lhsVal).value_or(1);
      unsigned rhsW = getFIRRTLBitWidth(rhsVal).value_or(1);
      auto lhs = asBV(handleValue(lhsVal, state), lhsW, state);
      auto rhs = asBV(handleValue(rhsVal, state), rhsW, state);
      auto [lhsResolved, rhsResolved] = typeConflictResolution(lhs, rhs, state);

      return z3::lshr(lhsResolved, rhsResolved);
    }

    if (auto dhslOp = llvm::dyn_cast<circt::firrtl::DShlPrimOp>(op)) {
      auto ops = dhslOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "and expects exactly 2 operands");
      mlir::Value lhsVal = ops[0];
      mlir::Value rhsVal = ops[1];

      unsigned lhsW = getFIRRTLBitWidth(lhsVal).value_or(1);
      unsigned rhsW = getFIRRTLBitWidth(rhsVal).value_or(1);
      auto lhs = asBV(handleValue(lhsVal, state), lhsW, state);
      auto rhs = asBV(handleValue(rhsVal, state), rhsW, state);
      auto [lhsResolved, rhsResolved] = typeConflictResolution(lhs, rhs, state);
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling DSHR, "
                 << "lhs: " << lhsResolved.to_string()
                 << " (sort: " << lhsResolved.get_sort().to_string() << "), "
                 << "rhs: " << rhsResolved.to_string()
                 << " (sort: " << rhsResolved.get_sort().to_string() << ") "
                 << "for op " << *op << "\n");

      return z3::shl(lhsResolved, rhsResolved);
    }

    if (auto catOp = llvm::dyn_cast<circt::firrtl::CatPrimOp>(op)) {
      auto ops = catOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 2 && "cat expects exactly 2 operands");

      mlir::Value highVal = ops[0];
      mlir::Value lowVal = ops[1];

      z3::expr high = asBV(handleValue(highVal, state),
                           getFIRRTLBitWidth(highVal).value_or(1), state);
      z3::expr low = asBV(handleValue(lowVal, state),
                          getFIRRTLBitWidth(lowVal).value_or(1), state);

      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling CatPrimOp, high: " << high.to_string()
                 << ", low: " << low.to_string() << " for op " << *op << "\n");
      return z3::concat(high, low);
    }

    if (auto padOp = llvm::dyn_cast<circt::firrtl::PadPrimOp>(op)) {
      mlir::Value inVal = padOp.getInput();
      unsigned targetW = padOp.getAmount();
      unsigned inW = getFIRRTLBitWidth(inVal).value_or(1);
      z3::expr in = asBV(handleValue(inVal, state), inW, state);
      ASSERT_STATE_DEBUG(in.is_bv() && "pad input must lower to a bitvector");
      if (targetW <= inW) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[PETRINET] Handling PadPrimOp (noop), in: "
                   << in.to_string() << " (w=" << inW << ") targetW=" << targetW
                   << " for op " << *op << "\n");
        return in;
      }

      unsigned ext = targetW - inW;

      auto firTy = llvm::cast<circt::firrtl::FIRRTLType>(inVal.getType());
      bool isSigned = firTy.isa<circt::firrtl::SIntType>();

      z3::expr out = isSigned ? z3::sext(in, ext)  // sign-extend
                              : z3::zext(in, ext); // zero-extend
      return out;
    }

    if (auto tailOp = llvm::dyn_cast<circt::firrtl::TailPrimOp>(op)) {
      auto ops = tailOp->getOperands();
      ASSERT_STATE_DEBUG(ops.size() == 1 && "tail expects exactly 1 operand");
      mlir::Value inputVal = ops[0];
      z3::expr input = asBV(handleValue(inputVal, state),
                            getFIRRTLBitWidth(inputVal).value_or(1), state);
      unsigned amount = tailOp.getAmount();
      unsigned inW = input.get_sort().bv_size();
      assert(amount < inW && "tail amount must be less than input width");
      return input.extract(inW - 1, amount);
    }

    if (auto subIndexOp = llvm::dyn_cast<circt::firrtl::SubindexOp>(op)) {
      mlir::Value vecVal = subIndexOp.getInput();

      unsigned index = subIndexOp.getIndex();
      auto vecTy = llvm::dyn_cast<circt::firrtl::FVectorType>(vecVal.getType());
      assert(vecTy && "firrtl.subindex input must be a vector type");

      unsigned vecLen = vecTy.getNumElements();
      assert(index < vecLen && "subindex out of bounds");
      circt::firrtl::FIRRTLType elemTy = vecTy.getElementType();
      unsigned elemW = getFIRRTLBitWidth(elemTy).value_or(1);
      unsigned packedW = elemW * vecLen;
      z3::expr vec = asBV(handleValue(vecVal, state), packedW, state);
      assert(vec.is_bv() &&
             "subindex input must lower to a bitvector (packed vector)");

      unsigned lo = index * elemW;
      unsigned hi = lo + elemW - 1;

      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling SubindexOp, vec:" << vec.to_string()
                 << " index:" << index << " elemW:" << elemW << " slice:[" << hi
                 << ":" << lo << "]"
                 << " for op " << *op << "\n");

      return vec.extract(hi, lo);
    }

    if (auto subFieldOp = llvm::dyn_cast<circt::firrtl::SubfieldOp>(op)) {
      auto fieldAttr = subFieldOp.getFieldName();
      std::string fieldName = fieldAttr.str();
      state.localState.fields.push_back(fieldName);

      auto inputVal = subFieldOp.getInput();

      // if (auto ba = inputVal.dyn_cast<mlir::BlockArgument>()) {
      //   z3::expr e = handleBlockArgument(ba, state);
      //   state.localState.fields.pop_back();
      //   return e;
      // }

      z3::expr res = handleValue(inputVal, state);
      state.localState.fields.pop_back();
      return res;
    }

    if (auto bitsPrimOp = llvm::dyn_cast<circt::firrtl::BitsPrimOp>(op)) {
      mlir::Value inVal = bitsPrimOp.getInput();
      z3::expr in = asBV(handleValue(inVal, state),
                         getFIRRTLBitWidth(inVal).value_or(1), state);

      auto lo = bitsPrimOp.getLo();
      auto hi = bitsPrimOp.getHi();

      auto e = in.extract(hi, lo);
      return e;
    }

    if (auto cvtPrimOp = llvm::dyn_cast<circt::firrtl::CvtPrimOp>(op)) {
      mlir::Value inVal = cvtPrimOp.getInput();
      unsigned inWidth = getFIRRTLBitWidth(inVal).value_or(1);
      z3::expr in = asBV(handleValue(inVal, state), inWidth, state);

      if (mlir::isa<circt::firrtl::UIntType>(inVal.getType()))
        return z3::zext(in, 1);

      return in;
    }

    // TODO: FIX THIS
    if (auto InvalidValueOp =
            llvm::dyn_cast<circt::firrtl::InvalidValueOp>(op)) {
      auto ty = InvalidValueOp.getType();
      auto firTy = llvm::dyn_cast<circt::firrtl::FIRRTLType>(ty);
      assert(firTy && "InvalidValueOp type  is not a FIRRTLType");
      unsigned width = getFIRRTLBitWidth(firTy).value_or(1);
      if (width == 0)
        width = 1; // TOOD: Sometimes it creates a signal with 0 width
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] Handling InvalidValueOp, type: " << firTy
                 << ", width: " << width << "for op " << *op << "\n");
      return state.ctx().bv_val(0, width);
    }

    if (llvm::isa<circt::firrtl::BitCastOp>(op) ||
        llvm::isa<circt::firrtl::AsAsyncResetPrimOp>(op) ||
        llvm::isa<circt::firrtl::AsClockPrimOp>(op) ||
        llvm::isa<circt::firrtl::ConstCastOp>(op)) {
      mlir::Value value = op->getOperand(0);
      LLVM_DEBUG(llvm::dbgs()
                 << "[PETRINET] IGNORING cast op: " << *op << "\n");
      return handleValue(value, state);
    }

    if (auto asSIntOp = llvm::dyn_cast<circt::firrtl::AsSIntPrimOp>(op)) {
      mlir::Value value = asSIntOp.getOperand();
      z3::expr e = handleValue(value, state);
      unsigned width = getFIRRTLBitWidth(value).value_or(1);
      LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Handling AsSIntPrimOp, operand:"
                              << e.to_string() << ", width: " << width
                              << " for op " << *op << "\n");
      return asInt(e, state, /*signed=*/true);
    }

    if (auto asUIntOp = llvm::dyn_cast<circt::firrtl::AsUIntPrimOp>(op)) {
      mlir::Value value = asUIntOp.getOperand();
      z3::expr e = handleValue(value, state);
      unsigned width = getFIRRTLBitWidth(value).value_or(1);
      LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Handling AsUIntPrimOp, operand:"
                              << e.to_string() << ", width: " << width
                              << " for op " << *op << "\n");
      return asInt(e, state, /*signed=*/false);
    }

    op->emitError("PETRINET: unknown/unhandled FIRRTL expression op");

    for (auto operand : op->getOperands()) {
      handleValue(operand, state);
    }
    assert(false && "unhandled FIRRTL expression op");
    // return state.ctx().bool_val(true);
  }

  mlir::Operation *findDrivingOp(mlir::Value v,
                                 FIRRTLPrintPetriNetPassState &state) {
    for (Operation *user : v.getUsers()) {
      if (auto connect = llvm::dyn_cast<firrtl::FConnectLike>(user)) {
        if (connect.getDest() == v) {
          LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Found connect " << *connect
                                  << " driving value " << v << "\n");
          return connect;
        }
      }
      // Uncessary after lowering types
      // if (auto subIndexOp =
      // llvm::dyn_cast<circt::firrtl::SubindexOp>(user))
      // {
      //   if (subIndexOp.getInput() == v) {
      //     LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Found subindex " <<
      //     *subIndexOp
      //                             << " driving value " << v << "\n");
      //     FIRRTLPrintPetriNetPassState::LocalState backup =
      //     state.localState; auto res =
      //     findDrivingOp(subIndexOp.getResult(), state); state.localState =
      //     backup; return res;
      //   }
      // }
      // if (auto subFieldOp =
      // llvm::dyn_cast<circt::firrtl::SubfieldOp>(user))
      // {
      //   if (subFieldOp.getInput() == v) {
      //     LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Found subfield " <<
      //     *subFieldOp
      //                             << " driving value " << v << "\n");
      //     FIRRTLPrintPetriNetPassState::LocalState backup =
      //     state.localState; auto res =
      //     findDrivingOp(subFieldOp.getResult(), state); state.localState =
      //     backup; return res;
      //   }
      // }
    }

    state.dump(llvm::errs());
    llvm::errs() << "Could not find driver for value " << v << "\n";
    assert(false && "could not find driver for value");
  }

  void kickOffRunAtActivePoint(ActivationPoint ap,
                               FIRRTLPrintPetriNetPassState &state) {
    auto op = ap.op;
    assert(op->getNumResults() == 1 && "op must have one result");
    assert(isDefinition(op) &&
           "kickOffRunAtActivePoint called on non-register");

    auto target = getTargetAnno(op, state);
    state.stageState.visited.push_back(target);
    auto name = annoToString(target);

    state.globalState.targetCache[name] = target;
    state.localState = ap.localState;

    mlir::Operation *starting_op = nullptr;
    mlir::Value starting_value = nullptr;

    if (llvm::isa<firrtl::RegOp>(op) || llvm::isa<firrtl::RegResetOp>(op)) {
      auto regResult = op->getResult(0);

      for (Operation *user : regResult.getUsers()) {
        if (auto connect = llvm::dyn_cast<firrtl::FConnectLike>(user)) {
          if (connect.getDest() == regResult) {
            state.mapConnectToReg(connect, op);
            starting_op = connect.getOperation();
            starting_value = connect.getDest();

            LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Found connect " << *connect
                                    << " driving register " << name << "\n");
            break;
          }
        }
      }

      if (!starting_op) {
        op->emitError()
            << "could not find driving connect for register-like op";
        return;
      }
    } else if (auto node = llvm::dyn_cast<firrtl::NodeOp>(op)) {
      starting_value = node.getInput();
      starting_op = starting_value.getDefiningOp();

      if (!starting_op) {
        op->emitError()
            << "node input has no defining op; cannot kick off stage";
        return;
      }
    } else {
      llvm_unreachable("unknown active op type");
    }

    LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Starting stage at node: "
                            << *starting_op << "\n");

    // llvm::errs() << "[PETRINET] Starting stage at node: " << *starting_op
    //              << " with driving value of ";
    // starting_value.print(llvm::errs());
    // llvm::errs() << "\n";

    z3::expr rhs = handleOp(starting_op, starting_value, state);
    z3::expr lhs = rhs.is_bv() ? state.ctx().bv_const(name.c_str(),
                                                      rhs.get_sort().bv_size())
                               : state.ctx().bool_const(name.c_str());

    // Simplified does nothing in our case for now
    // z3::expr final = z3::implies(rhs , lhs ==
    // state.ctx().bool_val(true)); z3::expr simplified_lhs =
    // lhs.simplify(); z3::expr final = z3::implies(rhs , simplified_lhs);
    // Unsimplfiied
    // z3::expr final = z3::implies(rhs , lhs);

    z3::expr final = z3::expr(rhs);

    auto &gs = state.globalState;
    int stg = state.getStageNumber();

    gs.perStageExprs[stg].push_back(final);
    gs.perStageNames[stg].push_back(name);

    state.globalState.finalExprs.push_back(final);
    state.nodeNames().push_back(name);
    LLVM_DEBUG(llvm::dbgs() << "[PETRINET] RHS expr for register " << name
                            << ": " << (rhs).to_string() << "\n");

    LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Finished processing stage: "
                            << final.to_string() << "\n");
  }

  FIRRTLPrintPetriNetPassState runStage(FIRRTLPrintPetriNetPassState state) {
    state.incrementStageNumber();

    // LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Performing stage: "
    //                         << state.getStageNumber() << " ");
    // llvm::dbgs()
    //     << "\n"
    //     << "============================================================\n"
    //     << "====================  PETRINET STAGE
    //     ========================\n"
    //     << "============================================================\n"
    //     << "  Stage Number : " << state.getStageNumber() << "\n"
    //     << "============================================================\n"
    //     << "\n";

    for (int i = 0; true; ++i) {
      bool allRegisters = true;

      for (auto point : state.activePoints()) {

        // After the first round, keep running until we hit actual registers
        // at all points.
        if (i != 0 && isRegisterLike(point.op)) {
          // Forward the unhandled active points to the next round
          state.stageState.nextActivePoints.insert(point);
          continue;
        }

        allRegisters = false;

        auto op = point.op;
        state.localState = point.localState;

        assert(op && "active op is null");
        assert(isDefinition(op) && "active op is not a register");
        LLVM_DEBUG(llvm::dbgs() << "at " << *op << "\n");
        // Recursively walk back through. Do we build expression along the
        // way?
        kickOffRunAtActivePoint(point, state);
      }

      state.stageState.activePoints = state.stageState.nextActivePoints;
      state.stageState.nextActivePoints = {};

      if (allRegisters) {
        break;
      }
      for (auto point : state.activePoints()) {
        LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Next active op for next stage:"
                                << *point.op << "\n");
      }
    }

    return state;
  }

  void writeStagePetriNet(FIRRTLPrintPetriNetPassState &state) {
    std::error_code ec;
    std::string filename = debugDirectory + "/petri_stage_" +
                           std::to_string(state.getStageNumber()) + ".dot";
    llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "failed to open '" << filename << "': " << ec.message();
      signalPassFailure();
      return;
    }
    state.getPetriNet()->writeGraph(os);
  }

  void writeStageZ3(FIRRTLPrintPetriNetPassState &state) {
    std::error_code ec;
    std::string filename = debugDirectory + "/stage_" +
                           std::to_string(state.getStageNumber()) + ".z3";
    llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "failed to open '" << filename << "': " << ec.message();
      signalPassFailure();
      return;
    }

    assert(state.nodeNames().size() == state.globalState.finalExprs.size() &&
           "nodeNames and finalExprs must be the same size");
    for (unsigned long i = 0; i < state.nodeNames().size(); i++) {
      auto name = state.nodeNames()[i];
      auto expr = state.globalState.finalExprs[i];

      os << name << ":\n";
      os << expr.to_string() << "\n\n";
    }
  }

  // Write to DOT for graphviz

  static std::string dotEscape(llvm::StringRef s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
      switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
      }
    }
    return out;
  }

  static std::string sanitizeDotFilename(llvm::StringRef name) {
    std::string out;
    out.reserve(name.size());
    for (char c : name) {
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.') {
        out.push_back(c);
      } else {
        out.push_back('_');
      }
    }
    if (out.empty())
      out = "unnamed";
    return out;
  }

  static unsigned astId(const z3::expr &e) { return Z3_get_ast_id(e.ctx(), e); }

  static bool isAtom(const z3::expr &e) {
    return e.is_app() && e.num_args() == 0;
  }
  static bool isNumeralAtom(const z3::expr &e) {
    return isAtom(e) && e.is_numeral();
  }
  static bool isSymbolAtom(const z3::expr &e) {
    return isAtom(e) && !e.is_numeral();
  }

  static inline Z3_decl_kind getDeclKind(const z3::expr &e) {
    return Z3_get_decl_kind(e.ctx(), e.decl());
  }

  static inline unsigned getDeclNumParams(const z3::expr &e) {
    return Z3_get_decl_num_parameters(e.ctx(), e.decl());
  }

  static inline int getDeclIntParam(const z3::expr &e, unsigned idx) {
    // Z3 stores params on the decl; for extract these are ints (hi, lo)
    return Z3_get_decl_int_parameter(e.ctx(), e.decl(), idx);
  }

  static std::string internalLabel(const z3::expr &e) {
    Z3_decl_kind k = getDeclKind(e);

    // --- Special-case parameterized ops so constants appear in the node
    // label.
    switch (k) {
    case Z3_OP_EXTRACT: {
      // Params: [hi, lo]
      // Z3 generally uses 2 int params for extract.
      if (getDeclNumParams(e) >= 2) {
        int hi = getDeclIntParam(e, 0);
        int lo = getDeclIntParam(e, 1);
        return "extract[" + std::to_string(hi) + ":" + std::to_string(lo) + "]";
      }
      return "extract";
    }
    case Z3_OP_ZERO_EXT: {
      // Param: extension amount
      if (getDeclNumParams(e) >= 1) {
        int n = getDeclIntParam(e, 0);
        return "zero_ext to" + std::to_string(n);
      }
      return "zero_ext";
    }
    case Z3_OP_SIGN_EXT: {
      if (getDeclNumParams(e) >= 1) {
        int n = getDeclIntParam(e, 0);
        return "sign_ext[" + std::to_string(n) + "]";
      }
      return "sign_ext";
    }
    case Z3_OP_REPEAT: {
      if (getDeclNumParams(e) >= 1) {
        int n = getDeclIntParam(e, 0);
        return "repeat[" + std::to_string(n) + "]";
      }
      return "repeat";
    }
    case Z3_OP_ROTATE_LEFT: {
      if (getDeclNumParams(e) >= 1) {
        int n = getDeclIntParam(e, 0);
        return "rotl[" + std::to_string(n) + "]";
      }
      return "rotl";
    }
    case Z3_OP_ROTATE_RIGHT: {
      if (getDeclNumParams(e) >= 1) {
        int n = getDeclIntParam(e, 0);
        return "rotr[" + std::to_string(n) + "]";
      }
      return "rotr";
    }
    default:
      break;
    }

    return e.decl().name().str();
  }

  static bool isIte(const z3::expr &e) {
    return e.is_app() && e.decl().decl_kind() == Z3_OP_ITE && e.num_args() == 3;
  }

  static std::string argRoleLabel(const z3::expr &parent, unsigned argIdx) {
    if (isIte(parent)) {
      if (argIdx == 0)
        return "cond";
      if (argIdx == 1)
        return "then";
      if (argIdx == 2)
        return "else";
    }
    return std::to_string(argIdx);
  }

  // Canonicalize Z3-printed symbol names to match your nodeNames.
  // Example: "|~RocketCore\\|RocketCore>_T_4158|" ->
  // "~RocketCore|RocketCore>_T_4158"
  static std::string canonicalizeZ3Symbol(llvm::StringRef s) {
    std::string out = s.str();

    if (out.size() >= 2 && out.front() == '|' && out.back() == '|')
      out = out.substr(1, out.size() - 2);

    std::string unesc;
    unesc.reserve(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
      char c = out[i];
      if (c == '\\' && i + 1 < out.size()) {
        char n = out[i + 1];
        // common escapes you showed: \| and \\ (and sometimes \>)
        if (n == '|' || n == '\\' || n == '>') {
          unesc.push_back(n);
          ++i;
          continue;
        }
        // default: drop '\' and keep next
        unesc.push_back(n);
        ++i;
        continue;
      }
      unesc.push_back(c);
    }
    return unesc;
  }
  // ---------- main ----------
  void writeStageZ3DotFromName(FIRRTLPrintPetriNetPassState &state,
                               llvm::StringRef startName) {
    auto &gs = state.globalState;

    assert(gs.nodeNames.size() == gs.finalExprs.size() &&
           "nodeNames and finalExprs must match");

    auto canonName = [&](llvm::StringRef s) -> std::string {
      // Must match how Z3 leaf names are canonicalized in
      // getExprNode()/walkExpr().
      return canonicalizeZ3Symbol(s.str());
    };

    // Build def map from ALL currently-available finalExprs, but canonicalize
    // keys
    std::unordered_map<std::string, const z3::expr *> defMap;
    defMap.reserve(gs.nodeNames.size() * 2 + 8);

    for (size_t i = 0; i < gs.nodeNames.size(); ++i) {
      std::string key = canonName(gs.nodeNames[i]);

      // Prefer first definition if duplicates exist
      auto [it, inserted] = defMap.emplace(key, &gs.finalExprs[i]);
      (void)it;
      if (!inserted) {
        // Optional debug:
        // llvm::errs() << "Duplicate canonical name in defMap: " << key <<
        // "\n";
      }
    }

    // Canonicalize start name too
    std::string startCanon = canonName(startName);

    // Find the starting definition
    auto itStart = defMap.find(startCanon);
    if (itStart == defMap.end() || !itStart->second) {
      llvm::errs() << "writeStageZ3DotFromName: could not find definition for '"
                   << startName << "' (canonical: '" << startCanon << "')\n";
      return;
    }

    // Output file
    std::string filename = debugDirectory + "/stage_" +
                           std::to_string(state.getStageNumber()) + "__" +
                           sanitizeDotFilename(startCanon) + ".z3.dot";

    std::error_code ec;
    llvm::raw_fd_ostream os(filename, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "Failed to open " << filename << ": " << ec.message()
                   << "\n";
      return;
    }

    // Register membership: canonicalize to match symbol labels in the graph
    llvm::DenseSet<llvm::StringRef> regSyms;
    llvm::SmallVector<std::string, 256> regStorage;
    regStorage.reserve(gs.registers.size());
    for (const std::string &r : gs.registers) {
      regStorage.push_back(canonName(r));
      regSyms.insert(regStorage.back());
    }

    auto isRegisterSym = [&](llvm::StringRef canonSym) -> bool {
      return regSyms.contains(canonSym);
    };
    auto isRegisterSymStr = [&](const std::string &canonSym) -> bool {
      return regSyms.contains(canonSym);
    };

    // DOT header
    os << "digraph Stage" << state.getStageNumber() << " {\n";
    os << "  rankdir=LR;\n";
    os << "  node [fontname=\"Helvetica\"];\n";
    os << "  edge [fontname=\"Helvetica\"];\n\n";

    unsigned nextId = 0;

    // ----- node tables -----
    llvm::StringMap<unsigned> symToId;

    auto getOrCreateSymbolNode = [&](llvm::StringRef label) -> unsigned {
      auto it = symToId.find(label);
      if (it != symToId.end())
        return it->second;

      unsigned id = nextId++;
      symToId.insert({label, id}); // StringMap copies/owns the key

      bool isReg = isRegisterSym(label);
      os << "  n" << id << " [label=\"" << dotEscape(label) << "\""
         << ", shape=oval, style=filled"
         << ", fillcolor=\"" << (isReg ? "lightcoral" : "lightgray")
         << "\"];\n";
      return id;
    };

    auto createConstNode = [&](llvm::StringRef label) -> unsigned {
      unsigned id = nextId++;
      os << "  n" << id << " [label=\"" << dotEscape(label) << "\""
         << ", shape=triangle, style=filled, fillcolor=\"white\"];\n";
      return id;
    };

    llvm::DenseMap<unsigned, unsigned> astToDot;
    auto getOrCreateInternalNode = [&](const z3::expr &e) -> unsigned {
      unsigned aid = astId(e);
      if (auto it = astToDot.find(aid); it != astToDot.end())
        return it->second;

      unsigned id = nextId++;
      astToDot[aid] = id;

      std::string label = internalLabel(e);
      os << "  n" << id << " [label=\"" << dotEscape(label) << "\""
         << ", shape=box, style=filled, fillcolor=\"white\"];\n";
      return id;
    };

    auto getExprNode = [&](const z3::expr &e) -> unsigned {
      if (isNumeralAtom(e))
        return createConstNode(e.to_string());
      if (isSymbolAtom(e)) {
        std::string canon = canonicalizeZ3Symbol(e.decl().name().str());
        return getOrCreateSymbolNode(canon);
      }
      return getOrCreateInternalNode(e);
    };

    // ----- edge emission (dedup by from/to/label) -----
    llvm::DenseSet<uint64_t> emittedEdges;
    auto edgeKey = [&](unsigned from, unsigned to,
                       llvm::StringRef lab) -> uint64_t {
      uint64_t h = (uint64_t(from) << 32) ^ uint64_t(to);
      uint64_t lh = (uint64_t)llvm::hash_value(lab);
      return h ^ (lh + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    };

    auto emitEdge = [&](unsigned from, unsigned to, llvm::StringRef label) {
      uint64_t key = edgeKey(from, to, label);
      if (!emittedEdges.insert(key).second)
        return;
      os << "  n" << from << " -> n" << to << " [label=\"" << dotEscape(label)
         << "\"];\n";
    };

    // ----- expansion logic -----
    std::unordered_set<std::string> expandedNames;
    expandedNames.reserve(512);

    llvm::DenseSet<unsigned> walkedInternalAst;

    std::function<void(const z3::expr &)> walkExpr;
    std::function<void(const std::string &, bool)> expandName;

    walkExpr = [&](const z3::expr &e) {
      if (isSymbolAtom(e)) {
        std::string canon = canonicalizeZ3Symbol(e.decl().name().str());

        // Registers are leaves during recursive traversal
        if (isRegisterSymStr(canon))
          return;

        if (defMap.find(canon) != defMap.end())
          expandName(canon, /*allowRegisterExpansion=*/false);
        return;
      }

      // Numerals / other atoms: nothing to do structurally
      if (isAtom(e))
        return;

      unsigned parent = getExprNode(e);

      for (unsigned i = 0; i < (unsigned)e.num_args(); ++i) {
        z3::expr childExpr = e.arg(i);
        unsigned child = getExprNode(childExpr);
        emitEdge(child, parent, argRoleLabel(e, i));

        if (isSymbolAtom(childExpr)) {
          std::string canon =
              canonicalizeZ3Symbol(childExpr.decl().name().str());

          // Registers are leaves during recursive traversal
          if (isRegisterSymStr(canon))
            continue;

          if (defMap.find(canon) != defMap.end())
            expandName(canon, /*allowRegisterExpansion=*/false);
        }
      }

      unsigned aid = astId(e);
      if (!walkedInternalAst.insert(aid).second)
        return;

      for (unsigned i = 0; i < (unsigned)e.num_args(); ++i)
        walkExpr(e.arg(i));
    };

    expandName = [&](const std::string &name, bool allowRegisterExpansion) {
      // Registers are leaves, except for the initial root expansion
      if (isRegisterSymStr(name) && !allowRegisterExpansion) {
        (void)getOrCreateSymbolNode(name);
        return;
      }

      if (!expandedNames.insert(name).second)
        return;

      auto it = defMap.find(name);
      if (it == defMap.end() || !it->second) {
        // Still ensure symbol exists visually
        (void)getOrCreateSymbolNode(name);
        return;
      }

      const z3::expr &defExpr = *it->second;

      // If this definition is just an alias to another symbol, expand that
      // too, but only recursive expansions should stop at registers.
      if (isSymbolAtom(defExpr)) {
        std::string rhs = canonicalizeZ3Symbol(defExpr.decl().name().str());
        expandName(rhs, /*allowRegisterExpansion=*/false);
      }

      unsigned symNode = getOrCreateSymbolNode(name);
      unsigned rootNode = getExprNode(defExpr);

      emitEdge(rootNode, symNode, "def");
      walkExpr(defExpr);
    };

    // Kick off at requested name (canonical): allow first-stage expansion
    // even if register
    expandName(startCanon, /*allowRegisterExpansion=*/true);

    os << "}\n";
  }
}; // namespace
} // namespace

void FIRRTLPrintPetriNetPass::init() {

  if (petriFile.empty()) {
    llvm::errs() << "missing --petri-file";
    // signalPassFailure();
    return;
  }

  std::error_code ec;
  llvm::raw_fd_ostream os(petriFile, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "failed to open '" << petriFile << "': " << ec.message();
    signalPassFailure();
    return;
  }

  os << "# Petri net...\n";
}
mlir::Operation *FIRRTLPrintPetriNetPass::findProgressSignal(
    circt::firrtl::CircuitOp circuitOp, std::string progressSignal,
    std::string moduleName, std::string topModule,
    std::vector<FIRRTLPrintPetriNetPassState::LocalState::InterproceduralState::
                    CallInfo> &startingCallerContext) {

  using CallInfo =
      FIRRTLPrintPetriNetPassState::LocalState::InterproceduralState::CallInfo;

  mlir::Operation *progressOp = nullptr;
  startingCallerContext.clear();

  llvm::DenseMap<llvm::StringRef, firrtl::FModuleOp> moduleMap;
  circuitOp.walk([&](firrtl::FModuleOp moduleOp) {
    moduleMap[moduleOp.getName()] = moduleOp;
  });

  auto topIt = moduleMap.find(topModule);
  if (topIt == moduleMap.end()) {
    emitError(circuitOp.getLoc()) << "could not find top module: " << topModule;
    signalPassFailure();
    return nullptr;
  }

  auto targetIt = moduleMap.find(moduleName);
  if (targetIt == moduleMap.end()) {
    emitError(circuitOp.getLoc())
        << "could not find target module: " << moduleName;
    signalPassFailure();
    return nullptr;
  }

  std::function<bool(firrtl::FModuleOp, std::vector<CallInfo> &)> dfs =
      [&](firrtl::FModuleOp currentModule,
          std::vector<CallInfo> &callerContext) -> bool {
    // Only search for the signal inside the requested target module.
    if (currentModule.getName() == moduleName) {
      bool foundInThisModule = false;
      currentModule.walk([&](mlir::Operation *op) {
        if (foundInThisModule)
          return;

        if (auto nameableOp = llvm::dyn_cast<firrtl::FNamableOp>(op)) {
          auto nameAttr = nameableOp.getNameAttr();
          if (nameAttr && nameAttr.getValue() == progressSignal) {
            progressOp = op;
            startingCallerContext = callerContext;
            foundInThisModule = true;
          }
        }
      });

      if (foundInThisModule)
        return true;
    }

    // Recurse through instances reachable from the current module.
    for (auto &op : *currentModule.getBodyBlock()) {
      auto inst = llvm::dyn_cast<firrtl::InstanceOp>(&op);
      if (!inst)
        continue;

      auto calleeIt = moduleMap.find(inst.getModuleName());
      if (calleeIt == moduleMap.end())
        continue;

      firrtl::FModuleOp calleeModule = calleeIt->second;

      CallInfo ci{
          currentModule.getOperation(), // caller
          inst.getOperation(),          // callsite
          calleeModule.getOperation(),  // callee
          0                             // injectPortIdx placeholder
      };

      callerContext.push_back(ci);
      if (dfs(calleeModule, callerContext))
        return true;
      callerContext.pop_back();
    }

    return false;
  };

  std::vector<CallInfo> callerContext;
  dfs(topIt->second, callerContext);

  if (!progressOp) {
    emitError(circuitOp.getLoc())
        << "could not find progress signal: " << progressSignal
        << " inside module: " << moduleName
        << " reachable from top module: " << topModule;
    signalPassFailure();
  }

  return progressOp;
}

void FIRRTLPrintPetriNetPass::runOnOperation() {
  assert(progressSignal != "" &&
         "progress signal must be specified for FIRRTLPrintPetriNetPass");
  LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Progress signal: " << progressSignal
                          << "\n");
  init();
  using CallInfo =
      FIRRTLPrintPetriNetPassState::LocalState::InterproceduralState::CallInfo;
  std::vector<CallInfo> startingCallerContext;
  mlir::Operation *op =
      findProgressSignal(getOperation(), progressSignal, moduleName, topModule,
                         startingCallerContext);

  for (auto &ci : startingCallerContext) {
    llvm::errs() << "  caller module: "
                 << llvm::cast<firrtl::FModuleOp>(ci.caller).getName() << ", "
                 << "callee module: "
                 << llvm::cast<firrtl::FModuleOp>(ci.callee).getName() << "\n";
  }
  auto circuit = llvm::cast<circt::firrtl::CircuitOp>(getOperation());
  mlir::StringAttr topNameAttr = circuit.getNameAttr();
  auto topModule =
      mlir::SymbolTable::lookupNearestSymbolFrom<circt::firrtl::FModuleOp>(
          circuit, topNameAttr);

  assert(topModule && "Top module not found");
  // Add this so we can exit the progress signal at the start:

  FIRRTLPrintPetriNetPassState state =
      initializeState(circuit, topModule, op, startingCallerContext);
  // auto progressAnnoPathValue = createAnnoPathForCallerContext(state, op);
  // auto progressAnnoName = annoToString(progressAnnoPathValue);
  // llvm::dbgs() << "[PETRINET] Progress annotation name: " << progressAnnoName
  //              << "\n";
  // LLVM_DEBUG(llvm::dbgs() << "[PETRINET] Progress signal found " << *op
  //                         << "\n");
  // llvm::errs() << "With starting callercontext: "
  //              << "\n";

  auto target = getTargetAnno(op, state);
  auto rootGraphName = annoToString(target);
  llvm::dbgs() << "[PETRINET] Root graph name: " << rootGraphName << "\n";
  state.globalState.targetCache[rootGraphName] = target;

  if (llvm::isa<firrtl::RegResetOp>(op) || llvm::isa<firrtl::RegOp>(op)) {
    llvm::errs() << "We see register op, so we insert " << annoToString(target)
                 << " into registers set directly.\n";
    state.globalState.registers.insert(annoToString(target));
  }

  while (true) {
    state = runStage(std::move(state));
    if (debugDirectory != "") {
      // writeStagePetriNet(state);
      writeStageZ3(state);
      for (const auto &name :
           state.globalState.perStageNames[state.getStageNumber()]) {

        if (state.globalState.registers.count(name) == 0) {
        }
        continue;

        // writeStageZ3DotFromName(state, name);
      }
    }
    // Only run max level of stages if this is set
    if (levels >= 0 && state.getStageNumber() >= levels)
      break;
    if (state.isDone())
      break;
    if (state.activePoints().empty()) {
      LLVM_DEBUG(llvm::dbgs() << "[PETRINET] No more active ops, finishing.\n");
      break;
    }
  }

  // Cleanup to add the final registers at the analysis boundary?
  for (const auto &ap : state.activePoints()) {
    z3::expr edge_node_expr = handleRegisterLike(ap.op, state);
    auto edge_target = getTargetAnno(ap.op, state);
    auto edge_node_name = annoToString(edge_target);
    state.globalState.targetCache[edge_node_name] = edge_target;
    state.finalExprs().push_back(edge_node_expr);
    state.nodeNames().push_back(edge_node_name);
    // llvm::errs() << "Inserting final edge node for active point: " << *ap.op
    //              << " with name: " << edge_node_name << "\n";
    state.globalState.registers.insert(edge_node_name);
  }

  llvm::errs() << "Building final Z3 graph with root name " << progressSignal
               << " and " << state.globalState.finalExprs.size()
               << " total expressions\n";
  z3::context &ctx = state.ctx();

  auto z3graph = std::make_unique<Z3Graph>(
      ctx, state.globalState.nodeNames, rootGraphName,
      state.globalState.finalExprs, state.globalState.registers);
  z3graph->buildZ3Graph();

  // auto rootGraphName = z3graph.rootName;
  // z3graph->buildDominatorTree();

  // if (!z3graph->domTree) {
  //   emitError(getOperation()->getLoc()) << "failed to build dominator tree";
  //   signalPassFailure();
  //   return;
  // }

  std::unordered_map<std::string, RegisterArtifacts> artifactsMap;

  auto emitErrorFn = [&](const Twine &msg) {
    emitError(getOperation()->getLoc()) << msg;
  };
  auto signalFailure = [&]() { signalPassFailure(); };

  if (!computeRegisterArtifacts(*z3graph, ctx, emitErrorFn, signalFailure,
                                artifactsMap))
    return;

  if (!withOutputFile(
          debugDirectory + "/full_graph.dot",
          [&](llvm::raw_ostream &os) { z3graph->printFullGraphDot(os); },
          getOperation(), this))
    return;

  // if (!withOutputFile(
  //         debugDirectory + "/dominator_tree.dot",
  //         [&](llvm::raw_ostream &os) {
  //           z3graph->domTree->printFullGraphDot(os);
  //         },
  //         getOperation(), this))
  //   return;

  if (!emitRegisterArtifacts(debugDirectory, artifactsMap, getOperation(),
                             this))
    return;

  auto &ig = getAnalysis<circt::firrtl::InstanceGraph>();

  auto counterInserter = CounterInserter(circuit, ig);

  if (counterMode == "place") {
    insertPlaceCounters(state, artifactsMap, counterInserter);
  } else if (counterMode == "transition") {
    insertIncomingTransitionEdgeCounters(state, artifactsMap, counterInserter);
  } else {
    llvm::errs() << "Unknown counter mode: " << this->counterMode << "\n";
    signalPassFailure();
    return;
  }

  llvm::errs() << "Finished pass\n";
  llvm::errs() << "Number of expressions: "
               << state.globalState.finalExprs.size() << "\n";
  llvm::errs() << "Number of registers: " << state.globalState.registers.size()
               << "\n";

  // Now build full petri net graph: start with rootnode, and then recursively
  // expand all nodes in the z3 graph that are registers but not quasi
  // registers.
  auto rootArtifact = artifactsMap.find(rootGraphName);
  if (rootArtifact == artifactsMap.end()) {
    emitError(getOperation()->getLoc())
        << "could not find root graph for progress signal: " << rootGraphName;
    for (auto &[name, artifact] : artifactsMap) {
      llvm::errs() << "  available graph: " << name << "\n";
    }
    signalPassFailure();
    return;
  }

  // auto normalized_expr = rootArtifact->second.finalExpr.simplify();

  auto normalized_expr = rootArtifact->second.finalExpr;
  auto can = Z3Graph::Canonicalizer(ctx);
  normalized_expr = can.canonicalize(normalized_expr);
  normalized_expr = normalized_expr.simplify();
  normalized_expr = Z3Graph::Canonicalizer::toNNF(normalized_expr);
  normalized_expr = normalized_expr.simplify();

  z3::set_param("pp.max_depth", 1000000);
  z3::set_param("pp.max_num_lines", 1000000);
  z3::set_param("pp.min_alias_size", 1000000000);

  if (!withOutputFile(
          debugDirectory + "/full_graph.expr",
          [&](llvm::raw_ostream &os) { os << normalized_expr.to_string(); },
          getOperation(), this))
    return;

  // llvm::errs() << "Normalized root expressio from:  "
  //              << rootArtifact->second.finalExpr.to_string()
  //              << "to: " << normalized_expr.to_string() << "\n";

  auto full_petri = Petrinet::createFromZ3(normalized_expr, progressSignal);
  if (!full_petri) {
    llvm::errs() << "failed to build initial petrinet for progress signal: "
                 << progressSignal;
    return;
  }

  std::unordered_set<std::string> visited;
  visited.insert(rootGraphName);

  std::function<bool(const std::string &)> expandRegister =
      [&](const std::string &regName) -> bool {
    if (!visited.insert(regName).second)
      return true;

    auto it = artifactsMap.find(regName);
    if (it == artifactsMap.end()) {
      llvm::errs() << "Warning: missing artifact for register dependency: "
                   << regName << "\n";
      // emitWarning(getOperation()->getLoc())
      //     << "missing artifact for register dependency: " << regName;
      return true; // continue recursion without failing
    }
    auto &artifact = it->second;

    if (!artifact.quasiSubgraph) {
      emitError(getOperation()->getLoc())
          << "missing quasi subgraph for register dependency: " << regName;
      signalPassFailure();
      return false;
    }

    llvm::errs() << "Creating petri net for " << regName << "\n";
    full_petri->extendGraphWithExpression(artifact.finalExpr, regName);

    for (const auto &node : artifact.quasiSubgraph->nodes) {
      if (!node)
        continue;

      if (node->isRegister && !node->isQuasiRegister) {
        if (!expandRegister(node->name))
          return false;
      }
    }

    return true;
  };

  for (const auto &node : rootArtifact->second.quasiSubgraph->nodes) {
    if (!node)
      continue;

    if (node->isRegister && !node->isQuasiRegister) {
      if (!expandRegister(node->name))
        return;
    }
  }

  std::string petriFile = debugDirectory + "/full.petri.dot";
  if (!withOutputFile(
          petriFile, [&](llvm::raw_ostream &os) { full_petri->writeGraph(os); },
          op, this)) {
    llvm::errs() << "Could not open file for writing: " << petriFile << "\n";
    return;
  }
};

std::unique_ptr<mlir::Pass> circt::firrtl::createFIRRTLPrintPetriNetPass(
    std::string moduleName, std::string progressSignal, std::string topModule,
    std::string petriFile, std::string debugDirectory, int levels,
    std::string counterMode) {
  return std::make_unique<FIRRTLPrintPetriNetPass>(
      moduleName, progressSignal, topModule, petriFile, debugDirectory, levels,
      counterMode);
}
