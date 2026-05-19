//===- PowerTrace.cpp - Check and remove const types -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PowerTrace pass.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <fstream>
#include <map>
#include <stdint.h>

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/FIRRTL/Power.h"
#include "mlir/IR/Threading.h"

#include "circt/Dialect/Perf/PerfDialect.h"
#include "circt/Dialect/Perf/PerfHelpers.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "circt/Dialect/Perf/PerfPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {
class PowerTracePass : public PowerTraceBase<PowerTracePass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<circt::perf::PerfDialect>();
    registry.insert<circt::firrtl::FIRRTLDialect>();
  }

  void runOnOperation() override {
    // context variables
    auto circuit = getOperation();
    auto context = circuit->getContext();
    auto *body = circuit.getBodyBlock();
    //auto &nlaTable = getAnalysis<NLATable>();
    circt::igraph::InstanceGraph *inst_graph = &getAnalysis<InstanceGraph>();
    markAllAnalysesPreserved();

    // file management
    std::ofstream log_f(logFilename);
    log_f << "Hello, world! Parsing " << maxNumClusters << " clusters from " << powerFilename << std::endl;
    std::ifstream file(powerFilename);
    if (!file) {
        log_f << "Unable to open the file!" << std::endl;
        return;
    }

    // initialize power counters and lists
    double clock_power = 0.0;
    double idle_power = 0.0;
    power_clusters_t clusters;

    // find top-level module for the CIRCT IR
    circt::igraph::InstanceGraphNode *top_node = inst_graph->getTopLevelNode();
    auto top_mod_ptr = top_node->getModule().getOperation();
    firrtl::FModuleLike top_module = dyn_cast<firrtl::FModuleLike>(top_mod_ptr);
    log_f << "Simulation top module " << std::string(top_module.getModuleName().data()) << std::endl; // firrtl.module

    // find path from top-level IR to top-level PNR
    // TestHarness
    // ChipTop chiptop0
    // DigitalTop system
    llvm::SmallVector<firrtl::InstanceOp, 16> pnr_top_path;
    circt::igraph::InstanceGraphNode *pnr_top_node = top_node;
    firrtl::InstanceOp inst_ptr = find_instance(pnr_top_node, "chiptop0", &pnr_top_node);
    pnr_top_path.push_back(inst_ptr);
    inst_ptr = find_instance(pnr_top_node, "system", &pnr_top_node);
    pnr_top_path.push_back(inst_ptr);

    // find top-level module in place and route
    auto it = llvm::find_if(*body, [&](Operation &op) -> bool {
        if (auto mod = dyn_cast<firrtl::FModuleLike>(op)) {
            log_f << "Module name " << mod.getModuleName().data() << std::endl;
            if (std::string(mod.getModuleName().data()) == pnrTop) {
                return true;
            }
        }
        return false;
    });
    if (it != body->end()) {
        top_module = dyn_cast<firrtl::FModuleLike>(*it);
        log_f << "Found top-level module with name " << top_module.getModuleName().data() << std::endl;
        top_node = inst_graph->lookup(cast<igraph::ModuleOpInterface>(*it));
    }

    log_f << "PNR top module " << top_node->getModule().getOperation()->getName().getIdentifier().str() << std::endl; // firrtl.module

    // =================================================
    // ===== associate power values with registers =====
    // =================================================

    // iterate through all power entries
    std::string line;
    uint32_t cluster_id = 1;
    while (!file.eof()) {
        // get line
        std::getline(file, line);
        if (!(int)line.length()) {
            continue;
        }

        // get arguments
        int delim = 0;
        int next_delim;

        next_delim = (int)line.find(',');
        std::string name = line.substr(delim, next_delim);
        delim = next_delim + 1;
        next_delim = (int)line.find(',', delim);
        double power = std::stod(line.substr(delim, next_delim - delim), nullptr);
        delim = next_delim + 1;
        next_delim = (int)line.find(',', delim);
        unsigned int num_instances = std::stoi(line.substr(delim, next_delim - delim));
        delim = next_delim + 1;
        next_delim = line.size();
        unsigned int bus_width = std::stoi(line.substr(delim));

        if (name == "clock") {
            clock_power += power;
        }
        else {
            // strip array and register tags
            bool is_bus;
            bool is_replicated;
            std::string stripped_name = strip_misc_name(name, &is_bus, &is_replicated);
            log_f << "Signal name is " << stripped_name << " is_bus = " << (is_bus ? "yes" : "no") << " is_replicated = " << (is_replicated ? "yes" : "no") << " power = " << power << " bus_width = " << bus_width << " num_instances = " << num_instances << std::endl;

            // ignore bus signals
            //if (is_bus) {
                //idle_power += power;
                //continue;
            //}
            // TODO bits [1], [2], ... not found until strip the name so we avoid adding duplicate signals

            // start entry to track metadata
            power_cluster_t entry;
            entry.id = cluster_id;
            entry.power = power;
            entry.name = stripped_name;
            entry.nodes = std::vector<reg_node_t>(1);

            // find module
            circt::igraph::InstanceGraphNode *node = top_node;
            llvm::SmallVector<firrtl::InstanceOp, 16> insts(pnr_top_path);
            uint32_t insts_in_path = 0;

            int sidx = 0;
            std::string p;
            while (true) {
                // get through module block
                firrtl::FModuleOp mod;
                circt::igraph::ModuleOpInterface mod_if = node->getModule();
                circt::igraph::InstanceGraphNode *new_node = nullptr;

                int new_sidx = stripped_name.find('/', sidx);
                if (new_sidx == -1 &&
                    (mod = dyn_cast<firrtl::FModuleOp>(mod_if.getOperation()))) {
                    // find register matching reference name
                    //entry.module = mod;
                    p = stripped_name.substr(sidx);
                    log_f << "Searching for register with name " << p << std::endl;

                    // get operations from module in node
                    auto *body = mod.getBodyBlock();
                    auto it = llvm::find_if(*body, [&](Operation &op) -> bool {
                        if (auto reg = dyn_cast<firrtl::RegOp>(op)) {
                            log_f << "Register name " << reg.getName().data() << std::endl;
                            if (std::string(reg.getName().data()) == p) {
                                return true;
                            }
                        }
                        if (auto reg = dyn_cast<firrtl::RegResetOp>(op)) {
                            log_f << "Register (with reset) name " << reg.getName().data() << std::endl;
                            if (std::string(reg.getName().data()) == p) {
                                return true;
                            }
                        }
                        return false;
                    });

                    // add target to cluster
                    if (it != body->end()) {
                        // add target to cluster
                        std::string reg_name = "NULL";
                        firrtl::RegOp reg_target;
                        firrtl::RegResetOp reg_reset_target;
                        Operation *op;
                        if ((reg_target = dyn_cast<firrtl::RegOp>(*it))) {
                            // save in cluster
                            entry.nodes[0] = REG_NODE_T(false, reg_target, nullptr, reg_target);
                            entry.indicator_path = circt::firrtl::AnnoPathValue(
                              insts,
                              firrtl::AnnoTarget(firrtl::detail:: 	AnnoTargetImpl(reg_target)),
                              0);
                            op = reg_target;
                        }
                        else if ((reg_reset_target = dyn_cast<firrtl::RegResetOp>(*it))) {
                            entry.nodes[0] = REG_NODE_T(true, nullptr, reg_reset_target, reg_reset_target);
                            entry.indicator_path = circt::firrtl::AnnoPathValue(
                              insts,
                              firrtl::AnnoTarget(firrtl::detail:: 	AnnoTargetImpl(reg_reset_target)),
                              0);
                            op = reg_reset_target;
                        }

                        // track fine-grained information
                        tag_register(context, op, name, power, num_instances, bus_width, cluster_id, &entry.indicator_fan_out);
                        entry.id = cluster_id;
                        entry.indicator_idx = 0;
                        entry.indicator_bus_width = bus_width;
                    }
                    else {
                        log_f << "Could not find register for path segment " << p << std::endl;
                        node = nullptr;
                    }
                    break;
                }
                else {
                    // find current segment
                    p = stripped_name.substr(sidx, new_sidx - sidx);

                    // corner case with element_reset_domain_*tile
                    std::string domain_identifier = "element_reset_domain";
                    if (p.substr(0, domain_identifier.size()) == domain_identifier.c_str()) {
                      p = domain_identifier;
                      new_sidx = sidx + domain_identifier.size();
                    }

                    // advance index to next segment
                    sidx = new_sidx + 1;
                    log_f << "Searching for segment " << p << std::endl;

                    // search for instance
                    firrtl::InstanceOp inst_ptr = find_instance(node, p, &new_node);
                    insts.push_back(inst_ptr);
                    ++insts_in_path;
                }

                // follow chain or break
                node = new_node;
                if (node == nullptr) {
                    log_f << "Could not find module for path segment " << p << " in signal " << name << std::endl;
                    break;
                }
            }

            // could not find reference, so leave as "idle"
            if (node == nullptr || insts_in_path == 0) {
                idle_power += power;
                continue;
            }

            // add cluster to list
            clusters[cluster_id] = entry;
            ++cluster_id;
        }
    }
    file.close();

    // ==========================
    // ===== run clustering =====
    // ==========================

    PowerClusterer *clusterer = PowerClusterer::create(clusteringAlg, clusteringAlgArgs, maxNumClusters);
    log_f << "Cluster using the " << clusteringAlg << " class" << std::endl;
    if (clusterer) {

      clusterer->runOnCircuit(
        context,
        circuit,
        inst_graph,
        clusters,
        &idle_power,
        log_f
      );

      delete clusterer;

    }
    else {
      log_f << "Unrecognized clustering algorithm, aborting" << std::endl;
    }

    // =========================
    // ===== write outputs =====
    // =========================

    // List of outputs for each cluster
    //   perf.trace operation
    //   name,power,temperature_credit in csv
    //   list of signals inside the cluster
    //   signal_<i>,power,temperature_credit in csv
    //     one line for each signal with same power consumption
    //   generate adjacency list to other clusters

    // global information
    std::ofstream global_header_f(globalHeaderFilename);
    global_header_f << std::endl
      << "#ifndef __GLOBAL_POWER_TRACE_H__\n"
      << "#define __GLOBAL_POWER_TRACE_H__\n\n"
      << "#define CLOCK_POWER " << clock_power << "\n"
      << "#define IDLE_POWER " << idle_power << "\n"
      << "\n#endif // __GLOBAL_POWER_TRACE_H__\n";
    global_header_f.close();

    // create operations
    std::ofstream cluster_indicator_f(indicatorFilename);
    std::ofstream cluster_breakdown_f(clusterFilename);
    int i = 0;
    for (auto map : clusters) {
      int id = map.first;
      power_cluster_t entry = map.second;

      if (i >= maxNumClusters) {
        idle_power += entry.power;
        continue;
      }

      if (entry.power == 0.0 || entry.id == 0) { continue; }

      log_f << i << "(" << id << "): " << entry.name << " => " << entry.power << std::endl;

      //reg_node_t reg_target = entry.nodes[entry.indicator_idx];
      //mlir::Value reg_val;
      //if (REG_NODE_T_IS_RESET(reg_target)) {
      //  reg_val = REG_NODE_T_GET_REG_RESET_OP(reg_target).getResult();
      //}
      //else {
      //  reg_val = REG_NODE_T_GET_REG_OP(reg_target).getResult();
      //}
      log_f << "  Register " << entry.name << std::endl;

      log_f << "  Has path:" << std::endl;
      for (firrtl::InstanceOp &segment : entry.indicator_path.instances) {
        auto mod_ptr = segment->getParentOfType<circt::firrtl::FModuleOp>();
        log_f << "    "
              << std::string(segment.getName().data())
              << ": "
              << (mod_ptr
                ? std::string(mod_ptr.getModuleName().data())
                : "UNKNOWN")
              << std::endl;
      }

      // create perf operation for the indicator
      std::string name = "cluster_" + std::to_string(entry.id);
      std::string description = "\\\"" + entry.name + "\\\"," + std::to_string(entry.power) + "";

      log_f << "  Inserting trace operation for cluster " << entry.name << " with name " << name << std::endl;
      circt::perf::FIRRTLPerfInserter::insertTraceOp(entry.indicator_path,
        llvm::StringRef(name), llvm::StringRef(description));

      // write cluster indicator
      std::string rtl_path = get_reg_node_attr<StringAttr, llvm::StringRef>(
        entry.nodes[entry.indicator_idx],
        getRtlPathAttrName()).str();
      cluster_indicator_f << std::to_string(entry.id) << "," << rtl_path << std::endl;

      // write cluster breakdown for visualization scripts
      for (reg_node_t &reg : entry.nodes) {

        Operation *op = REG_NODE_T_GET_OP(reg);

        rtl_path = get_reg_node_attr<StringAttr, llvm::StringRef>(reg, getRtlPathAttrName()).str();
        cluster_breakdown_f << std::to_string(entry.id) << "," << rtl_path << std::endl;

        untag_register(context, op);
      }

      ++i;
    }
    cluster_breakdown_f.close();
    cluster_indicator_f.close();

    markAnalysesPreserved<InstanceGraph>();
  }
};
} // namespace

namespace circt {
namespace firrtl {

std::unique_ptr<mlir::Pass> createPowerTracePass() {
  return std::make_unique<PowerTracePass>();
}

} // namespace firrtl
} // namespace circt
