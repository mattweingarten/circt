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

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/Perf/PerfHelpers.h"
#include "mlir/IR/Threading.h"

#include "circt/Dialect/Perf/PerfDialect.h"
#include "circt/Dialect/Perf/PerfHelpers.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "circt/Dialect/Perf/PerfPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "power-trace"

namespace circt {
namespace perf {
#define GEN_PASS_DEF_INSERTTRACE
#include "circt/Dialect/Perf/PerfPasses.h.inc"
} // namespace perf
} // namespace circt

using namespace circt;
using namespace circt::perf;
using namespace firrtl;

inline StringRef getPowerAnnotationAttrName() { return "annotations"; }
inline StringRef getPowerAttrName() { return "power"; }
inline StringRef getNumInstancesAttrName() { return "num_instances"; }
inline StringRef getBusWidthAttrName() { return "bus_width"; }

inline FloatAttr getFloatAttr(MLIRContext *context, double f) { return FloatAttr::get(mlir::FloatType::getF64(context), f); }
inline IntegerAttr getUintAttr(MLIRContext *context, uint32_t d) {
  return IntegerAttr::get(
    mlir::IntegerType::get(
      context, 32,
      mlir::IntegerType::Signless),
    d);
}



// strip register and array index from name
static std::string strip_misc_name(std::string name, bool *is_bus, bool *is_replicated) {
    // strip bus tag
    int size = name.length();
    int idx = name.rfind("[0]/q");
    std::string ret = name;
    if (idx == size - 5) {
        ret = name.substr(0, idx);
        *is_bus = true;
    }
    else {
        // ret = ret;
        *is_bus = false;
    }

    // strip register tag
    size = ret.length();
    idx = ret.rfind("/q");
    if (idx == size - 2) {
        ret = name.substr(0, idx);
    }

    // determine if one of many instances
    size = ret.length();
    idx = ret.find("[0]");
    *is_replicated = idx != -1;

    return ret;
}

// power entry type
typedef struct {
  double power;
  std::string name;
  uint32_t num_instances;
  uint32_t bus_width;
  firrtl::RegOp reg;
  firrtl::RegResetOp regReset;
  firrtl::FModuleOp module;
} register_node_entry_t;

typedef struct {

  firrtl::FModuleOp module;
  firrtl::FNamableOp reg_op;
  bool is_reset;
  double power;

} register_node_t;

// power node
typedef std::tuple<bool, firrtl::RegOp, firrtl::RegResetOp> reg_node_t;
#define REG_NODE_T(is_reset, reg_op, reg_reset_op) std::tuple<bool, firrtl::RegOp, firrtl::RegResetOp>{is_reset, reg_op, reg_reset_op}
#define REG_NODE_T_IS_RESET(n) std::get<0>(n)
#define REG_NODE_T_GET_REG_OP(n) std::get<1>(n)
#define REG_NODE_T_GET_REG_RESET_OP(n) std::get<2>(n)

// power cluster
typedef struct {

  firrtl::FModuleOp module;
  double power;
  std::string name;
  std::vector<reg_node_t> nodes;
  int indicator_idx;

} power_cluster_t;

static void merge_clusters(std::vector<power_cluster_t> &clusters, int idx0, int idx1, int num_clusters) {
  // add registers in clusters[idx1] to clusters[idx0]
  power_cluster_t flattened = clusters[idx1];

  // shift over merged cluster
  for (int i = idx1; i < num_clusters; ++i) {
    clusters[i] = clusters[i + 1];
  }
  clusters.resize(clusters.size()-1);
}

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
    std::vector<power_cluster_t> clusters;

    // find top-level module in place and route
    firrtl::FModuleLike top_module;
    circt::igraph::InstanceGraphNode *top_node;
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

    log_f << "Top module " << top_node->getModule().getOperation()->getName().getIdentifier().str() << std::endl; // firrtl.module

    // iterate through all power entries
    // associate power values with registers
    std::string line;
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
            if (is_bus) {
                idle_power += power;
                continue;
            }

            // start entry to track metadata
            power_cluster_t entry = {
              nullptr,
              power,
              stripped_name,
              std::vector<reg_node_t>(1),
              0
            };

            // find module
            circt::igraph::InstanceGraphNode *node = top_node;
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
                    entry.module = mod;
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
                        std::string reg_name = "NULL";
                        firrtl::RegOp reg_target;
                        firrtl::RegResetOp reg_reset_target;
                        if ((reg_target = dyn_cast<firrtl::RegOp>(*it))) {
                            // add target to cluster
                            entry.nodes[0] = REG_NODE_T(false, reg_target, nullptr);
                            entry.indicator_idx = 0;

                            //reg_name = reg_target.getName().data();
                            reg_target->setAttr(getPowerAttrName(), getFloatAttr(context, power));
                            reg_target->setAttr(getNumInstancesAttrName(), getUintAttr(context, num_instances));
                            reg_target->setAttr(getBusWidthAttrName(), getUintAttr(context, bus_width));
                        }
                        else if ((reg_reset_target = dyn_cast<firrtl::RegResetOp>(*it))) {
                            // add target to cluster
                            entry.nodes[0] = REG_NODE_T(true, nullptr, reg_reset_target);
                            entry.indicator_idx = 0;

                            //reg_name = reg_reset_target.getName().data();
                            reg_reset_target->setAttr(getPowerAttrName(), getFloatAttr(context, power));
                            reg_reset_target->setAttr(getNumInstancesAttrName(), getUintAttr(context, num_instances));
                            reg_reset_target->setAttr(getBusWidthAttrName(), getUintAttr(context, bus_width));
                        }

                        //log_f << "Found target register with name " << reg_name << std::endl;
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
                    sidx = new_sidx + 1;
                    log_f << "Searching for segment " << p << std::endl;

                    // corner case with element_reset_domain_*tile
                    std::string domain_identifier = "element_reset_domain";
                    if (p.substr(0, domain_identifier.size()) == domain_identifier.c_str()) {
                      p = domain_identifier;
                    }

                    // get instances from module in node
                    for (auto it = node->begin(); it != node->end(); it++) {
                        auto inst_node = (*it)->getTarget();
                        auto inst = (*it)->getInstance().getOperation();
                        auto module = inst_node->getModule().getOperation();

                        auto inst_ptr = dyn_cast<firrtl::InstanceOp>(inst);
                        auto mod_ptr = dyn_cast<firrtl::FModuleLike>(module);

                        log_f << "Instance " << inst_ptr.getName().data() << " of module " << mod_ptr.getModuleName().data() << std::endl;

                        if (std::string(inst_ptr.getName().data()) == p) {
                            std::string modName = std::string(mod_ptr.getModuleName().data());
                            new_node = inst_node;
                            log_f << "Module for path segment " << p << " is " << modName << std::endl;
                            break;
                        }
                    }
                }

                // follow chain or break
                node = new_node;
                if (node == nullptr) {
                    log_f << "Could not find module for path segment " << p << " in signal " << name << std::endl;
                    break;
                }
            }

            // could not find reference, so leave as "idle"
            if (node == nullptr) {
                idle_power += power;
                continue;
            }

            // add cluster to list
            clusters.push_back(entry);
        }
    }
    file.close();

    // reduce number of clusters
    int num_clusters = (int)clusters.size();
    int i = 0;
    for (; i < maxNumClusters && i < num_clusters; ++i) {
        // find maximum power cluster
        int max_power_i = i;
        double max_power = clusters[i].power;
        for (int j = i + 1; j < num_clusters; ++j) {
            if (clusters[j].power > max_power) {
                max_power_i = j;
                max_power = clusters[j].power;
            }
        }

        // swap clusters
        if (max_power_i != i) {
            power_cluster_t tmp = clusters[i];
            clusters[i] = clusters[max_power_i];
            clusters[max_power_i] = tmp;
        }
    }
    for (; i < num_clusters; ++i) {
        idle_power += clusters[i].power;
    }
    /*
    while (num_clusters > numClusters) {
        // find slot in power array
        int idx = 0;
        for (; idx < numClusters; ++idx) {
            if (power > max_power_entries[idx].power) {
                break;
            }
        }

        // determine "significant enough" so leave as idle
        if (idx == numClusters) {
            idle_power += power;
            continue;
        }
        else {
            log_f << "  Insert at index " << idx << " which has power " << max_power_entries[idx].power << std::endl;
        }

        // make space for power-significant event
        idle_power += max_power_entries[numClusters-1].power;
        for (int j = numClusters-1; j >= idx + 1; --j) {
            max_power_entries[j] = max_power_entries[j-1];
        }
        max_power_entries[idx] = entry;
    }
    */

    // create operations
    i = 0;
    //for (power_entry_t &entry : max_power_entries) {
    for (power_cluster_t &entry : clusters) {

        if (entry.power == 0.0) { break; }

        //log_f << i << ": " << entry.name << " => " << entry.power << ", " << entry.num_instances << "x" << entry.bus_width << std::endl;
        log_f << i << ": " << entry.name << " => " << entry.power << std::endl;

        reg_node_t reg_target = entry.nodes[entry.indicator_idx];
        mlir::Value reg_val;
        if (REG_NODE_T_IS_RESET(reg_target)) {
            reg_val = REG_NODE_T_GET_REG_RESET_OP(reg_target).getResult();
        }
        else {
            reg_val = REG_NODE_T_GET_REG_OP(reg_target).getResult();
        }
        log_f << "  Register " << entry.name << std::endl;

        circt::perf::FIRRTLPerfInserter::insertTraceOp(reg_val, entry.module,
          llvm::StringRef(entry.name), llvm::StringRef(entry.name + "desc"));

#ifdef POWER_TRACE_ADD_ANNO
        SmallVector<Attribute> newAnnos;
        if (!annosAttr) {
            log_f << "Target " << entry.name << " does not have any annotations" << std::endl;
            newAnnos.reserve(1);
        }
        else {
            newAnnos.reserve(annosAttr.size() + 1);

            for (Attribute anno : annosAttr) {
                newAnnos.push_back(anno);
            }

            SmallVector<NamedAttribute> fields;
            fields.emplace_back(
                StringAttr::get(context, "class"),
                StringAttr::get(context, "freechips.rocketchip.util.PowerAnnotation")
            );
            fields.emplace_back(
                StringAttr::get(context, "power"),
                FloatAttr::get(mlir::FloatType::getF64(context), entry.power)
            );
            fields.emplace_back(
                StringAttr::get(context, "num_instances"),
                IntegerAttr::get(
                    mlir::IntegerType::get(context, 32, mlir::IntegerType::Signless),
                    entry.num_instances)
            );
            fields.emplace_back(
                StringAttr::get(context, "bus_width"),
                IntegerAttr::get(
                    mlir::IntegerType::get(context, 32, mlir::IntegerType::Signless),
                    entry.bus_width)
            );

            newAnnos.push_back(DictionaryAttr::get(context, fields));
        }

        // attach annotation to target
        // TODO attach to specific instance, not just the operation
        log_f << "Attaching " << (newAnnos.size()) << " annotation(s) to " << entry.name << std::endl;
        if (entry.reg) {
            entry.reg->setAttr(StringAttr::get(context, getPowerAnnotationAttrName()), ArrayAttr::get(context, newAnnos));
        }
        else {
            entry.regReset->setAttr(StringAttr::get(context, getPowerAnnotationAttrName()), ArrayAttr::get(context, newAnnos));
        }

#endif // POWER_TRACE_ADD_ANNO

        ++i;
    }

    markAnalysesPreserved<InstanceGraph>();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createPowerTracePass() {
  return std::make_unique<PowerTracePass>();
}
