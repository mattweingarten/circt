//===- StripNameAttributes.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Block.h"
#include "mlir/Pass/Pass.h"

#include <string>

using namespace circt;
using namespace circt::firrtl;

namespace {
struct StripNameAttributesPass
    : public StripNameAttributesBase<StripNameAttributesPass> {
  void runOnOperation() override;
};
} // namespace

static std::string makeUniqueName(llvm::StringRef base,
                                  llvm::StringMap<unsigned> &usedNames) {
  if (!usedNames.count(base)) {
    usedNames[base] = 1;
    return base.str();
  }

  unsigned suffix = 0;
  std::string candidate;
  do {
    candidate = (base + "_" + std::to_string(suffix++)).str();
  } while (usedNames.count(candidate));

  usedNames[candidate] = 1;
  return candidate;
}

static void uniquifyNamesInBlock(mlir::Block *block) {
  llvm::StringMap<unsigned> canonicalNames;
  llvm::StringMap<unsigned> allUsedNames;

  // First pass:
  // Collect canonical names of all FNamableOps in this block.
  for (mlir::Operation &op : *block) {
    if (auto nameableOp = llvm::dyn_cast<firrtl::FNamableOp>(&op)) {
      auto canonicalName = nameableOp.getNameAttr();
      if (canonicalName) {
        ++canonicalNames[canonicalName.getValue()];
        ++allUsedNames[canonicalName.getValue()];
      }
    }
  }

  // Second pass:
  // Rewrite explicit {name = "..."} attrs if they collide with names already
  // used in this block.
  for (mlir::Operation &op : *block) {
    auto explicitName = op.getAttrOfType<mlir::StringAttr>("name");
    if (!explicitName)
      continue;

    auto explicitNameStr = explicitName.getValue();

    bool shouldRename = false;

    // Rename if this explicit name collides with a canonical block-local name.
    auto canonIt = canonicalNames.find(explicitNameStr);
    if (canonIt != canonicalNames.end())
      shouldRename = true;

    // Also rename if another explicit attr in this block already claimed it.
    // To avoid self-conflict, remove this attr from the used set first, then
    // test/reinsert.
    auto usedIt = allUsedNames.find(explicitNameStr);
    if (usedIt != allUsedNames.end()) {
      if (--usedIt->second == 0)
        allUsedNames.erase(usedIt);
    }

    if (allUsedNames.count(explicitNameStr))
      shouldRename = true;

    if (!shouldRename) {
      ++allUsedNames[explicitNameStr];
      continue;
    }

    std::string newName = makeUniqueName(explicitNameStr, allUsedNames);

    // llvm::errs() << "[STRIP NAMES] In block @" << block << " rename explicit "
    //              << "name \"" << explicitNameStr << "\" -> \"" << newName
    //              << "\" on op " << op.getName().getStringRef() << "\n";

    op.setAttr("name", mlir::StringAttr::get(op.getContext(), newName));
  }

  // Recurse into nested regions/blocks.
  for (mlir::Operation &op : *block) {
    for (mlir::Region &region : op.getRegions()) {
      for (mlir::Block &nestedBlock : region)
        uniquifyNamesInBlock(&nestedBlock);
    }
  }
}

void StripNameAttributesPass::runOnOperation() {
  auto circuit = getOperation();

  circuit.walk([&](firrtl::FModuleOp module) {
    for (mlir::Region &region : module->getRegions()) {
      for (mlir::Block &block : region)
        uniquifyNamesInBlock(&block);
    }
  });
}

namespace circt {
namespace firrtl {

std::unique_ptr<mlir::Pass> createStripNameAttributesPass() {
  return std::make_unique<StripNameAttributesPass>();
}

} // namespace firrtl
} // namespace circt