/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNonTranspose final : public PredicateBasedPass {
  explicit EliminateNonTranspose()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override { return "eliminate_non_transpose"; }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kTranspose;
  }
  bool runTransform(Node* node, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    if (node->hasAttribute(kperm)) {
      auto perm = node->is(kperm);
      for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != i) {
          return false;
        }
      }
    }
    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), node->input());
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
