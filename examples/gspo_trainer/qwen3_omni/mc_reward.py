# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exact-match reward for the FineVideo multiple-choice task.

Wire into verl via:
  custom_reward_function.path=<this file>
  custom_reward_function.name=compute_score

The model is asked to answer with a single letter (A/B/C/D). We extract that
letter robustly and give 1.0 for an exact match with ground_truth, else 0.0.
"""

import re

_CHOICES = "ABCD"
# Explicit-answer patterns first (most reliable), then a bare trailing letter.
_EXPLICIT = re.compile(r"(?:answer|choice|option)\s*(?:is|:|=)?\s*\(?\s*([A-D])\b", re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{\s*([A-D])\s*\}", re.IGNORECASE)
_BARE = re.compile(r"\b([A-D])\b")


def extract_choice(text):
    """Return the chosen letter (upper-case) from a model response, or None."""
    if not text:
        return None
    for pat in (_BOXED, _EXPLICIT):
        m = list(pat.finditer(text))
        if m:
            return m[-1].group(1).upper()
    m = list(_BARE.finditer(text))
    if m:
        return m[-1].group(1).upper()  # fall back to the last standalone A-D
    return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    gt = (ground_truth or "").strip().upper()
    if gt not in _CHOICES:
        return 0.0
    return 1.0 if extract_choice(solution_str) == gt else 0.0
