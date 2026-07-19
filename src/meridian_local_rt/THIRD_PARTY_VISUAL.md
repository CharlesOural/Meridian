# Visual frontend provenance

The implementation is an original Meridian API and state machine, informed by
the following research-code seams checked into `../slam-reference`:

- `GVINS/feature_tracker/src/feature_tracker.cpp`: distorted-pixel pyramidal
  KLT, lifetime-prioritized spatial suppression, corner replenishment, and
  normalized bearing output. Meridian adds deterministic 8x6 cell quotas,
  forward/backward validation, typed failure/reporting, exact timing/epoch
  checks, and an IMU-rotation initial-flow seed. No GVINS source is copied.
- `OKVIS2-X/okvis_cv/include/okvis/cameras/EquidistantDistortion.hpp` and
  `implementation/EquidistantDistortion.hpp`, plus
  `implementation/PinholeCamera.hpp`: the four-coefficient equidistant
  projection, inverse, and chain-rule Jacobian structure. Meridian adapts this
  BSD-licensed math into a small Eigen-only camera API, uses a safeguarded
  one-dimensional radial inverse, and returns unit bearings.
- `OKVIS2-X/okvis_frontend/src/Frontend.cpp`: keyframe overlap threshold and
  BRISK descriptors for keyframe/global-retrieval use. Meridian keeps BRISK out
  of ordinary tracking frames and makes recovery keyframes explicit.

## OKVIS2-X BSD-3-Clause license

OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense
Depth or LiDAR, and GNSS

Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich

Copyright (c) 2020, Smart Robotics Lab / Imperial College London

Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich and
ETH Zurich

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Autonomous Systems Lab / ETH Zurich, Smart Robotics Lab /
  Imperial College London, Mobile Robotics Lab / Technical University of
  Munich and ETH Zurich, nor the names of its contributors may be used to
  endorse or promote products derived from this software without specific
  prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
