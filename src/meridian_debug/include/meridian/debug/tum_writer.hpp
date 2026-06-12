#pragma once

#include <ostream>

#include "meridian/common/pose.hpp"
#include "meridian/common/time.hpp"

namespace meridian {

// Writes one TUM trajectory line: "stamp tx ty tz qx qy qz qw\n", stamp in seconds at
// 9 decimals, translation at 6, quaternion at 9. Leaves the stream in std::fixed with
// precision 9, so successive calls produce identical formatting.
inline void write_tum_line(std::ostream& out, Timestamp stamp, const Pose& T) {
  out << std::fixed;
  out.precision(9);
  out << stamp * 1e-9 << ' ';
  out.precision(6);
  out << T.t.x() << ' ' << T.t.y() << ' ' << T.t.z() << ' ';
  out.precision(9);
  out << T.q.x() << ' ' << T.q.y() << ' ' << T.q.z() << ' ' << T.q.w() << '\n';
}

}  // namespace meridian
