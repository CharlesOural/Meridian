#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "meridian/common/keyframe_packet.hpp"
#include "meridian/common/loop_constraint.hpp"
#include "meridian/common/sample.hpp"

namespace meridian {

// One record read back from a packet log. `kind` selects which members are valid:
// Keyframe -> kf; Gnss -> fix + nearest_kf_id; Loop -> loop. The camera image handle
// is never serialized, so kf.image is always null on read; kf.cloud_body is null
// unless the log was written with clouds included.
struct PacketRecord {
  enum class Kind : std::uint8_t { Keyframe = 1, Gnss = 2, Loop = 3 };
  Kind kind = Kind::Keyframe;
  KeyframePacket kf;
  GnssFix fix;
  std::uint64_t nearest_kf_id = 0;
  LoopConstraint loop;
};

// Versioned binary log of the back-end input stream (keyframes, GNSS fixes, loop
// closures), bit-exact on round-trip. Alongside the binary file it maintains a
// human-readable text index at <path>.index.txt with one line per record.
class PacketLogWriter {
 public:
  // `include_clouds` controls whether keyframe point clouds are embedded in the log.
  PacketLogWriter(const std::string& path, bool include_clouds);
  ~PacketLogWriter();  // flushes and closes both files
  PacketLogWriter(const PacketLogWriter&) = delete;
  PacketLogWriter& operator=(const PacketLogWriter&) = delete;

  void write_keyframe(const KeyframePacket& kf);
  void write_gnss(const GnssFix& fix, std::uint64_t nearest_kf_id);
  void write_loop(const LoopConstraint& loop);

 private:
  void write_record(std::uint8_t kind, const std::vector<std::uint8_t>& payload);

  std::ofstream bin_;
  std::ofstream index_;
  bool include_clouds_;
};

// Reads a log produced by PacketLogWriter. Throws std::runtime_error on a bad
// magic/version header.
class PacketLogReader {
 public:
  explicit PacketLogReader(const std::string& path);

  bool clouds_included() const { return clouds_included_; }

  // Fills *out with the next record. Returns false at clean end-of-file; throws
  // std::runtime_error if the file ends mid-record.
  bool next(PacketRecord* out);

 private:
  std::ifstream bin_;
  bool clouds_included_ = false;
};

}  // namespace meridian
