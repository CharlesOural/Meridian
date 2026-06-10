#!/usr/bin/env bash
# Back-end iteration loop, runnable from the host. Every subcommand executes inside the
# `meridian` container (workspace mounted at /workspace, same paths as the host repo).
#
#   backend_dev.sh build                          # build the backend stack
#   backend_dev.sh test [ctest-regex]             # run backend-stack tests
#   backend_dev.sh dump <config> <bag_dir> <out_prefix>
#                                                 # replay bag -> <p>.tum + <p>.packets.bin (+index)
#   backend_dev.sh run  <config> <packets.bin> <out.tum> [backend_runner flags...]
#   backend_dev.sh loops <make_loops.py args...>  # generate an injected-loops yaml
#   backend_dev.sh check <config> <packets.bin>   # byte-determinism + identity property
#   backend_dev.sh eval <diagnose_run.py args...> # trajectory diagnosis
#
# Do not `build` while a replay/run is in flight: symlink-install swaps .so files under
# live processes.
set -euo pipefail

PKGS="meridian_common meridian_config meridian_calib meridian_debug meridian_backend meridian_pipeline meridian_ros"
RR=/workspace/install/meridian_ros/lib/meridian_ros/replay_runner
BR=/workspace/install/meridian_backend/lib/meridian_backend/backend_runner

DX() { docker exec meridian bash -lc "set +u; source /opt/ros/humble/setup.bash; cd /workspace && $*"; }
DXI() { docker exec meridian bash -lc "set +u; source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; cd /workspace && $*"; }

cmd="${1:?usage: backend_dev.sh build|test|dump|run|loops|check|eval ...}"
shift
case "$cmd" in
  build)
    DX "colcon build --symlink-install --packages-select $PKGS"
    ;;
  test)
    if [ $# -ge 1 ]; then
      DX "colcon test --packages-select $PKGS --ctest-args -R '$1' && colcon test-result --verbose"
    else
      DX "colcon test --packages-select $PKGS && colcon test-result --verbose"
    fi
    ;;
  dump)
    cfg="${1:?config}" bag="${2:?bag_dir}" prefix="${3:?out_prefix}"
    DXI "mkdir -p \$(dirname '$prefix') && $RR '$cfg' '$bag' '$prefix.tum' --dump-keyframes '$prefix.packets.bin'"
    ;;
  run)
    DXI "$BR $*"
    ;;
  loops)
    DX "python3 tools/make_loops.py $*"
    ;;
  check)
    cfg="${1:?config}" pkts="${2:?packets.bin}"
    DXI "$BR '$cfg' '$pkts' /tmp/bd_a.tum >/dev/null 2>&1"
    DXI "$BR '$cfg' '$pkts' /tmp/bd_b.tum >/dev/null 2>&1"
    DX "a=\$(md5sum </tmp/bd_a.tum | cut -d' ' -f1); b=\$(md5sum </tmp/bd_b.tum | cut -d' ' -f1); \
        echo \"md5 a=\$a b=\$b\"; [ \"\$a\" = \"\$b\" ] && echo DETERMINISM-OK || { echo DETERMINISM-FAIL; exit 1; }"
    DX "python3 tools/check_identity.py --index '$pkts.index.txt' --backend /tmp/bd_a.tum && echo IDENTITY-OK"
    ;;
  eval)
    DX "python3 tools/diagnose_run.py $*"
    ;;
  *)
    echo "unknown subcommand: $cmd" >&2
    exit 2
    ;;
esac
