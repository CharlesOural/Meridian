#!/usr/bin/env bash
# Serve PTP on end0 (grandmaster) and tie the NIC PHC to the host system clock.
#   ptp4l   -> serves IEEE-1588 to the Ouster from the end0 PHC
#   phc2sys -> source = host CLOCK_REALTIME, target = end0 PHC, so PTP time == ROS time
# Either dying takes the container down (restart policy brings it back).
set -e
IFACE="${PTP_IFACE:-end0}"
SENSOR_IP="${SENSOR_HOST_IP:-192.168.100.1/24}"

# end0 needs a routable (non-link-local) address: a 169.254/16 link-local address
# is forced to link scope and cannot be the source for a global PTP multicast
# group, so the kernel would source PTP from the other NIC and the Ouster (on the
# sensor subnet) would ignore the grandmaster. This matches the Ouster's static IP.
if ! ip -4 addr show dev "${IFACE}" | grep -qw "${SENSOR_IP%/*}"; then
  echo "[ptp] assigning ${SENSOR_IP} to ${IFACE}"
  ip addr add "${SENSOR_IP}" dev "${IFACE}"
fi

echo "[ptp] end0 timestamping capabilities:"
ethtool -T "${IFACE}" 2>/dev/null | sed 's/^/[ptp]   /' || true

echo "[ptp] starting ptp4l grandmaster on ${IFACE}"
ptp4l -i "${IFACE}" -f /etc/ptp4l.conf &
PTP4L_PID=$!

# Let ptp4l bind the port before phc2sys attaches.
sleep 3

echo "[ptp] starting phc2sys: CLOCK_REALTIME -> ${IFACE} PHC"
phc2sys -s CLOCK_REALTIME -c "${IFACE}" -w -O 0 -m &
PHC2SYS_PID=$!

wait -n "${PTP4L_PID}" "${PHC2SYS_PID}"
