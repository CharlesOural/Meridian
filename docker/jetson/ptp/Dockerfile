# syntax=docker/dockerfile:1
#
# Free-running PTP grandmaster for the rig. Runs ptp4l (serves IEEE-1588 on end0,
# the Ouster's link) + phc2sys (keeps the end0 PHC equal to the host system clock),
# so the Ouster's PTP-disciplined timestamps land on the same clock as everything
# stamped at host time. No GPS: the timeline is relative-but-consistent. arm64.
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      linuxptp ethtool iproute2 \
 && rm -rf /var/lib/apt/lists/*

COPY ptp4l.conf /etc/ptp4l.conf
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
