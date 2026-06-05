# One multiplexed scalar/vector telemetry channel, keyed by `key`.
builtin_interfaces/Time stamp
string key                # e.g. "frontend/lidar/n_inlier"
float64[] values          # 1 element for scalar(), N for vec()
string axis_order         # "" for scalar; e.g. "tx,ty,tz,rx,ry,rz" for vec
string unit               # documented unit, e.g. "count","m","rad","ratio"
