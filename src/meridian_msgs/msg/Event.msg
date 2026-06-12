# Structured event; also mirrored to the text log.
builtin_interfaces/Time stamp
uint8 level               # 0 Trace .. 4 Error (matches meridian::Level)
string tag                # "frontend/window_restart","place/loop_accepted",...
string message            # human-readable detail (structured key=value)
