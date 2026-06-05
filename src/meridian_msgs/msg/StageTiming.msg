# Per-stage wall-time of one invocation, with sink-maintained running stats.
builtin_interfaces/Time stamp
string stage              # "preprocess","frontend.window_solve",...
float64 ms                # this invocation's wall time
float64 ms_avg            # running average (wrapper-maintained EWMA)
float64 ms_max            # running max since reset
uint64 count              # invocations since reset
