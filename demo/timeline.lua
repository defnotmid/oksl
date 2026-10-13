-- Where things happen, in beats from the start of the video (4 beats a
-- bar). The scenes animate from these numbers and track.lua puts its hits
-- on them, so picture and sound share one list.

local T = {}

-- Sections: 2 bars each, the source 4; the outro has a bar of tail.
T.TITLE, T.SOURCE, T.RIPPLE, T.RAY, T.TUNNEL, T.NUMBERS, T.OUTRO, T.TOTAL = 0, 8, 24, 32, 40, 48, 56, 68
T.END = 64 -- the last hit

T.TITLE_SLAM = 4
T.CODE_TYPED = 4              -- source: the code types in over the first bar
T.OUTPUTS = { 8, 9, 10 }      -- source: SPIR-V, MSL, CPU pop one a beat
T.RIPPLE_HITS = { 0, 4 }      -- ripple: a ring and a glitch burst
T.NUMBER_POPS = { 0, 2, 4 }
T.WORDMARK = 4

return T
