-- Where things happen, in beats from the start of the video (4 beats a
-- bar), read from the song (track.lua): its sections are the scenes, its
-- blips and stutters the pops the picture answers to, so picture and sound
-- share one list.

local score = require("track")

local T = {}

-- The beat at a position in the song's grammar ("source", "outro@1m").
local function at(pos) return score.beat(score.time(pos)) end
-- The song's hits of `part` in section `section`, in beats into it.
local function hits(part, section)
  local out, start = {}, at(section)
  for i, t in ipairs(score.hits { part = part, range = section }) do out[i] = score.beat(t) - start end
  return out
end

-- Sections: 2 bars each, the source 4; the outro has a bar of tail.
T.TITLE, T.SOURCE, T.RIPPLE, T.RAY, T.TUNNEL, T.NUMBERS, T.OUTRO = at("title"), at("source"), at("ripple"),
  at("ray"), at("tunnel"), at("numbers"), at("outro")
T.END = at("tail") -- the last hit
T.TOTAL = score.beat(score.duration)

T.TITLE_SLAM = at("slam") - T.TITLE
T.CODE_TYPED = at("source@1m") - T.SOURCE      -- source: the code types in over the first bar
T.OUTPUTS = hits("blips", "source")            -- source: SPIR-V, MSL, CPU pop one a beat
T.RIPPLE_HITS = hits("glitch", "ripple")       -- ripple: a ring and a glitch burst
do
  local pops = hits("blips", "numbers")        -- the three numbers (a blip and its fifth each)
  T.NUMBER_POPS = { pops[1], pops[3], pops[5] }
end
T.WORDMARK = at("outro@1m") - T.OUTRO

return T
