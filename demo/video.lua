-- oksl in 27 seconds: an onimite script (Odin + Lua video framework,
-- ../../onimite) showing the SkSL compiler its shader layers run through.
-- Music included: the soundtrack is synthesized in Lua when the script
-- loads (track.lua) and every cut and pop sits on its 150 BPM grid
-- (beat.lua, timeline.lua). Numbers on screen: facts.lua says where each
-- comes from.
--
--     onimite render demo/video.lua -o demo/demo.mp4

local beat = require("beat")
local ui = require("ui")
local T = require("timeline")
local track = require("track")

-- Synthesized once per render (shared), registered as a media clip.
audio_add("track.wav", shared("track", track.build))

local SCENES = {
  require("scenes.title"),
  require("scenes.source"),
  require("scenes.ripple"),
  require("scenes.ray"),
  require("scenes.tunnel"),
  require("scenes.numbers"),
  (require("scenes.outro")), -- parenthesized: require also returns the file name
}

local SECTIONS = { "01 / TITLE", "02 / ONE SOURCE, THREE OUTPUTS", "03 / RIPPLE + GLITCH", "04 / RAYMARCH",
  "05 / SHADERTOY", "06 / NUMBERS", "07 / OUTRO" }

-- Bars where the kick plays: the picture pumps with it.
local PUMPING = { { T.TITLE_SLAM, T.RAY }, { T.TUNNEL, T.END } }
-- Cuts that glitch for three frames, and the impacts that flash.
local GLITCH_CUTS = { T.SOURCE, T.RIPPLE, T.RAY, T.TUNNEL, T.NUMBERS, T.OUTRO }
local FLASHES = { T.TITLE_SLAM, T.OUTRO }

local function inside(b, spans)
  for _, s in ipairs(spans) do if b >= s[1] and b < s[2] then return true end end
  return false
end

local function energy(f, b)
  local s = 1 + (inside(b, PUMPING) and beat.pulse(b, 9) or 0) * 0.008
  local since_cut
  for _, c in ipairs(GLITCH_CUTS) do
    local fr = f.global_index - beat.beat_frame(c)
    if fr >= 0 and fr < 3 then since_cut = fr end
  end
  group({ transform = { scale = s, origin = { 960, 540 } } }, function()
    if since_cut then
      -- the picture splits in bands pushed sideways
      for i = 0, 5 do
        local dx = (beat.hash(i * 13 + since_cut * 7 + b) - 0.5) * 160 * (1 - since_cut / 3)
        group({ transform = translate(dx, 0), clip = { x = 0, y = i * 180, w = 1920, h = 180 } }, function()
          render_scenes(f)
        end)
      end
    else
      render_scenes(f)
    end
  end)
  local flash = 0
  for _, d in ipairs(FLASHES) do
    if b >= d then flash = math.max(flash, math.exp(-(b - d) * 5)) end
  end
  if flash > 0.01 then rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.TEXT, opacity = flash * 0.5 } end
end

local function hud(f, b)
  if b >= T.END then return end
  local bar, q = math.floor(b / 4) + 1, math.floor(b % 4) + 1
  local section = 1
  for i, s in ipairs(SCENES) do if b >= s.start then section = i end end
  group({ opacity = 0.8 }, function()
    text { "OKSL · SKSL COMPILER", x = 60, y = 66, font = ui.MONO, size = 16, fill = ui.DIM }
    text { string.format("%d BPM   BAR %02d.%d", beat.BPM, bar, q), x = 1830, y = 66, align = "right",
      font = ui.MONO, size = 16, fill = ui.DIM }
    rect { x = 1844, y = 53, w = 14, h = 14, fill = ui.A, opacity = 0.2 + 0.8 * beat.pulse(b, 7) }
    text { SECTIONS[section], x = 1860, y = 1050, align = "right", font = ui.MONO, size = 16, fill = ui.DIM }
    rect { x = 60, y = 1062, w = 1800, h = 1, fill = ui.LINE }
    rect { x = 60, y = 1061, w = math.max(1800 * b / T.TOTAL, 1), h = 3, fill = ui.DIM }
  end)
end

video {
  width = 1920, height = 1080, fps = beat.FPS, background = ui.BG,
  audio = { { "track.wav", start = 0, fade_out = 0.5 } },
  -- the synth peaks at -1 dBFS: lift it to about -14 LUFS under a limiter
  -- that keeps true peaks below -1 dBTP (audio analyze in README.md)
  master_gain_db = 4.0,
  limiter = { ceiling_db = -3.4 },
  render = function(f)
    local b = beat.gbeat(f)
    rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.BG }
    energy(f, b)
    hud(f, b)
  end,
}

for _, s in ipairs(SCENES) do
  local start, render = s.start, s.render
  scene { s.name, duration = { frames = beat.beat_frame(s.stop) - beat.beat_frame(start) }, render = function(f)
    render(f, beat.gbeat(f) - start)
  end }
end
