-- glitch.sksl over a text group child: a ripple ring and a glitch burst
-- on the two hits, small kicks on every beat.

local beat = require("beat")
local ui = require("ui")
local T = require("timeline")
local prog, snap, soft, pulse = beat.prog, beat.snap, beat.soft, beat.pulse

local function words()
  text { "RIPPLE", x = 960, y = 330, align = "center", font = ui.DISPLAY, size = 300, fill = ui.TEXT }
  text { "& GLITCH", x = 960, y = 560, align = "center", font = ui.DISPLAY, size = 220, fill = ui.B }
end

local function render(f, lb)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = radial_gradient { cx = 960, cy = 540, r = 900,
    stops = { { 0, "#22103a" }, { 1, ui.BG } } } }
  local last = T.RIPPLE_HITS[1]
  for _, h in ipairs(T.RIPPLE_HITS) do if lb >= h then last = h end end
  local glitch = math.max(math.exp(-(lb - last) * 2.5) * (last == T.RIPPLE_HITS[2] and 1 or 0.4), 0.25 * pulse(lb, 9))
  local s = 1 + 0.03 * pulse(lb, 8)
  group({ transform = { scale = s, origin = { 960, 540 } } }, function()
    shader(f, { src = "glitch.sksl", x = 0, y = 180, w = 1920, h = 660,
      uniforms = { uText = words, uGlitch = glitch, uSeed = math.floor(lb * 8), uRipple = beat.seconds(lb - last) } })
  end)
  text { "uniform shader uText;  ·  uText = function() text{ … } end", x = 960, y = 900, align = "center",
    font = ui.MONO, size = 26, fill = ui.DIM }
  ui.label(lb, 2, "a group of nodes as the shader's child")
end

return { name = "Ripple", start = T.RIPPLE, stop = T.RAY, render = render }
