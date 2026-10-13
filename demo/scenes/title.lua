-- "OKSL" drawn into a text group that glitch.sksl reads as its `uniform
-- shader` child: slices and colour split in the intro bar, a ripple ring
-- from the slam.

local beat = require("beat")
local ui = require("ui")
local T = require("timeline")
local prog, snap, soft, pulse = beat.prog, beat.snap, beat.soft, beat.pulse

local BOX = { x = 0, y = 200, w = 1920, h = 520 }

local function word()
  text { "OKSL", x = 960, y = 400, align = "center", font = ui.DISPLAY, size = 420,
    fill = linear_gradient { x1 = 0, y1 = 80, x2 = 0, y2 = 400, stops = { { 0, "#eaffc4" }, { 1, ui.A } } } }
end

local function render(f, lb)
  local slam = T.TITLE_SLAM
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = radial_gradient { cx = 960, cy = 470, r = 900,
    stops = { { 0, "#1d1236" }, { 1, ui.BG } } } }
  local glitch, opacity, s
  if lb < slam then
    local on = beat.hash(math.floor(lb * 4)) > 0.35 or pulse(lb, 6) > 0.5
    glitch = 0.5 + 0.5 * pulse(lb, 4)
    opacity = on and 0.9 or 0.2
    s = 0.9
  else
    glitch = math.max(math.exp(-(lb - slam) * 3), 0.12 * pulse(lb, 10))
    opacity = 1
    s = 1 + 0.15 * (1 - soft(lb - slam))
  end
  group({ opacity = opacity, transform = { scale = s, origin = { 960, BOX.y + 260 } } }, function()
    shader(f, { src = "glitch.sksl", x = BOX.x, y = BOX.y, w = BOX.w, h = BOX.h,
      uniforms = { uText = word, uGlitch = glitch, uSeed = math.floor(lb * 8),
                   uRipple = lb >= slam and (lb - slam) * beat.BEAT or -1 } })
  end)
  if lb >= slam + 1 then
    local p = snap(lb - slam - 1)
    group({ opacity = prog(lb, slam + 1, slam + 1.2), transform = translate(0, (1 - p) * 30) }, function()
      text { "SkSL → SPIR-V · MSL · CPU, in plain Odin", x = 960, y = 770, align = "center", font = ui.SANS,
        size = 50, fill = ui.TEXT }
    end)
  end
  if lb >= slam + 2 then
    local label = "this title: SkSL reading a live text group, compiled by oksl"
    local w = text_width { label, font = ui.MONO, size = 22 } + 31
    ui.chip(960 - w / 2, 830, label, ui.LINE, ui.A, 22, snap(lb - slam - 2))
  end
end

return { name = "Title", start = T.TITLE, stop = T.SOURCE, render = render }
