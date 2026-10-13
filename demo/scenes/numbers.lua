-- Three claims a bar, each with its source, over rings.sksl dimmed.

local beat = require("beat")
local ui = require("ui")
local T = require("timeline")
local facts = require("facts")
local prog, snap, pulse = beat.prog, beat.snap, beat.pulse

local ROWS = {
  { "GPU = CPU", string.format("%d / %d test shaders match · %d within 1/255", facts.PARITY_MATCH, facts.PARITY_TOTAL,
    facts.PARITY_WITHIN1), facts.PARITY_SRC .. " · run 2026-10-11" },
  { "SHADERTOY IN", "GLSL ES 3 through the same front end: a real preprocessor, texture(), switch, discard",
    "oksl README · GLSL (Shadertoy)" },
  { "FFRAMES' SHADERS", string.format("all %d fframes shader examples ported · shader_mode: %d SkSL programs a frame",
    facts.FFRAMES_SHADER_EXAMPLES, facts.SHADER_MODE_PROGRAMS), "onimite PARITY.md · BENCHMARK.md" },
}

local function render(f, lb)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.BG }
  group({ opacity = 0.25 }, function()
    shader(f, { src = "rings.sksl", x = 0, y = 0, w = 1920, h = 1080, time = lb * beat.BEAT,
      uniforms = { uKick = pulse(lb, 6) } })
  end)
  for i, r in ipairs(ROWS) do
    local at = T.NUMBER_POPS[i]
    if lb >= at then
      local p = snap(lb - at)
      local y = 260 + (i - 1) * 230
      group({ opacity = prog(lb, at, at + 0.15), transform = translate((1 - p) * 80, 0) }, function()
        text { r[1], x = 120, y = y, font = ui.DISPLAY, size = 120, fill = i == 1 and ui.A or ui.TEXT }
        text { r[2], x = 124, y = y + 50, font = ui.SANS, size = 34, fill = ui.TEXT }
        text { r[3], x = 124, y = y + 88, font = ui.MONO, size = 20, fill = ui.DIM }
      end)
    end
  end
  ui.label(lb, 5, "measured")
end

return { name = "Numbers", start = T.NUMBERS, stop = T.OUTRO, render = render }
