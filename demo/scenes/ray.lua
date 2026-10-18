-- raymarch.sksl over the whole frame: a rounded box that melts into a
-- torus on bar 1's third beat and back on bar 2's, pumped by the kick.

local beat = require("beat")
local ui = require("ui")
local T = require("timeline")
local prog, pulse = beat.prog, beat.pulse

local function smooth(t) return t * t * (3 - 2 * t) end

local function render(f, lb)
  local morph = smooth(prog(lb, 2, 3.5)) - smooth(prog(lb, 6, 7.5))
  shader(f, { src = "raymarch.sksl", x = 0, y = 0, w = 1920, h = 1080, time = beat.seconds(lb),
    uniforms = { uKick = pulse(lb, 6), uMorph = morph } })
  text { "80 sphere-tracing steps a pixel · normals by central differences · mix(box, torus)", x = 960, y = 1000,
    align = "center", font = ui.MONO, size = 22, fill = ui.TEXT }
  ui.label(lb, 3, "raymarched SDF · raymarch.sksl")
end

return { name = "Raymarch", start = T.RAY, stop = T.TUNNEL, render = render }
