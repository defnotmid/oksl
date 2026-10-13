-- A Shadertoy-style tunnel passed as `shadertoy = [[...]]`: oksl compiles
-- it as GLSL through the same front end as SkSL (its preprocessor expands
-- the defines, the i* inputs are declared, bottom-left origin, opaque
-- output). Written for this demo.

local beat = require("beat")
local ui = require("ui")
local T = require("timeline")
local prog, snap, soft, pulse = beat.prog, beat.snap, beat.soft, beat.pulse

local TUNNEL = [[
#define SPEED 1.6
#define TWIST 0.35

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    float r = length(uv);
    float a = atan(uv.y, uv.x) / 3.14159265;
    float z = 1.0 / max(r, 0.001) + iTime * SPEED;
    float u = a + TWIST * z;
    float grid = step(0.5, fract(u * 4.0)) * 0.6 + step(0.5, fract(z * 2.0)) * 0.4;
    vec3 col = 0.5 + 0.5 * cos(6.28318 * (vec3(0.0, 0.33, 0.67) + z * 0.15));
    col *= 0.35 + 0.65 * grid;
    col *= smoothstep(0.0, 0.6, r);
    col += vec3(1.0, 0.8, 0.4) * 0.03 / max(r, 0.02);
    fragColor = vec4(col, 1.0);
}
]]

local function render(f, lb)
  local s = 1 + 0.04 * pulse(lb, 7) + 0.3 * (1 - soft(lb))
  group({ transform = { scale = s, origin = { 960, 540 } } }, function()
    shader(f, { shadertoy = TUNNEL, x = 0, y = 0, w = 1920, h = 1080, time = lb * beat.BEAT })
  end)
  local p = snap(lb - 1)
  group({ opacity = prog(lb, 1, 1.2), transform = translate(0, (1 - p) * 30) }, function()
    rect { x = 560, y = 900, w = 800, h = 120, rx = 24, fill = ui.BG, fill_opacity = 0.75 }
    text { "void mainImage(out vec4 fragColor, in vec2 fragCoord)", x = 960, y = 950, align = "center",
      font = ui.MONO, size = 24, fill = ui.A }
    text { "Shadertoy GLSL in, compiled as GLSL: no rewrite to SkSL", x = 960, y = 992, align = "center",
      font = ui.MONO, size = 20, fill = ui.TEXT }
  end)
  ui.label(lb, 4, "shadertoy = [[ … ]]")
end

return { name = "Shadertoy", start = T.TUNNEL, stop = T.NUMBERS, render = render }
