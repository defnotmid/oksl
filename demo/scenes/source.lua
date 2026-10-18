-- media/rings.sksl as code on screen, typed in over the first bar, and
-- the same file rendering beside it; then its three outputs pop in as
-- labelled tiles, one a beat.

local beat = require("beat")
local ui = require("ui")
local T = require("timeline")
local facts = require("facts")
local prog, snap, soft, pulse = beat.prog, beat.snap, beat.soft, beat.pulse

-- The source, read once at load and split into coloured spans.
local KEYWORDS = { uniform = ui.B, half4 = ui.C, half3 = ui.C, float = ui.C, float2 = ui.C, float3 = ui.C,
  ["return"] = ui.B }
local BUILTINS = { length = ui.A, atan = ui.A, sin = ui.A, cos = ui.A, abs = ui.A, smoothstep = ui.A, main = ui.A }
local LINES = {}
do
  local fh = assert(io.open(SCRIPT_DIR .. "/media/rings.sksl", "r"))
  for line in fh:lines() do
    local spans = {}
    if line:match("^%s*//") then
      spans[1] = { line, fill = ui.DIM }
    else
      local pos = 1
      for s, word, e in line:gmatch("()([%a_][%w_]*)()") do
        if s > pos then spans[#spans + 1] = { line:sub(pos, s - 1), fill = ui.TEXT } end
        spans[#spans + 1] = { word, fill = KEYWORDS[word] or BUILTINS[word] or ui.TEXT }
        pos = e
      end
      if pos <= #line then spans[#spans + 1] = { line:sub(pos), fill = ui.TEXT } end
    end
    LINES[#LINES + 1] = spans
  end
  fh:close()
end

local CODE = { x = 90, y = 230, size = 22, lh = 40 }

local OUTPUTS = {
  { "SPIR-V", "Vulkan, through SDL_GPU", ui.A },
  { "MSL", "Metal, through SDL_GPU on macOS", ui.B },
  { "CPU interpreter", "the software backend", ui.C },
}

local function rings(f, lb, x, y, w, h)
  shader(f, { src = "rings.sksl", x = x, y = y, w = w, h = h, time = beat.seconds(lb),
    uniforms = { uKick = pulse(lb, 6) } })
end

local function render(f, lb)
  rect { x = 0, y = 0, w = 1920, h = 1080, fill = ui.BG }
  -- the code panel
  rect { x = CODE.x - 30, y = CODE.y - 50, w = 1000, h = #LINES * CODE.lh + 70, rx = 20, fill = ui.PANEL,
    stroke = ui.LINE, stroke_width = 2 }
  local shown = math.floor(prog(lb, 0, T.CODE_TYPED) * #LINES + 0.999)
  for i = 1, shown do
    if #LINES[i] > 0 then
      text { spans = LINES[i], x = CODE.x, y = CODE.y + (i - 1) * CODE.lh, font = ui.MONO, size = CODE.size }
    end
  end
  if lb < T.CODE_TYPED + 2 and pulse(lb * 2, 2) > 0.3 then
    rect { x = CODE.x, y = CODE.y + math.max(shown - 1, 0) * CODE.lh + 10, w = 14, h = 4, fill = ui.A }
  end

  -- the effect: one big box, then the three outputs
  local split = T.OUTPUTS[1]
  if lb < split then
    local s = snap(lb - 1)
    if lb >= 1 then
      group({ transform = { scale = 0.8 + 0.2 * s, origin = { 1450, 540 } }, opacity = prog(lb, 1, 1.2) }, function()
        rings(f, lb, 1110, 190, 700, 700)
      end)
      text { "shader{ src = \"rings.sksl\" }", x = 1460, y = 950, align = "center", font = ui.MONO, size = 24,
        fill = ui.DIM }
    end
  else
    for i, o in ipairs(OUTPUTS) do
      local at = T.OUTPUTS[i]
      if lb >= at then
        local p = snap(lb - at)
        local y = 150 + (i - 1) * 250
        group({ transform = { translate = { (1 - p) * -200, 0 } }, opacity = prog(lb, at, at + 0.15) }, function()
          rings(f, lb, 1110, y, 380, 214)
          text { o[1], x = 1520, y = y + 90, font = ui.DISPLAY, size = 64, fill = o[3] }
          text { o[2], x = 1522, y = y + 130, font = ui.MONO, size = 18, fill = ui.TEXT }
        end)
      end
    end
  end
  ui.label(lb, 1, "one SkSL source · three backends")
  if lb >= T.OUTPUTS[3] + 1 then
    local b = T.OUTPUTS[3] + 1
    local q = snap(lb - b)
    group({ opacity = prog(lb, b, b + 0.2), transform = translate(0, (1 - q) * 30) }, function()
      text { string.format("okslc rings.sksl: %d SPIR-V words · %d MSL bytes · one flat CPU program",
        facts.RINGS_SPIRV_WORDS, facts.RINGS_MSL_BYTES), x = 90, y = 960, font = ui.SANS, size = 36, fill = ui.TEXT }
      text { "the tiles here are drawn by this render's backend; the GPU picks the SPIR-V or MSL, --software runs the CPU program",
        x = 90, y = 1004, font = ui.MONO, size = 18, fill = ui.DIM }
    end)
  end
end

return { name = "Source", start = T.SOURCE, stop = T.RIPPLE, render = render }
