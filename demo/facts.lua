-- Every number and claim on screen, with where it comes from.

local M = {}

-- `okslc media/rings.sksl` (oksl/okslc, 2026-10-11): what the shader on
-- screen compiles to.
M.RINGS_SPIRV_WORDS, M.RINGS_MSL_BYTES = 894, 4566

-- `odin run tools/gpu_parity -- out_dir` in oksl (2026-10-11, SDL_GPU on
-- Vulkan, RX 9070 XT): oksl's testdata rendered on the GPU and by the CPU
-- interpreter; "21/21 shaders match", 19 of them with a max difference of
-- at most 1/255 (the other two, defines and noise_fbm, differ only at
-- threshold edges, which the oksl README's Limitations describe).
M.PARITY_MATCH, M.PARITY_TOTAL, M.PARITY_WITHIN1 = 21, 21, 19
M.PARITY_SRC = "oksl tools/gpu_parity · SDL_GPU on Vulkan (RX 9070 XT) vs the CPU interpreter"

-- onimite PARITY.md: all 17 fframes examples ported, the five with
-- shaders among them (shaders, neon-triangle, shader-mode, fframes-intro,
-- made-of-motion); BENCHMARK.md: shader_mode draws 18 SkSL programs (one
-- Shadertoy) on every frame.
M.FFRAMES_SHADER_EXAMPLES = 5
M.SHADER_MODE_PROGRAMS = 18

return M
