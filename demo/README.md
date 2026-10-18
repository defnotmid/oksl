# oksl demo

A 27 second showcase of oksl, music included: 1920x1080, 60 fps, 150 BPM.
It is an [onimite](../../onimite) script (Odin + Lua video framework,
whose shader layers are compiled by oksl), not part of oksl's build.
`demo.mp4` is the render.

```sh
cd ../../onimite
odin build . -collection:deps=.. -o:speed -out:build/onimite_speed
build/onimite_speed render ../oksl/demo/video.lua -o ../oksl/demo/demo.mp4   # re-render
build/onimite_speed preview ../oksl/demo/video.lua                          # play, hot reload on save
```

| bars | scene | what |
| --- | --- | --- |
| 1-2 | Title | "OKSL" drawn into a text group that `glitch.sksl` reads as its `uniform shader` child: glitch slices and colour split, a ripple ring from the slam |
| 3-6 | Source | `media/rings.sksl` as code on screen, typed in over a bar, rendering beside it; then its three outputs (SPIR-V, MSL, CPU interpreter) as labelled tiles, one a beat, and what okslc makes of it |
| 7-8 | Ripple | `glitch.sksl` over a text group child: a ripple ring and a glitch burst on the hits |
| 9-10 | Raymarch | `raymarch.sksl`: a sphere-traced rounded box melting into a torus and back |
| 11-12 | Shadertoy | a tunnel written as Shadertoy GLSL, passed as `shadertoy = [[...]]` and compiled as GLSL |
| 13-14 | Numbers | GPU = CPU, the GLSL front end, fframes' shader examples |
| 15-17 | Outro | the wordmark over the soundtrack's spectrum |

The three output tiles are drawn by the backend that rendered the video
(SDL_GPU runs the SPIR-V, `--software` the CPU program); they label what
the one source compiles to, and the parity between them is the measured
fact in the Numbers scene.

## Facts on screen

Every number is in `facts.lua` with its source:

| on screen | source |
| --- | --- |
| okslc rings.sksl: 894 SPIR-V words, 4566 MSL bytes | `okslc media/rings.sksl` (oksl's `okslc`, 2026-10-11) |
| GPU = CPU: 21 / 21 test shaders match, 19 within 1/255 | `odin run tools/gpu_parity -- out_dir` in oksl, 2026-10-11, SDL_GPU on Vulkan (RX 9070 XT): every testdata shader "ok", max difference 1/255 or less on 19; the other two (defines, noise_fbm) differ only at threshold edges, as the README's Limitations describe |
| Shadertoy in: GLSL ES 3 through the same front end (a real preprocessor, `texture()`, `switch`, `discard`) | oksl README, GLSL (Shadertoy) |
| all 5 fframes shader examples ported; shader_mode: 18 SkSL programs a frame | onimite `PARITY.md` (shaders, neon-triangle, shader-mode, fframes-intro, made-of-motion all ✅; the ports themselves are kept out of the repository), `BENCHMARK.md` (shader_mode footnote) |
| no shader compiler to install, depends on `core:` only | oksl README, introduction |

## Files

`video.lua` sets up the video, the beat-synced glitch cuts and the HUD;
`scenes/` holds one file per scene; `media/*.sksl` are the shaders (the
Shadertoy tunnel is inline in `scenes/tunnel.lua`); `track.lua` is the
soundtrack, an odawg `song{}` (an F# minor breakbeat, its sections named
after the scenes, its parts the pops and stutters the picture answers
to); `timeline.lua` reads from it on which beat everything happens;
`beat.lua` is the clock and easing on the song's beat.

Mix (`onimite audio analyze`): -14.1 LUFS integrated, -1.5 dBTP true
peak. Render: 1632 frames in 3.8 s wall, 165 MB peak RSS.

## Media

- Shaders: written for this demo (`glitch.sksl` after onimite's
  `examples/why_onimite/scenes/title.lua`).
- Fonts, copied from onimite's examples: Bebas Neue, Space Grotesk and
  JetBrains Mono, under the SIL Open Font License 1.1
  ([media/OFL.txt](media/OFL.txt), with their copyright notices).
- The soundtrack is composed by odawg from `track.lua`; there are no other media.
