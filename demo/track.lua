-- The soundtrack: an odawg song, electro at 128 BPM in F# phrygian, 17
-- bars, its sections named after the scenes (the title has two: the intro
-- bar and the slam). Its own parts carry what the picture answers to: the
-- three outputs' pops, the ripple's stutters, the numbers' pops, a short
-- boom on the slam, the tunnel and the outro, a cymbal on the scenes'
-- downbeats. timeline.lua reads them back.
--
-- Why: a shader compiler's demo is the demoscene's home ground, and its
-- music is machines, so electro after Drexciya and Kraftwerk (the Numbers
-- scene is a nod): the 808 (machines/808.json, copied from onimite's
-- library) on a broken electro beat (the kick on one, the and of two and
-- the a of three, the clap on two and four, sixteenth hats with accents, a
-- cowbell and congas answering), a square bass sequence pushing the
-- phrygian half step (F# to G), a robot arpeggio of our own (a narrow pulse
-- through a resonant low-pass a short envelope plucks), a string machine
-- (saws through a chorus) under the raymarcher, and a lead in the tunnel and
-- the outro. F# phrygian: F#m, its G above (the dark half step), A, Bm, D
-- and Em around it. No risers.

local LIB = "" -- rows copied into media/ from onimite's library: machines/808.json, instruments/808_bass.json and sfx.json
local KIT = { LIB .. "machines/808.json", tone = 0.55 }

-- the electro beat, two bars
local BEAT = {
  kick = { "x.....x...x.....", "x.....x...x..x.." },
  clap = "....x.......x...",
  hat = "x.xXx.xXx.xXx.xX",
  cowbell = { "", "..........x....." },
  conga_high = { "......x.......x.", "...x......x.x..." },
}

return song { "score", key = "F# phrygian", bpm = 128, seed = 128,
  instruments = { LIB .. "instruments/sfx.json", {
    -- the robot: a narrow pulse through a resonant low-pass a short
    -- envelope plucks open, a little drive
    robot = {
      nodes = {
        pulse = { "Osc", wave = "pulse", width = 0.22 },
        pluck = { "Decay", time = 0.09 },
        lp = { "Filter", "pulse", type = "lowpass", rolloff = -24, q = 3.2,
               frequency = { { 420, by = "pluck", octaves = 2.6 }, { by = "velocity", octaves = 0.8 } } },
        drive = { "Drive", "lp", amount = 1.3 },
        env = { "Env", attack = 0.002, decay = 0.18, sustain = 0.2, release = 0.06 },
        out = { "Mix", "drive", amp = "env", gain_db = -12 },
      },
      out = "out",
      options = { cutoff = "lp/frequency" },
    },
    -- the bass: a square and a saw an octave down, one voice, a ladder that
    -- barks on each note
    sequence = {
      mono = true, portamento = 0.02,
      nodes = {
        sq = { "Osc", wave = "square" },
        saw = { "Osc", wave = "sawtooth", ratio = 0.5, gain_db = -6 },
        both = { "Mix", "sq", "saw" },
        bark = { "Decay", time = 0.12 },
        lp = { "Ladder", "both", q = 2.2, frequency = { { 220, by = "bark", octaves = 2.2 }, { by = "velocity", octaves = 0.8 } } },
        env = { "Env", attack = 0.002, decay = 0.2, sustain = 0.7, release = 0.05 },
        out = { "Mix", "lp", amp = "env", gain_db = -12 },
      },
      out = "out",
    },
    -- the string machine: six saws spread wide, a slow swell, a gentle low-pass
    strings = {
      nodes = {
        saws = { "Osc", wave = "fatsawtooth", count = 6, spread = 24, width = 0.9, phase = "random" },
        lp = { "Filter", "saws", type = "lowpass", frequency = 2200, q = 0.7 },
        env = { "Env", attack = 0.6, decay = 0.4, sustain = 0.85, release = 1.2 },
        out = { "Mix", "lp", amp = "env", gain_db = -20 },
      },
      out = "out",
    },
    -- the lead: a square an octave over a saw, a short glide, vibrato that
    -- fades in
    lead = {
      mono = true, portamento = 0.04,
      nodes = {
        vib = { "Lfo", rate = 5.5, delay = 0.2, fade = 0.3 },
        sq = { "Osc", wave = "square", frequency = { by = "vib", octaves = 0.006 } },
        saw = { "Osc", wave = "sawtooth", ratio = 0.5, gain_db = -8, frequency = { by = "vib", octaves = 0.006 } },
        both = { "Mix", "sq", "saw" },
        lp = { "Filter", "both", type = "lowpass", rolloff = -24, frequency = 2600, q = 1.2 },
        env = { "Env", attack = 0.005, decay = 0.3, sustain = 0.75, release = 0.12 },
        out = { "Mix", "lp", amp = "env", gain_db = -14 },
      },
      out = "out",
    },
  } },
  sections = {
    { "title", duration = "1m", chords = "F#m" },
    { "slam", duration = "1m", chords = "G" },
    { "source", duration = "4m", chords = "F#m G Em F#m" },
    { "ripple", duration = "2m", chords = "Bm G" },
    { "ray", duration = "2m", chords = "D Em" },
    { "tunnel", duration = "2m", chords = "F#m G" },
    { "numbers", duration = "2m", chords = "Em F#m" },
    { "outro", duration = "2m", chords = "F#m G" },
    { "tail", duration = "1m", chords = "F#m" },
  },
  returns = { room = { { "Reverb", decay = 0.9, pre_delay = 0.005 }, { "Filter", type = "highpass", frequency = 250 } },
              echo = { { "PingPongDelay", delay_time = "8n.", feedback = 0.35, wet = 1 }, { "Filter", type = "lowpass", frequency = 3200 } } },
  parts = {
    -- the 808: hats alone in the title and the raymarcher, the beat from
    -- the slam, congas answering
    drums = { role = "drums", instrument = KIT, gain_db = 0, sends = { room = -18 }, play = {
      { "title", steps = { hat = "x.xXx.xXx.xXx.xX" }, velocity = 0.8 },
      { "slam", steps = { kick = "x.....x...x.....", hat = "x.xXx.xXx.xXx.xX" }, humanize = 0.003 },
      { "source..ripple", steps = BEAT, humanize = 0.003 },
      { "ray", steps = { hat = "x.x.x.x.x.x.x.x.", conga_high = "......x.......x." }, velocity = 0.7, humanize = 0.003 },
      { "tunnel..outro", steps = BEAT, humanize = 0.003 },
      { "tail", steps = { kick = "X..............." } },
    } },
    -- the bass sequence: the root, its octave, the half step above
    bass = { role = "bass", instrument = "sequence", gain_db = 6, sidechain = { "drums/kick", depth = 0.3 }, play = {
      { "slam..ripple", degrees = "1 . . 1 . . 8 . 1 . 2 . . 1 . 7", subdivision = "16n", octave = 2, velocity = 0.85, humanize = 0.002 },
      { "tunnel..outro", degrees = "1 . . 1 . . 8 . 1 . 2 . . 1 . 7", subdivision = "16n", octave = 2, velocity = 0.9, humanize = 0.002 },
      { "tail", degrees = "1", subdivision = "1m", octave = 2, velocity = 0.7 },
    } },
    -- the robot: up and down the chord in sixteenths, through the echo
    robot = { role = "counter", instrument = "robot", gain_db = 0, pan = 0.2, sends = { echo = -14 }, play = {
      { "source..ripple", arpeggio = { pattern = "upDown", interval = "16n", octaves = 2 }, octave = 4, velocity = 0.7 },
      { "ray", arpeggio = { pattern = "up", interval = "16n", octaves = 2 }, octave = 4, velocity = 0.7 },
      { "numbers", arpeggio = { pattern = "upDown", interval = "16n", octaves = 2 }, octave = 4, velocity = 0.75 },
    } },
    -- the string machine under the raymarcher and the outro
    strings = { role = "pad", instrument = "strings", gain_db = 0, sends = { room = -10 },
      effects = { { "Chorus", frequency = 0.5, delay_time = 0.008, depth = 0.6, wet = 0.45 } }, play = {
      { "ray", chords = { voicing = "spread", rhythm = "x..............." }, octave = 3, duration = "1m", velocity = 0.6 },
      { "outro..tail", chords = { voicing = "spread", rhythm = "x..............." }, octave = 3, duration = "1m", velocity = 0.5 },
    } },
    -- the lead: a phrygian call in the tunnel, answered in the outro
    lead = { role = "melody", instrument = "lead", gain_db = 0, sidechain = { "drums/kick", depth = 0.2 }, sends = { echo = -16, room = -14 }, play = {
      { "tunnel", subdivision = "8n", velocity = 0.85,
        notes = "F#5 - G5 F#5 - C#5 - - | A5 - G5 - F#5 - E5 F#5" },
      { "outro", subdivision = "8n", velocity = 0.85,
        notes = "F#5 - G5 F#5 - C#6 - - | B5 - A5 G5 - F#5 - -" },
    } },
    -- what the picture answers to: a short 808 boom on the slam, the tunnel
    -- and the outro (the picture flashes on it); the 808's cymbal on the
    -- scenes' downbeats and the last kick
    fx = { role = "fx", instrument = { LIB .. "instruments/808_bass.json", decay = 0.9 }, gain_db = 0, sends = { room = -14 }, play = {
      { "slam", events = { { time = 0, duration = "2n", note = "F#1", velocity = 0.9 } } },
      { "tunnel", events = { { time = 0, duration = "2n", note = "F#1", velocity = 0.9 } } },
      { "outro", events = { { time = 0, duration = "2n", note = "F#1", velocity = 0.9 } } },
    } },
    crash = { role = "fx", instrument = KIT, gain_db = 6, play = {
      { "source", steps = { crash = { "X...............", "", "", "" } }, velocity = 0.5 },
      { "ripple..ray", steps = { crash = { "X...............", "" } }, velocity = 0.5 },
      { "numbers", steps = { crash = { "X...............", "" } }, velocity = 0.5 },
      { "outro", steps = { crash = { "", "X..............." } }, velocity = 0.5 },
      { "tail", steps = { crash = "X..............." }, velocity = 0.5 },
    } },
    -- two stutters in the title; the ripple's ring and its glitch burst
    glitch = { role = "fx", instrument = "glitch", gain_db = -10,
      play = { { "title", steps = "x.......x.......", note = "1", duration = "4n" },
               { "ripple", steps = "x...............", note = "1", duration = "4n" } } },
    -- the pops: SPIR-V, MSL and CPU one a beat; the three numbers (each a
    -- blip and its fifth)
    blips = { role = "fx", instrument = "blip", gain_db = -2, octave = 5,
      play = { { "source@2m..", degrees = "1 3 5 . . . . .", subdivision = "4n" },
               { "numbers", degrees = "1 5 . . . . . . 1 5 . . . . . . | 1 5 . . . . . . . . . . . . . .",
                 subdivision = "16n", octave = 6 } } },
  },
  automation = {
    -- the strings up for the raymarcher's breakdown and under the outro's
    -- lead, the last kick and root down to the tail's level (inspect's
    -- role_balance: the strings 19 LU under the mix in the outro, the kick
    -- 2 over it and the root 1 under it on the tail)
    ["parts/strings/gain_db"] = { { "title", 0 }, { "ray", 2.5, "step" }, { "tunnel", 0, "step" }, { "outro", 5, "step" }, { "tail", 0, "step" } },
    ["parts/drums/gain_db"] = { { "title", 0 }, { "tail", -4, "step" } },
    ["parts/bass/gain_db"] = { { "title", 0 }, { "tail", -4, "step" } },
  },
}
