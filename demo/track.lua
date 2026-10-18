-- The soundtrack: an odawg song, 150 BPM breakbeat in F# minor (F#m D A E,
-- a chord a bar), 17 bars, its sections named after the scenes (the title
-- has two: the intro bar and the slam). House's bass, stabs, pad and pluck
-- arpeggio over a broken beat (the kick on 1, the "a" of 2 and the "and"
-- of 3, claps on 2 and 4, sixteenth hats); a breakdown under the
-- raymarcher, an ambient tail. The transitions make the risers and the
-- impacts on the slam, the tunnel and the outro; the parts below add what
-- the picture answers to: the three outputs' pops, the ripple's stutters,
-- the numbers' pops, a crash on each scene's downbeat. timeline.lua reads
-- them back.

local MACHINE = { "DrumMachine", machine = "909" }
local BREAK = { kick = "x......x..x.....", clap = "....x.......x...", hat = "xxXxxxXxxxXxxxXx",
  open_hat = "..x...x...x...x." }

return song { "score", key = "F# minor", bpm = 150, seed = 150,
  sections = {
    { "title", duration = "1m", style = "house", energy = 0.3, chords = "F#m" },
    { "slam", duration = "1m", energy = 0.5, chords = "D", transition = "impact" },
    { "source", duration = "4m", energy = 0.8, chords = "A E F#m D", transition = "none" },
    { "ripple", duration = "2m", energy = 0.95, chords = "A E", transition = "none" },
    { "ray", duration = "2m", energy = 0.35, chords = "F#m D", transition = "none" },
    { "tunnel", duration = "2m", energy = 0.95, chords = "A E", transition = "impact" },
    { "numbers", duration = "2m", energy = 0.9, chords = "F#m D", transition = "none" },
    { "outro", duration = "2m", energy = 1, chords = "A E", transition = "impact" },
    { "tail", duration = "1m", style = "ambient", energy = 0.3, chords = "F#m", transition = "cut" },
  },
  parts = {
    -- the broken beat: hats alone in the title and the breakdown
    drums = { gain_db = 1,
      play = { { "title", steps = { hat = "xxXxxxXxxxXxxxXx" } },
               { "slam", steps = { kick = BREAK.kick, hat = BREAK.hat } },
               { "source..ripple", steps = BREAK },
               { "ray", steps = { hat = "x.x.x.x.x.x.x.x." }, velocity = 0.7 },
               { "tunnel..outro", steps = BREAK },
               { "tail", steps = { kick = "X..............." } } } },
    pad = { gain_db = -16 },
    keys = { mute = { "numbers" } },                              -- the numbers groove without the stabs
    arp = { play = { { "ray", arpeggio = { pattern = "upDown", interval = "16n", octaves = 2 }, velocity = 0.8 } } },
    -- a crash on the source's, the ripple's, the raymarcher's and the
    -- numbers' downbeats, on the wordmark and on the last kick
    crash = { instrument = MACHINE, gain_db = -2,
      play = { { "source", steps = { crash = { "X...............", "", "", "" } } },
               { "ripple..ray", steps = { crash = { "X...............", "" } } },
               { "numbers", steps = { crash = { "X...............", "" } } },
               { "outro", steps = { crash = { "", "X..............." } } },
               { "tail", steps = { crash = "X..............." } } } },
    -- two stutters in the title; the ripple's ring and its glitch burst
    glitch = { instrument = "glitch", gain_db = -10,
      play = { { "title", steps = "x.......x.......", note = "1", duration = "4n" },
               { "ripple", steps = "x...............", note = "1", duration = "4n" } } },
    -- the pops: SPIR-V, MSL and CPU one a beat; the three numbers (each a
    -- blip and its fifth)
    blips = { instrument = "blip", gain_db = -2, octave = 5,
      play = { { "source@2m..", degrees = "1 3 5 . . . . .", subdivision = "4n" },
               { "numbers", degrees = "1 5 . . . . . . 1 5 . . . . . . | 1 5 . . . . . . . . . . . . . .",
                 subdivision = "16n", octave = 6 } } },
  },
}
