// The oksl mascot: a 32x32 pixel-art firefly, in the family of onimite's Pip,
// odecs' turntable, ottis' Ottis and ovg's chameleon. A firefly because oksl
// is light made by a program: its lantern is the shader. Writes an SVG whose
// parts are groups a script shows or hides by id (onimite:
// svg{ overrides = { glow_b = { visible = true } } }), so the glow can pulse:
//
//   body          outline, helmet, face, wings, legs, antennae and the lit
//                 lantern (on)
//   lantern_dim   the lantern out (off): show it over body
//   glow_a        a close halo round the lantern (on)
//   glow_b        a wider, fainter halo (off)
//   eyes_open     big eyes with two shines (on)   eyes_closed  a happy blink
//   eyes_o        round surprised eyes
//   mouth_smile   a small smile (on)              mouth_o      a little "o"
//
// Also writes the project logo as PNGs, scaled without smoothing:
//
//   odin run tools/mascot -- assets   (assets/logo.svg, logo.png 256 px, logo_32x32.png)
package mascot

import "core:fmt"
import "core:math"
import "core:os"
import "core:path/filepath"
import "core:strings"
import stbi "vendor:stb/image"

W :: 32
H :: 32

Grid :: [H][W]u8 // palette keys, '.' empty

color :: proc(c: u8) -> string {
	switch c {
	case 'O': return "#1a1630" // outline
	case 'B': return "#30335e" // body
	case 'D': return "#1f2142" // body shade
	case 'H': return "#545a9a" // body highlight
	case 'F': return "#6b72c4" // face
	case 'f': return "#4c529a" // face shade
	case 'h': return "#8f96e0" // face highlight
	case 'P': return "#ff6b3d" // helmet
	case 'p': return "#c9442a" // helmet shade
	case 'q': return "#ffa47a" // helmet highlight
	case 'E': return "#1b1220" // eyes
	case 'W': return "#ffffff" // shines
	case 'K': return "#ff9db4" // cheeks
	case 'M': return "#5a1f24" // mouth
	case 'V': return "#d7e8ff" // wings
	case 'v': return "#a5c0ea" // wing shade
	case 'L': return "#e6ff5c" // lantern
	case 'l': return "#a6d82e" // lantern bands
	case 'Y': return "#fbffd6" // lantern core
	case 'd': return "#5d6b2a" // lantern out
	case 'e': return "#46511f" // lantern out, bands
	case 'G': return "#f2ff9a" // glow near
	case 'g': return "#d9f5a0" // glow far
	}
	return "#ff00ff"
}

empty :: proc() -> (g: Grid) {
	for &row in g do for &p in row do p = '.'
	return
}

put :: proc(g: ^Grid, x, y: int, c: u8) {
	if 0 <= x && x < W && 0 <= y && y < H do g[y][x] = c
}

in_ellipse :: proc(x, y: int, cx, cy, rx, ry: f64) -> (bool, f64, f64) {
	u := (f64(x) + 0.5 - cx) / rx
	v := (f64(y) + 0.5 - cy) / ry
	return u * u + v * v <= 1, u, v
}

// Fills an ellipse in `base`, `shade` along the lower right rim and `light`
// at the upper left.
shaded :: proc(g: ^Grid, cx, cy, rx, ry: f64, base, shade, light: u8) {
	for y in 0 ..< H do for x in 0 ..< W {
		inside, u, v := in_ellipse(x, y, cx, cy, rx, ry)
		if !inside do continue
		c := base
		if u * u + v * v > 0.55 && u * 0.5 + v * 0.85 > 0.35 do c = shade
		else if u * u + v * v > 0.4 && u * 0.6 + v * 0.8 < -0.55 do c = light
		put(g, x, y, c)
	}
}

// Outlines every filled pixel's empty 4-neighbours.
outline :: proc(g: ^Grid) {
	src := g^
	for y in 0 ..< H do for x in 0 ..< W {
		if src[y][x] != '.' do continue
		for d in ([4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
			nx, ny := x + d.x, y + d.y
			if 0 <= nx && nx < W && 0 <= ny && ny < H && src[ny][nx] != '.' && src[ny][nx] != 'O' {
				g[y][x] = 'O'
				break
			}
		}
	}
}

// Draws `fg` over `g`, outlining fg's shapes against everything else.
over :: proc(g: ^Grid, fg: Grid) {
	shape := fg
	outline(&shape)
	for y in 0 ..< H do for x in 0 ..< W do if shape[y][x] != '.' do g[y][x] = shape[y][x]
}

LANTERN :: [4]f64{16, 24.6, 5.2, 5.0} // cx, cy, rx, ry

// The lantern's pixels: lit (core, body, bands) or out.
lantern :: proc(g: ^Grid, lit: bool) {
	cx, cy, rx, ry := LANTERN[0], LANTERN[1], LANTERN[2], LANTERN[3]
	for y in 0 ..< H do for x in 0 ..< W {
		inside, u, v := in_ellipse(x, y, cx, cy, rx, ry)
		if !inside do continue
		band := y == 22 || y == 25
		c: u8
		if lit do c = 'l' if band else ('Y' if u * u + v * v < 0.2 else 'L')
		else do c = 'e' if band else 'd'
		put(g, x, y, c)
	}
}

draw_body :: proc() -> Grid {
	g := empty()

	// Wings, spread behind: pale, shaded underneath.
	wings := empty()
	shaded(&wings, 8.2, 17.6, 5.4, 3.4, 'V', 'v', 'V')
	shaded(&wings, 23.8, 17.6, 5.4, 3.4, 'V', 'v', 'V')
	over(&g, wings)

	// Legs, three a side, under the wings.
	legs := empty()
	for p in ([?][2]int{{11, 21}, {10, 22}, {11, 23}, {10, 24}, {20, 21}, {21, 22}, {20, 23}, {21, 24}}) do put(&legs, p.x, p.y, 'D')
	over(&g, legs)

	// The lantern, the thorax over its top, then the head.
	lamp := empty()
	lantern(&lamp, true)
	over(&g, lamp)

	thorax := empty()
	shaded(&thorax, 16, 18.2, 4.6, 3.2, 'B', 'D', 'H')
	over(&g, thorax)

	// Antennae curving out to glowing tips.
	ant := empty()
	for p in ([?][2]int{{13, 5}, {12, 4}, {12, 3}, {11, 2}, {18, 5}, {19, 4}, {19, 3}, {20, 2}}) do put(&ant, p.x, p.y, 'D')
	for p in ([?][2]int{{10, 1}, {21, 1}}) do put(&ant, p.x, p.y, 'L')
	over(&g, ant)

	// The head: a dark face under an orange helmet, cheeks.
	head := empty()
	shaded(&head, 16, 11.6, 6.6, 4.9, 'F', 'f', 'h')
	for y in 0 ..< H do for x in 0 ..< W {
		inside, u, v := in_ellipse(x, y, 16, 7.6, 7.0, 2.9)
		if !inside || head[y][x] == '.' && y > 9 do continue
		c := u8('P')
		if v > 0.45 do c = 'p'
		else if u < -0.2 && v < -0.2 do c = 'q'
		put(&head, x, y, c)
	}
	// spots on the helmet, as fireflies have
	for p in ([?][2]int{{14, 6}, {18, 6}}) do put(&head, p.x, p.y, 'p')
	for p in ([?][2]int{{10, 14}, {11, 14}, {20, 14}, {21, 14}}) do put(&head, p.x, p.y, 'K')
	over(&g, head)
	return g
}

draw_lantern_dim :: proc() -> Grid {
	g := empty()
	lantern(&g, false)
	return g
}

// Halos: dithered rings round the lantern, outside its outline, drawn
// without one of their own.
draw_glows :: proc(body: Grid) -> (near, far: Grid) {
	near, far = empty(), empty()
	cx, cy := LANTERN[0], LANTERN[1]
	for y in 0 ..< H do for x in 0 ..< W {
		if body[y][x] != '.' do continue
		dx := (f64(x) + 0.5 - cx) / 1.0
		dy := (f64(y) + 0.5 - cy) / 0.95
		r := math.sqrt(dx * dx + dy * dy)
		if r < 7.4 && (x + y) % 2 == 0 do put(&near, x, y, 'G')
		if r >= 6.6 && r < 9.6 && (x + y) % 2 == 1 && (x % 3 != 0) do put(&far, x, y, 'g')
	}
	return
}

draw_eyes :: proc() -> (open, closed, round: Grid) {
	open, closed, round = empty(), empty(), empty()
	for ex in ([2]int{11, 18}) {
		for y in 11 ..< 14 do for x in ex ..< ex + 3 do put(&open, x, y, 'E')
		put(&open, ex, 11, 'W')
		put(&open, ex + 2, 13, 'W')
		put(&closed, ex, 13, 'E')
		put(&closed, ex + 1, 12, 'E')
		put(&closed, ex + 2, 13, 'E')
		for y in 10 ..< 15 do for x in ex - 1 ..< ex + 4 {
			if (y == 10 || y == 14) && (x == ex - 1 || x == ex + 3) do continue
			put(&round, x, y, 'E')
		}
		put(&round, ex, 11, 'W')
		put(&round, ex + 1, 11, 'W')
		put(&round, ex, 12, 'W')
	}
	return
}

draw_mouths :: proc() -> (smile, o: Grid) {
	smile, o = empty(), empty()
	for p in ([?][2]int{{14, 15}, {15, 16}, {16, 16}, {17, 15}}) do put(&smile, p.x, p.y, 'M')
	for p in ([?][2]int{{15, 15}, {16, 15}, {15, 16}, {16, 16}}) do put(&o, p.x, p.y, 'M')
	return
}

Layer :: struct {
	name: string,
	grid: Grid,
	on:   bool,
}

svg :: proc(layers: []Layer) -> string {
	b := strings.builder_make()
	fmt.sbprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" shape-rendering="crispEdges">`, W, H, W * 8, H * 8)
	strings.write_byte(&b, '\n')
	for l in layers {
		fmt.sbprintf(&b, `<g id="%s"%s>`, l.name, "" if l.on else ` display="none"`)
		for y in 0 ..< H do for x in 0 ..< W {
			c := l.grid[y][x]
			if c == '.' do continue
			fmt.sbprintf(&b, `<rect x="%d" y="%d" width="1" height="1" fill="%s"/>`, x, y, color(c))
		}
		strings.write_string(&b, "</g>\n")
	}
	strings.write_string(&b, "</svg>\n")
	return strings.to_string(b)
}

hex_byte :: proc(s: string) -> u8 {
	v: u8
	for c in transmute([]u8)s {
		v <<= 4
		switch c {
		case '0' ..= '9': v |= c - '0'
		case 'a' ..= 'f': v |= c - 'a' + 10
		}
	}
	return v
}

// The visible layers flattened, scaled by `scale` without smoothing.
png :: proc(path: string, layers: []Layer, scale: int) -> bool {
	size := W * scale
	pixels := make([]u8, size * size * 4)
	defer delete(pixels)
	for l in layers {
		if !l.on do continue
		for y in 0 ..< H do for x in 0 ..< W {
			c := l.grid[y][x]
			if c == '.' do continue
			hex := color(c)
			rgba := [4]u8{hex_byte(hex[1:3]), hex_byte(hex[3:5]), hex_byte(hex[5:7]), 255}
			for dy in 0 ..< scale do for dx in 0 ..< scale {
				i := ((y * scale + dy) * size + x * scale + dx) * 4
				pixels[i + 0], pixels[i + 1], pixels[i + 2], pixels[i + 3] = rgba[0], rgba[1], rgba[2], rgba[3]
			}
		}
	}
	return stbi.write_png(strings.clone_to_cstring(path, context.temp_allocator), i32(size), i32(size), 4, raw_data(pixels), i32(size * 4)) != 0
}

main :: proc() {
	dir := os.args[1] if len(os.args) > 1 else "assets"
	body := draw_body()
	glow_a, glow_b := draw_glows(body)
	eyes_open, eyes_closed, eyes_o := draw_eyes()
	mouth_smile, mouth_o := draw_mouths()
	layers := []Layer {
		{"glow_b", glow_b, false},
		{"glow_a", glow_a, true},
		{"body", body, true},
		{"lantern_dim", draw_lantern_dim(), false},
		{"eyes_open", eyes_open, true},
		{"eyes_closed", eyes_closed, false},
		{"eyes_o", eyes_o, false},
		{"mouth_smile", mouth_smile, true},
		{"mouth_o", mouth_o, false},
	}
	join :: proc(dir, name: string) -> string {
		p, _ := filepath.join({dir, name}, context.temp_allocator)
		return p
	}
	if err := os.write_entire_file(join(dir, "logo.svg"), svg(layers)); err != nil {
		fmt.eprintfln("writing %s: %v", join(dir, "logo.svg"), err)
		os.exit(1)
	}
	if !png(join(dir, "logo.png"), layers, 8) || !png(join(dir, "logo_32x32.png"), layers, 1) {
		fmt.eprintln("writing the PNGs failed")
		os.exit(1)
	}
}
