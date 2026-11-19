package oksl

import "base:runtime"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:sync"

JIT_DUMP :: JIT_ENABLED && #config(OKSL_JIT_DUMP, false)

when !JIT_DUMP {

_ :: runtime
_ :: fmt
_ :: os
_ :: strings
_ :: sync

when JIT_ENABLED {
	@(private)
	jit_dump_program :: #force_inline proc(_: ^Program, _: ^Jit_Emitter, _: []u8, _, _, _: int) {}
}

} else {

@(private = "file")
jit_dump_seq: int

// Dumps generated machine code (AVX2 or NEON) and basic-block / VM-instruction boundaries
// when built with `-define:OKSL_JIT_DUMP=true`. Writes:
//   - `/tmp/perf-<pid>.map` symbol entries for `perf record` / `perf script`
//   - `<dir>/jit_<pid>_<id>_body.bin` (and `_init.bin` when present)
//   - `<dir>/jit_<pid>_<id>_body.map` mapping machine-code offsets to BBs, PCs and stubs
// where `<dir>` is `$OKSL_JIT_DUMP_DIR` (default `/tmp/oksl_jit`).
@(private)
jit_dump_program :: proc(p: ^Program, e: ^Jit_Emitter, mem_slice: []u8, init_bytes, body_off, total: int) {
	heap := runtime.default_allocator()
	id := sync.atomic_add(&jit_dump_seq, 1)
	pid := os.get_pid()
	base_va := uintptr(raw_data(mem_slice))

	perf_path := fmt.aprintf("/tmp/perf-%d.map", pid, allocator = heap)
	defer delete(perf_path, heap)
	if pf, err := os.open(perf_path, {.Write, .Create, .Append}); err == nil {
		defer os.close(pf)
		if init_bytes > 0 {
			line := fmt.aprintf("%x %x oksl_jit_%d_init\n", uint(base_va), uint(init_bytes), id, allocator = heap)
			_, _ = os.write_string(pf, line)
			delete(line, heap)
		}
		body_len := total - body_off
		if body_len > 0 {
			line := fmt.aprintf("%x %x oksl_jit_%d_body\n", uint(base_va) + uint(body_off), uint(body_len), id, allocator = heap)
			_, _ = os.write_string(pf, line)
			delete(line, heap)
		}
	}

	dir_buf: [512]u8
	dir := os.get_env(dir_buf[:], "OKSL_JIT_DUMP_DIR")
	dump_dir := dir if len(dir) > 0 else "/tmp/oksl_jit"
	_ = os.make_directory(dump_dir)

	if init_bytes > 0 {
		init_path := fmt.aprintf("%s/jit_%d_%d_init.bin", dump_dir, pid, id, allocator = heap)
		_ = os.write_entire_file(init_path, mem_slice[:init_bytes])
		delete(init_path, heap)
	}

	body_bytes := mem_slice[body_off:total]
	bin_path := fmt.aprintf("%s/jit_%d_%d_body.bin", dump_dir, pid, id, allocator = heap)
	_ = os.write_entire_file(bin_path, body_bytes)
	delete(bin_path, heap)

	code := p.code[p.init_len:]
	n_code := len(code)
	mem_write_ops := 0
	for mm in e.mem_mask[:n_code] {
		if mm != 0 {
			mem_write_ops += 1
		}
	}

	sb := strings.builder_make(0, max(n_code * 64, 1024), heap)
	defer strings.builder_destroy(&sb)

	fmt.sbprintfln(&sb, "META id=%d pid=%d stage=%v base_va=0x%x body_bytes=%d init_bytes=%d num_slots=%d mem_write_ops=%d init_instrs=%d body_instrs=%d",
		id, pid, p.stage, uint(base_va) + uint(body_off), len(body_bytes), init_bytes, p.num_slots, mem_write_ops, p.init_len, n_code)

	if n_code > 0 {
		pro_end := e.pc_off[0] - body_off
		fmt.sbprintfln(&sb, "SECTION PROLOGUE 0x%x 0x%x", 0, pro_end)
	}

	bb_id := -1
	for pc in 0 ..< n_code {
		off0 := e.pc_off[pc] - body_off
		off1 := e.pc_off[pc + 1] - body_off
		if e.bb_start[pc] {
			bb_id += 1
			end_pc := pc + 1
			for end_pc < n_code && !e.bb_start[end_pc] {
				end_pc += 1
			}
			bb_off1 := e.pc_off[end_pc] - body_off
			fmt.sbprintfln(&sb, "BB %d pc=%d..%d off=0x%x..0x%x bytes=%d", bb_id, pc, end_pc, off0, bb_off1, bb_off1 - off0)
		}
		in_ := code[pc]
		fmt.sbprintfln(&sb, "INS pc=%d bb=%d off=0x%x..0x%x bytes=%d op=%v n=%d k=%d dst=%d a=%d b=%d c=%d d=%d",
			pc, bb_id, off0, off1, off1 - off0, in_.op, in_.n, in_.k, in_.dst, in_.a, in_.b, in_.c, in_.d)
	}

	ep_off := e.epilogue_off - body_off
	ep_end := ep_off + (12 when JIT_X64 else 36)
	fmt.sbprintfln(&sb, "SECTION EPILOGUE 0x%x 0x%x", ep_off, ep_end)

	when JIT_X64 {
		for stub, si in e.bound_tex {
			s0 := stub.stub_off - body_off
			s1 := stub.stub_end - body_off
			in_ := code[stub.pc]
			fmt.sbprintfln(&sb, "STUB idx=%d pc=%d op=%v k=%d off=0x%x..0x%x bytes=%d", si, stub.pc, in_.op, in_.k, s0, s1, s1 - s0)
		}
		for stub, si in e.load_bufs {
			s0 := stub.stub_off - body_off
			s1 := stub.stub_end - body_off
			in_ := code[stub.pc]
			fmt.sbprintfln(&sb, "STUB idx=%d pc=%d op=%v k=%d off=0x%x..0x%x bytes=%d", len(e.bound_tex) + si, stub.pc, in_.op, in_.k, s0, s1, s1 - s0)
		}
	} else {
		for stub, si in e.unbound_tex {
			s0 := stub.stub_off - body_off
			s1 := stub.stub_end - body_off
			in_ := code[stub.pc]
			fmt.sbprintfln(&sb, "STUB idx=%d pc=%d op=%v k=%d off=0x%x..0x%x bytes=%d", si, stub.pc, in_.op, in_.k, s0, s1, s1 - s0)
		}
	}

	subs: [8]struct {
		name: string,
		off:  int,
	}
	n_subs := 0
	if e.chan_sample_off > 0 {
		subs[n_subs] = {"CHAN_SAMPLE_SUB", e.chan_sample_off - body_off}
		n_subs += 1
	}
	if e.chan_bilinear_off > 0 {
		subs[n_subs] = {"CHAN_BILINEAR_SUB", e.chan_bilinear_off - body_off}
		n_subs += 1
	}
	if e.chan_fetch_off > 0 {
		subs[n_subs] = {"CHAN_FETCH_SUB", e.chan_fetch_off - body_off}
		n_subs += 1
	}
	if e.chan_decode_off > 0 {
		subs[n_subs] = {"CHAN_DECODE_SUB", e.chan_decode_off - body_off}
		n_subs += 1
	}
	when JIT_X64 {
		if e.helper_ymm_off > 0 {
			subs[n_subs] = {"HELPER_YMM_SUB", e.helper_ymm_off - body_off}
			n_subs += 1
		}
		if e.unif_oob_off > 0 {
			subs[n_subs] = {"UNIF_OOB_SUB", e.unif_oob_off - body_off}
			n_subs += 1
		}
		if e.big_buf_off > 0 {
			subs[n_subs] = {"BIG_BUF_SUB", e.big_buf_off - body_off}
			n_subs += 1
		}
	}
	for si in 0 ..< n_subs {
		s0 := subs[si].off
		s1 := subs[si + 1].off if si + 1 < n_subs else len(body_bytes)
		fmt.sbprintfln(&sb, "SECTION %s 0x%x 0x%x", subs[si].name, s0, s1)
	}

	map_path := fmt.aprintf("%s/jit_%d_%d_body.map", dump_dir, pid, id, allocator = heap)
	_ = os.write_entire_file(map_path, transmute([]u8)strings.to_string(sb))
	delete(map_path, heap)
}

}
