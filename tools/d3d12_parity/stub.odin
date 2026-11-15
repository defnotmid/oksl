#+build !linux, !amd64
#+build !windows, !amd64
// The harness runs Direct3D 12 through vkd3d (libvkd3d-utils) on x86-64
// Linux, whose entry points are ms_abi, or Microsoft's runtime on x86-64
// Windows.
package d3d12_parity

import "core:fmt"
import "core:os"

main :: proc() {
	fmt.eprintln("d3d12_parity runs on x86-64 Linux (vkd3d) or x86-64 Windows only")
	os.exit(2)
}
