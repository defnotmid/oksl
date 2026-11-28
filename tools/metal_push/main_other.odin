#+build !darwin
package metal_push

import "core:fmt"

main :: proc() {
	fmt.println("metal_push requires macOS")
}
