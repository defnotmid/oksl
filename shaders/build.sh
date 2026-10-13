#!/bin/sh
# Compiles the layer vertex shader to the committed SPIR-V and MSL blobs.
# Only needed after editing layer.slang; users never need slangc.
set -e
cd "$(dirname "$0")"
slangc layer.slang -matrix-layout-column-major -O2 -entry vs_main -stage vertex -target spirv -profile spirv_1_0 -o layer.vert.spv
spirv-val --target-env vulkan1.0 layer.vert.spv
slangc layer.slang -matrix-layout-column-major -O2 -entry vs_main -stage vertex -target metal -o layer.vert.msl
# Fragment inputs are matched by location: [[user(locnN)]], as SPIRV-Cross
# names them, where slangc writes the HLSL semantic.
sed -i.bak 's/\[\[user(TEXCOORD)\]\]/[[user(locn0)]]/' layer.vert.msl && rm layer.vert.msl.bak
echo "layer.slang -> layer.vert.spv, layer.vert.msl"
