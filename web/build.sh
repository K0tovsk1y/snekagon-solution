#!/usr/bin/env bash
# Builds the web page into site-out/ (or the folder given): the engine compiled to WebAssembly, the page and the
# proof data. A static site for GitHub Pages or any static server. Needs Emscripten (em++ on PATH).
#   bash web/build.sh [out]
#   python -m http.server -d site-out 8000      # then http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-site-out}
mkdir -p "$out"
em++ -O3 -std=c++20 src/web.cpp -o "$out/snek.js" \
  -sMODULARIZE=1 -sEXPORT_NAME=SnekEngine -sENVIRONMENT=worker \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=128MB -sSTACK_SIZE=8MB \
  -sEXPORTED_FUNCTIONS=_snek_cmd,_main -sEXPORTED_RUNTIME_METHODS=cwrap,FS -sFORCE_FILESYSTEM=1 \
  '-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$UTF8ToString'
cp site/index.html site/snek-worker.js "$out/"
rm -rf "$out/proof"
cp -r proof "$out/proof"
echo "built $out"
