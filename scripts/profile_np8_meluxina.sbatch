#!/bin/bash -l
#SBATCH --job-name=profile_colls_8_nodes
#SBATCH --account=p201236
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --output=out/out
#SBATCH --error=out/err

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p out/tree out/butterfly plots

# ── Build & run ───────────────────────────────────────────────────────────────
cmake -B build -DCA_PROFILE=ON -DCA_CUDA=ON
cmake --build build --parallel
mpirun -np 8 ./build/bin/bench

# ── Plot style ────────────────────────────────────────────────────────────────
PALETTE="#00C2D4,#7B5EA7,#E8593C,#F2A623,#1D9E75,#D4537E,#5B8FF9,#A0D911,#FF9F7F,#B37FEB"
BG="#0F1117"
NODE_COLOR="#5B8FF9"
NODE_EDGE="#8AFFAA"
GHOST_COLOR="#1E2235"
ROOT_COLOR="#F2A623"

# ── Tree plots ────────────────────────────────────────────────────────────────
for json in out/tree/*.json; do
    base=$(basename "$json" .json)
    mkdir -p "plots/${base}"
    python3 ./scripts/tree.py "$json"       \
        --out        "plots/${base}/${base}_tree.png" \
        --palette    "$PALETTE"             \
        --bg         "$BG"                  \
        --node-color "$NODE_COLOR"          \
        --root-color "$ROOT_COLOR"          \
        --node-edge  "$NODE_EDGE"           \
        --ghost-color "$GHOST_COLOR"        \
        --no-show
done

# ── Butterfly plots ───────────────────────────────────────────────────────────
for json in out/butterfly/*.json; do
    base=$(basename "$json" .json)
    mkdir -p "plots/${base}"
    for mode in grid overlay butterfly candle ladder; do
        python3 ./scripts/butterfly.py "$json" \
            --mode        "$mode"                           \
            --out         "plots/${base}/${base}_${mode}.png" \
            --dpi         400                               \
            --palette     "$PALETTE"                        \
            --bg          "$BG"                             \
            --node-color  "$NODE_COLOR"                     \
            --node-edge   "$NODE_EDGE"                      \
            --ghost-color "$GHOST_COLOR"                    \
            --arrow-lw    1.6                               \
            --no-show
    done
done
