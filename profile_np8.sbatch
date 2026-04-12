#!/bin/bash
#SBATCH --job-name=profile_colls_8_nodes
#SBATCH --account=IscrC_FOCAL
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=out/out
#SBATCH --error=out/err

mkdir -p out/tree out/butterfly plots

make vendor
make bench PROFILE=1

mpirun -np 8 ./build/bin/bench

PALETTE="#00C2D4,#7B5EA7,#E8593C,#F2A623,#1D9E75,#D4537E,#5B8FF9,#A0D911,#FF9F7F,#B37FEB"
BG="#0F1117"
NODE_COLOR="#5B8FF9"
NODE_EDGE="#8AFFAA"
GHOST_COLOR="#1E2235"

for json in out/tree/*.json; do
	    base=$(basename "$json" .json)
	        mkdir -p plots/${base}
		    python3 ./scripts/tree.py "$json" \
			            --out "plots/${base}/${base}_tree.png" \
				            --palette "$PALETTE" \
					            --bg "$BG" \
						            --node-color "$NODE_COLOR" \
							            --root-color "#F2A623" \
								            --node-edge "$NODE_EDGE" \
									            --ghost-color "$GHOST_COLOR" \
										            --no-show
	    done

	    for json in out/butterfly/*.json; do
		        base=$(basename "$json" .json)
			    for mode in grid overlay butterfly candle ladder; do
				            mkdir -p plots/${base}
					            python3 ./scripts/butterfly.py "$json" \
							                --mode "$mode" \
									            --out "plots/${base}/${base}_${mode}.png" \
										                --dpi 400 \
												            --palette "$PALETTE" \
													                --bg "$BG" \
															            --node-color "$NODE_COLOR" \
																                --node-edge "$NODE_EDGE" \
																		            --ghost-color "$GHOST_COLOR" \
																			                --arrow-lw 1.6 \
																					            --no-show
						        done
						done
