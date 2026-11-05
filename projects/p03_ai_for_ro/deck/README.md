# The deck's data scripts

The deck reads its numbers from the studies' saved results. Three scripts that its source register names made the data of
some pages:

| Script | Pages | Reads |
|---|---|---|
| `make_p1_data.py` | phase 1's charts of each forecaster: its training, validation and test days, with the 90 % ranges | `../phase1/` — its saved forecasts (`prep.py` first makes the pairs) |
| `make_wave_data.py` | the what-ifs on waves, and the October day of the 12-train plant they read | the October control run in `../research_package/bigplant/big/energy/results/` |
| `make_composite.py` | our plant from above, to scale | AI-made textures, not included here; the script holds every size it places |

The deck itself is [`ai-for-ro/index.html`](../../../ai-for-ro/index.html), published at
[engmtaleb.github.io/chemai/ai-for-ro](https://engmtaleb.github.io/chemai/ai-for-ro/).
