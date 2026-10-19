# Snekagon is solved: red wins

**Snekagon**, the two-player hexagonal snake-building game by BoltKey, is a **win for the first player (red)**:
red opens in the centre, `1:60`, and wins against every defence. This repository holds the computer proof, a
program that checks it, and a web page where you can play blue against the proved strategy.

| | |
|---|---|
| result | red (player 1) wins; a winning first move is `1:60`, the centre |
| blue's replies to `1:60` | 1932 legal moves, 172 up to the board's symmetries, each refuted |
| the proof | [`proof/`](proof/): red's winning answer to each of the 172 replies, and red's strategy below it down to its 7th move (130 MB); the exact solver proves the rest |
| how it is built and checked | [PROOF.md](PROOF.md) |
| rules | [RULES.md](RULES.md) |

## Play it

The web page of this repository (GitHub Pages) lets you play blue against red's proved strategy. Every red move is
marked **theory** (stored in the proof: red's moves 1, 3, 5, 7) or **computed** (found right now by the exact
solver in your browser, from move 9 on; a few milliseconds there).

## Check it

```bash
g++ -O3 -march=native -std=c++20 -pthread -static -o build/snek.exe src/main.cpp -lws2_32   # Windows (MinGW)
g++ -O3 -march=native -std=c++20 -pthread -o build/snek src/main.cpp                        # Linux

build/snek proofcheck proof 16 1000      # the structure in full + 1000 random parts re-proved (~10 min on a laptop)
build/snek proofcheck proof 16           # everything (~26 laptop-hours; PROOFCHECK_PART splits it over machines)
build/snek proofgame proof 100           # red plays the proof against 100 random blue games
```

`proofcheck` checks that the stored strategy answers every blue move (up to symmetry) with a legal red move, then
re-proves with the exact solver that blue loses below red's 5th move, everywhere or at `k` random places. See
[PROOF.md](PROOF.md) for what exactly is checked and what a full check costs.

## What is here

| | |
|---|---|
| `src/` | the engine: rules and move generation (`core.hpp`), the exact solver (`solver.hpp`), the proof's reader, checker and player (`proof.hpp`, `proofplay.hpp`), the command line (`main.cpp`), the line protocol used by the web page (`serve.hpp`, `web.cpp`) |
| `proof/` | the proof data, format in [PROOF.md](PROOF.md) |
| `site/`, `web/build.sh` | the web page; `bash web/build.sh` builds it with Emscripten into `site-out/` (the GitHub Actions workflow does it on every push) |

## How it was found

The exact solver (negamax over win/draw/loss with sound bounds on the final snake lengths, transposition tables,
learned move ordering and many pruning rules) first showed that blue cannot win after `1:60`, then searched, for
each of blue's 172 replies, for a red answer after which blue loses. 123 replies fell to the first answer tried, the
hardest needed about 200. The search ran from 16 to 19 October 2026 and took about 160 laptop-hours (hours of an
8-core laptop).

## Credits

Snekagon, its rules, and the original engine, UI
are by BoltKey ([snekagon-lab](https://github.com/BoltKey/snekagon-lab)). The solver's speed-ups, the search and
the proof were built on top of that engine.
