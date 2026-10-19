# Snekagon is solved: red (the first player) wins

**Red wins by opening in the centre, `1:60`, against every defence.** This page describes the computed proof, the
data published with it and how to check it yourself, completely or on a random sample.

Play it: on the web page of this repository (GitHub Pages, built from [site/](site/)) you play blue and red plays
the stored winning strategy; every red move is marked as *theory* (stored) or *computed* (by the solver, live).

## The result

| question | answer |
|---|---|
| value of the game | **red (player 1) wins** |
| a winning first move | `1:60`, the centre (the other first moves were not needed) |
| blue's replies to `1:60` | 1932 legal moves, **172** up to the symmetries of the board, every one refuted |
| proof data | [`proof/`](proof/): `index.txt` and 172 files, 130 MB (about 37 MB compressed) |

Cells are numbered 0–120 row by row on an 11×11 rhombus whose 91 valid cells form the hexagonal board
([RULES.md](RULES.md)); a move is written `<piece>:<cells>`, e.g. `4B:16,27,28,39`.

## What the proof is

A winning strategy for red is a tree: one red move wherever red is to move, every legal blue move below it. Here
the tree is stored for the first seven moves, and the exact solver proves the rest:

```
move 1  red    1:60                                    stored
move 2  blue   every reply, up to symmetry             172 classes
move 3  red    the winning answer Z                    proof/index.txt (172)
move 4  blue   every reply, up to symmetry             166 912 classes   <- "records"
move 5  red    the winning answer                      stored (166 912)
        -------------------------------------------------------------------------------------------
        the position after move 5: blue to move, proved lost for blue by the exact solver
        -------------------------------------------------------------------------------------------
move 6  blue   every reply                             66 862 680 positions
move 7  red    a winning answer                        stored for 99.6 % (for instant play and spot checks)
```

The proof proper is the part above the line: 172 answers at move 3, 166 912 at move 5, and for each of those
positions a solver proof that blue loses. Move 7 is stored as well, so that the strategy can be played instantly
(the web version) and spot-checked leaf by leaf.

Each of the 172 files is the level-B output (`SOLVE_LEVELB`, [src/solver.hpp](src/solver.hpp)) of the solve that
proved "after `1:60 Y Z` blue loses" (`solvepos` with `SOLVE_DRAWTEST=1`: result `value -1`, every blue move
refuted). Format, little endian, records one after another:

```
uint16 'B' | uint8 type = 1 | int16 m | int16 rep | uint16 n | n x int16
```

`m` is blue's 4th move, `rep` red's winning 5th move, and the `n` entries are red's 7th move for each of blue's
6th moves, in the order of `Pos::genMoves` of the position after `m rep` (-1: not stored, -2: game over). Moves are
gids, indices into the engine's placement table (`moveStr` prints them in the text notation);
[src/proof.hpp](src/proof.hpp) reads the files.

## Checking it

```bash
g++ -O3 -march=native -std=c++20 -pthread -static -o build/snek.exe src/main.cpp -lws2_32   # Windows (MinGW)
g++ -O3 -march=native -std=c++20 -pthread -o build/snek src/main.cpp                        # Linux

build/snek proofcheck proof 16                      # everything, 16 threads
build/snek proofcheck proof 16 1000 7               # 1000 random records, seed 7
build/snek proofcheck proof 16 10000 7 leaves       # 10 000 random leaves below the stored 7th moves
PROOFCHECK_LOG=check.log build/snek proofcheck proof 16              # resumable: done records are skipped
PROOFCHECK_PART=0/4 PROOFCHECK_LOG=p0.log build/snek proofcheck proof 16   # a quarter (parts 0/4 .. 3/4)
```

`snek proofcheck <dir> [threads] [k|all] [seed] [records|leaves]` always checks the **structure** in full (seconds):
- the index covers every one of blue's replies to `1:60` exactly once up to symmetry, and every stored red move is
  legal;
- every file has a record for every class of blue's 4th move, and one entry per blue 6th move in each record.

Then the exact solver proves, for every **record** (or `k` random ones), that blue loses after red's 5th move.
This is the whole proof. With `leaves` it proves positions below the stored 7th moves instead, a spot check of
those moves. The output ends with `RESULT: VERIFIED - red wins with 1:60`, `RESULT: no counterexample in the
sample`, `RESULT: this part is verified ...`, or the failing positions and `RESULT: NOT VERIFIED`. Every run also
prints the average time per record (or leaf) and what all of them would take.

**Cost of the full check**: about **26 laptop-hours** (one laptop-hour: an hour of an 8-core laptop, about 20
million solver nodes per second), i.e. ~210 core-hours; `PROOFCHECK_PART` splits it over several machines.
Measured on 1000 random records: about 8 s per record and thread, very uneven (most take well under a second, a
few minutes). A random record is a full sub-proof, so 1000 random records (≈0.6 % of the proof) take about 10
minutes on such a laptop. Proving
all leaves instead would cost several times more (~15 ms per leaf, 67 M leaves, and no work shared between
siblings), which is why records are the default.

What you have to trust: the rules as implemented ([src/core.hpp](src/core.hpp), cross-checked by `perft` and a
reference move generator) and the exact solver ([src/solver.hpp](src/solver.hpp)). The solver's pruning rules can
be switched off by environment variables (`SOLVER_HORIZON=0 SOLVER_MIAI=0 SOLVER_MOVELB=0 ...`, see the top of
`struct Solver`); with all of them off it is close to a plain alpha-beta search with a transposition table, ~7
times slower, and checks samples independently of those rules.

## How it was computed

1. **Blue cannot win** (proved first, it was cheaper): for each of blue's 172 replies to `1:60` a red answer that
   holds at least a draw.
2. **Red wins**: for each reply a red answer after which blue loses, found by trying red's answers in the order of
   a shallow alpha-beta search (depth 1 and 2 interleaved), and first the answers that had already won against
   other replies. 123 replies fell to the first answer tried; the hardest, `3I:17,28,39` and `4B:25,36,37,48`,
   needed about 185 and 200 tries. Each try is one null-window solve (`SOLVE_DRAWTEST`) of the position after
   `1:60 Y Z`.

Compute, from the solve logs (16–19 October 2026): about 1.2·10¹³ solver nodes, roughly **160 laptop-hours**,
including the refuted answers and the "blue cannot win" step; the 172 winning solves alone took 4.0·10¹² nodes
(about 55 laptop-hours). The development runs before that are not counted.
