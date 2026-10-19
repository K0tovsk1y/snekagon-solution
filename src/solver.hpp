// Exact solver: negamax alpha-beta over {-1, 0, +1} (loss/draw/win for the side
// to move), transposition table keyed by the symmetry-canonical hash, snake-length
// bounds for early cutoffs, TT-move + killer + history + learned-eval move ordering,
// enhanced transposition cutoffs (ETC). The TT is lockless and can be shared by threads.
#pragma once
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include "core.hpp"
#include "eval.hpp"
#include "lastply.hpp"

// Lockless 2-way transposition table. Each slot stores key^data and data, so an entry
// torn by concurrent writes fails the key check instead of returning garbage.
// Slot 0 of a bucket keeps the entry with the most work behind it, slot 1 is always replaced.
struct LTT {
    struct Slot { std::atomic<uint64_t> k{0}, d{0}; };
    std::unique_ptr<Slot[]> s;
    size_t nb;
    uint64_t mask;
    explicit LTT(int log2buckets) : s(new Slot[size_t(2) << log2buckets]), nb(size_t(1) << log2buckets), mask(nb - 1) {}
    void clear() {
        for (size_t i = 0; i < 2 * nb; i++) { s[i].k.store(0, std::memory_order_relaxed); s[i].d.store(0, std::memory_order_relaxed); }
    }
    static uint64_t pack(int lo, int hi, int best, int work) {
        return uint64_t(lo + 2) | uint64_t(hi + 2) << 3 | uint64_t(best + 1) << 6 | uint64_t(work) << 22;
    }
    bool probe(uint64_t key, int& lo, int& hi, int& best) const {
        size_t b = (key & mask) * 2;
        for (int i = 0; i < 2; i++) {
            uint64_t k = s[b + i].k.load(std::memory_order_relaxed), d = s[b + i].d.load(std::memory_order_relaxed);
            if (d && (k ^ d) == key) {
                lo = int(d & 7) - 2; hi = int(d >> 3 & 7) - 2; best = int(d >> 6 & 0xFFFF) - 1;
                return true;
            }
        }
        return false;
    }
    void store(uint64_t key, int lo, int hi, int best, int work) {
        size_t b = (key & mask) * 2;
        uint64_t d = pack(lo, hi, best, work);
        uint64_t k0 = s[b].k.load(std::memory_order_relaxed), d0 = s[b].d.load(std::memory_order_relaxed);
        int i = ((k0 ^ d0) == key || work >= int(d0 >> 22 & 63)) ? 0 : 1;
        s[b + i].d.store(d, std::memory_order_relaxed);
        s[b + i].k.store(key ^ d, std::memory_order_relaxed);
    }
};

inline int envInt(const char* name, int def) { const char* v = getenv(name); return v ? atoi(v) : def; }
// SOLVER_RED_BONUS=b counts b extra cells for red at the end of the game (default 0 = the real rules).
// With it, "blue still wins" means blue wins by more than b: used to find the exact margin of a line.
inline const int RED_BONUS = envInt("SOLVER_RED_BONUS", 0);

#ifdef SOLVER_TIMING
#include <x86intrin.h>
#define TSTART uint64_t _t0 = __rdtsc()
#define TLAP(i) do { uint64_t _t1 = __rdtsc(); tm[pp][i] += _t1 - _t0; _t0 = _t1; } while (0)
#define TRESET _t0 = __rdtsc()
#define FSTART uint64_t _f0 = __rdtsc()
#define FLAP(k) do { uint64_t _f1 = __rdtsc(); ft[pp][k] += _f1 - _f0; fc[pp][k]++; _f0 = _f1; } while (0)
#define FRESET _f0 = __rdtsc()
#else
#define FSTART
#define FLAP(k)
#define FRESET
#define TSTART
#define TLAP(i)
#define TRESET
#endif
struct Solver {
    uint64_t tm[16][8] = {};
    uint64_t ft[16][8] = {}, fc[16][8] = {};  // SOLVER_TIMING: cycles / calls of the bound steps
    // eval-ordered nodes per ply: count, cut at the 1st / 2nd-3rd / later move, no cut, moves searched
    uint64_t cs[16][2][8] = {};  // SOLVER_TIMING: kinds of eval-ordered children, see TLAP(4)
    uint64_t cutDir[16][2][3] = {};  // SOLVER_TIMING: see TLAP(2)
    uint64_t eo[16][6] = {};  // SOLVER_TIMING: cycles per ply: 0 horizon 1 tt 2 bounds 3 gen 4 order 5 select+play 6 store
    std::shared_ptr<LTT> tt;
    std::vector<int32_t> hist[2];
    int16_t killer[32][2];
    uint64_t nodes = 0;
    std::atomic<bool>* stop = nullptr;
    std::chrono::steady_clock::time_point deadline;
    bool useDeadline = false;
    bool aborted = false;
    uint64_t nodeLimit = UINT64_MAX;
    // profiling counters per ply: visited, cut by TT, cut by bounds, expanded
    uint64_t pVisit[16] = {}, pTT[16] = {}, pBound[16] = {}, pExp[16] = {}, pHor[16] = {};
    bool useHorizon = envInt("SOLVER_HORIZON", 1) != 0;  // exact last-two-plies evaluation (lastply.hpp)
    bool useOnePiece = envInt("SOLVER_ONEPIECE", 1) != 0;  // exact bounds for a player with one piece left
    bool lazyBounds = envInt("SOLVER_LAZYBOUNDS", 1) != 0;  // compute an upper bound only if it can cut
    bool capBounds = envInt("SOLVER_CAPBOUNDS", 1) != 0;    // ... and stop flooding once it cannot
    bool moveLB = envInt("SOLVER_MOVELB", 1) != 0;          // side to move: its best single placement is a lower bound
    bool miaiLB = envInt("SOLVER_MIAI", 1) != 0;            // opponent: two far-apart placements cannot both be spoiled
    bool inheritUB = envInt("SOLVER_INHERIT", 1) != 0;      // exact upper bounds of the parent are valid in the child
    int moveLB2 = envInt("SOLVER_MOVELB2", 1);
    int moveLB2Top = envInt("SOLVER_MOVELB2_TOP", 0);
    bool moveLB2Leaf = envInt("SOLVER_MOVELB2_LEAF", 0) != 0;  // 0: no two-move bound at 2-vs-1 nodes
    int moveLB2MinRem = envInt("SOLVER_MOVELB2_MINREM", 0);     // no two-move bound with fewer pieces left
    int lazyBslotMode = envInt("SOLVER_LAZYBSLOT", 1);
    bool lazyBslot = lazyBslotMode != 0;      // the children's best placements from the 2nd child on              // two-move lower bound: best placement(s), then miai (0 = off, k = try k first placements)
    uint64_t pML2 = 0;
    int16_t ubKnown[48][2];                                  // per ply: exact upper bounds (raw) known at the current node
    // per ply: the opponent's best single placements at the current (expanded) node, for the
    // children, where that opponent is to move (the move between can only spoil some of them)
    struct BSlot { bool valid; int8_t player, V, n; BB moverOwn; BB kill[4]; int16_t gid[4]; };
    BSlot bslot[48] = {};
    bool inheritBest = envInt("SOLVER_INHERITBEST", 1) != 0;
    bool cheapUB = envInt("SOLVER_CHEAPUB", 1) != 0;        // try a flood-free upper bound first
    bool exactMe = envInt("SOLVER_EXACTME", 1) != 0;        // our own upper bound exact (for the children)
    int selPicks = envInt("SOLVER_SELPICKS", 8);      // deep nodes: only the first picks are sorted
    uint64_t pCheap = 0;
    // flood-free upper bound: k pieces join at most k + 1 snakes and add at most their cells
    static int cheapUpper(const Pos& s, int c) {
        int top[9] = {}, k = pieceCount(s.avail[c]) + 1;
        for (int j = 0; j < s.ncomp[c]; j++) {
            int l = popc(s.comp[c][j]);
            for (int i = 0; i < k; i++) if (l > top[i]) { for (int q = k - 1; q > i; q--) top[q] = top[q - 1]; top[i] = l; break; }
        }
        int sum = remainingCells(s.avail[c]);
        for (int i = 0; i < k; i++) sum += top[i];
        return std::max<int>(s.L[c], sum);
    }
    uint64_t pIBP = 0;
    bool fastLeaf = envInt("SOLVER_FASTLEAF", 1) != 0;      // 2-vs-1-piece nodes: children evaluated in place
    bool stagedGen = envInt("SOLVER_STAGED", 1) != 0;       // deep nodes: TT move and killers before generating moves
    int stagedEval = envInt("SOLVER_STAGEDEVAL", 2);
    const std::atomic<int>* gRef = nullptr;  // see SOLVE_ROOTREF in parallelSolve
    int gRefPly = -1;
    bool useHCache = envInt("SOLVER_HCACHE", 0) != 0;       // cache of the leaf children's exact values
    static constexpr uint64_t HCMASK = (1u << 16) - 1;
    std::vector<uint64_t> hcache = std::vector<uint64_t>(HCMASK + 1, 0);
    uint64_t pHC = 0;
    int evalOrderRem = envInt("SOLVER_EVAL_REM", 8);  // order children by the learned evaluation when >= this many pieces are left
    // deeper down, eval-order only where the side to move does not look lost: the losing side
    // has to try every move anyway, so ordering its moves is wasted work
    int evalOrderRemWin = envInt("SOLVER_EVAL_WIN_REM", 6);
    double evalWinThreshold = envInt("SOLVER_EVAL_WIN_THR", 50) / 100.0;
    int etcRem = envInt("SOLVER_ETC_REM", 99);        // enhanced transposition cutoffs when >= this many pieces are left
    int evalTopK = envInt("SOLVER_EVALTOPK", 0);           // only the K best children by the cheap score are evaluated (0 = all)
    bool evalChunk = envInt("SOLVER_EVALCHUNK", 0) != 0;   // with EVALTOPK: evaluate the rest once the top K fail
    bool winBounds = envInt("SOLVER_WINBOUNDS", 0) != 0;   // bounds aware of a null window
    bool inhFirst = envInt("SOLVER_INHFIRST", 1) != 0;     // inherited exact bound before the one-piece bound
    bool cheapFirst = envInt("SOLVER_CHEAPFIRST", 1) != 0;  // flood-free bound before the one-piece bound
    int eval2 = envInt("SOLVER_EVAL2", 0);
    bool ttLateLeaf = envInt("SOLVER_TTLATE", 1) != 0;      // 2-vs-1 nodes: bounds before the TT probe
    bool preCut = envInt("SOLVER_PRECUT", 1) != 0;          // 2-vs-1 children cut by the parent
    bool preCut2 = envInt("SOLVER_PRECUT2", 1) != 0;        // ... also when our move spoils their best placements
    bool preCutHi = envInt("SOLVER_PRECUT_HI", 0) != 0;     // ... and the other direction (the opponent loses)
    uint64_t pPreHi = 0;
    int evalUbW = envInt("SOLVER_EVAL_UBW", 10);
    double evalLoseThr = envInt("SOLVER_EVAL_LOSE_THR", 200) / 100.0;  // see doEval
    int blockW = envInt("SOLVER_BLOCKW", 256);
    int blockMode = envInt("SOLVER_BLOCKMODE", 0);
    int blockWWin = envInt("SOLVER_BLOCKW_WIN", -1);     // blocking bonus where beta > 0 (-1: same as BLOCKW)            // cheap ordering: bonus for taking a cell of the opponent's growth front            // eval ordering: minus W/100 logit per cell of the opponent's upper bound
    int counterMode = envInt("SOLVER_COUNTER", 0);
    bool useCounter = counterMode != 0;                     // countermove after TT move and killers
    int16_t lastGid[48];                                    // the move played at each ply on the current path
    std::vector<int16_t> cmove[2];                          // [side][opponent's previous move] -> refutation
    uint64_t pPre = 0;                  // two-stage eval ordering: full eval for the top M
    bool evalWinOnly = envInt("SOLVER_EVALWINONLY", 0) != 0;  // deeper eval ordering only when beta > 0
    double evalDrawThr = envInt("SOLVER_EVALDRAW_THR", 350) / 100.0;  // see doEval
    int evalDrawRem = envInt("SOLVER_EVALDRAW_REM", 7);  // ... only where >= this many pieces are left
    // SOLVER_EVALCACHE=<log2 entries>: per-thread cache of learned evaluations by Zobrist hash
    int evalCacheLog2 = envInt("SOLVER_EVALCACHE", 0);
    struct ECSlot { uint64_t key; float v; };
    std::vector<ECSlot> ecache = std::vector<ECSlot>(evalCacheLog2 ? size_t(1) << evalCacheLog2 : 0, ECSlot{0, 0});
    uint64_t pECHit = 0, pECMiss = 0;
    double cachedLogit(const Pos& c) {
        if (ecache.empty()) return learnedLogit(c);
        ECSlot& e = ecache[c.h[0] & (ecache.size() - 1)];
        if (e.key == c.h[0]) { pECHit++; return e.v; }
        pECMiss++;
        double v = learnedLogit(c);
        e.key = c.h[0]; e.v = (float)v;
        return v;
    }

    // log2size: log2 of the number of entries (16 bytes each)
    explicit Solver(int log2size = 22) : tt(std::make_shared<LTT>(log2size - 1)) { init(); }
    explicit Solver(std::shared_ptr<LTT> shared) : tt(std::move(shared)) { init(); }
    std::vector<Move> mvBuf[32];
    std::vector<int> scBuf[32];
    void init() {
        for (int i = 0; i < 32; i++) { mvBuf[i].resize(3000); scBuf[i].resize(3000); }
        for (auto& h : hist) h.assign(T.NGID, 0);
        std::memset(killer, -1, sizeof killer);
        for (auto& c : cmove) c.assign(T.NGID, -1);
        std::memset(lastGid, -1, sizeof lastGid);
    }
    void clear() {
        tt->clear();
        std::fill(hcache.begin(), hcache.end(), 0);
        std::fill(ecache.begin(), ecache.end(), ECSlot{0, 0});
        for (auto& h : hist) std::fill(h.begin(), h.end(), 0);
        std::memset(killer, -1, sizeof killer);
        for (auto& c : cmove) std::fill(c.begin(), c.end(), -1);
    }

    static int terminalValue(const Pos& s) {
        int d = finalScore(s) + RED_BONUS;
        int v = d > 0 ? 1 : d < 0 ? -1 : 0;
        return s.stm == 0 ? v : -v;
    }

    bool checkAbort() {
        if (aborted) return true;
        if ((nodes & 4095) == 0) {
            if (spec) spec->pump();
            if (stop && stop->load(std::memory_order_relaxed)) aborted = true;
            else if (useDeadline && std::chrono::steady_clock::now() > deadline) aborted = true;
        }
        if (nodes > nodeLimit) aborted = true;
        return aborted;
    }

    // Speculative cache filled by another engine (gpu/specsolve.cu): positions at taskPly are
    // looked up first (exact values); at specPly nodes the engine is told which positions of the
    // subtree to solve ahead of the search. The search never waits for it.
    struct SpecHook {
        virtual bool lookup(uint64_t key, int& v) = 0;
        virtual void bundle(Solver& S, const Pos& s) = 0;
        // at specPly nodes, before searching child i (moves/score as ordered so far)
        virtual void beforeChild(Solver& S, const Pos& s, const Move* moves, const int* score, int i, int n) = 0;
        // at taskPly - 1 nodes, after move ordering: children not sent yet can still be sent
        virtual void lateBundle(Solver& S, const Pos& s, const Move* moves, const int* score, int n) = 0;
        virtual void pump() = 0;
        virtual ~SpecHook() {}
    };
    SpecHook* spec = nullptr;
    int specPly = 6, taskPly = 9;
    uint64_t specHits = 0, specMiss = 0;

    // task dump (DUMP_PLY=p): every position searched at ply p with its window and result
    FILE* dumpOut = nullptr;
    std::mutex* dumpMu = nullptr;
    int dumpPly = -1;
    int search(const Pos& s, int alpha, int beta) {
        if (s.ply != dumpPly || !dumpOut) return searchImpl(s, alpha, beta);
        uint64_t n0 = nodes;
        int v = searchImpl(s, alpha, beta);
        if (aborted) return v;
        std::lock_guard<std::mutex> lk(*dumpMu);
        fprintf(dumpOut, "%016llx%016llx_%016llx%016llx_%d_%d_%d_%d_%d %d %d %d %llu\n",
                (unsigned long long)(uint64_t)(s.own[0] >> 64), (unsigned long long)(uint64_t)s.own[0],
                (unsigned long long)(uint64_t)(s.own[1] >> 64), (unsigned long long)(uint64_t)s.own[1],
                s.avail[0], s.avail[1], s.stm, s.ply, s.passes, alpha, beta, v, (unsigned long long)(nodes - n0));
        return v;
    }
    int searchImpl(const Pos& s, int alpha, int beta) {
        nodes++;
        if (checkAbort()) return 0;
        if (s.over()) return terminalValue(s);
        uint64_t n0 = nodes;
        int pp = std::min<int>(s.ply, 15);
        TSTART;
        if (useHorizon) {
            int hv = horizonValue(s, RED_BONUS);
            TLAP(0);
            if (hv != -9) { pHor[pp]++; return hv; }
        }
        if (spec) {
            if (s.ply == taskPly) {
                int t, v;
                if (spec->lookup(s.key(t), v)) { specHits++; return v; }
                specMiss++;
            } else if (s.ply == specPly) spec->bundle(*this, s);
        }
        pVisit[pp]++;

        int lo = -1, hi = 1;
        int t = 0;
        // SOLVER_TTLATE: at "2 pieces against 1" nodes the bounds decide ~98 % and the TT ~0.3 %,
        // so the TT (a cache miss) is probed only when the bounds did not cut
        const bool ttLate = ttLateLeaf && pieceCount(s.avail[s.stm]) == 2 && pieceCount(s.avail[s.stm ^ 1]) == 1;
        uint64_t k = ttLate ? 0 : s.key(t);
        int ttMove = -1, eLo, eHi, eBest = -1;
        if (!ttLate && tt->probe(k, eLo, eHi, eBest)) {
            lo = eLo; hi = eHi;
            if (lo >= hi || lo >= beta) { pTT[pp]++; return lo; }
            if (hi <= alpha) { pTT[pp]++; return hi; }
            if (eBest >= 0) ttMove = T.gidSym[eBest][T.inv[t]];
        }
        TLAP(1);
        // snake-length bound cutoffs
        int lbMe, ubMe, lbOp, ubOp;
        int bonus = s.stm == 0 ? RED_BONUS : -RED_BONUS;  // the side to move's share of SOLVER_RED_BONUS
        if (lazyBounds) {
            // An upper bound only matters if it can produce a cutoff: the opponent's when we may be
            // ahead of it (lbMe >= its length), ours when we may stay at or below the opponent's
            // length. Skipped bounds stay "unbounded" - the same decisions as computing them.
            const int big = 1000;
            bool meDone = useOnePiece && onePieceBounds(s, s.stm, lbMe, ubMe);
            if (!meDone) { lbMe = s.L[s.stm]; ubMe = big; }
            lbMe += bonus;
            if (meDone) ubMe += bonus;
            lbOp = s.L[s.stm ^ 1];
            ubOp = big;
            // the side to move places a piece now, where nothing can block it: its best single
            // placement is a sure lower bound
            int bestGid = -2;  // a placement achieving lb2 if known (-2: not looked up)
            FSTART;
            if (moveLB && !meDone && s.ply > 0 && pieceCount(s.avail[s.stm]) >= 2) {
                int lb2 = -1;
                // the parent found our best placements; its move spoils them only if it touches them
                const BSlot& bs = bslot[s.ply - 1];
                if (inheritBest && bs.valid && bs.player == s.stm) {
                    const BB moved = s.own[s.stm ^ 1] & ~bs.moverOwn;
                    for (int i = 0; i < bs.n; i++)
                        if (!(bs.kill[i] & moved)) { lb2 = bs.V; bestGid = bs.gid[i]; pIBP++; break; }
                }
                if (lb2 < 0) lb2 = bestValueAny(s.own[s.stm], s.comp[s.stm], s.ncomp[s.stm], s.L[s.stm], s.movable(), s.own[0] | s.own[1]);
                lbMe = std::max(lbMe, lb2 + bonus);
            }
            FLAP(0);
            // exact upper bounds known at the parent stay valid here (fewer continuations)
            const int inhMe = inheritUB && s.ply > 0 ? ubKnown[s.ply - 1][s.stm] : big;
            const int inhOp = inheritUB && s.ply > 0 ? ubKnown[s.ply - 1][s.stm ^ 1] : big;
            int knMe = inhMe, knOp = inhOp;  // exact (raw) upper bounds known here, for the children
            if (meDone) knMe = std::min(knMe, ubMe - bonus);
            bool opExact = true;
            // window-aware (SOLVER_WINBOUNDS): with alpha >= 0 only a win (lbMe > ubOp) can matter,
            // with beta <= 0 only a loss (ubMe < lbOp) - null-window searches inside proofs
            const bool needWin = winBounds && alpha >= 0, needLoss = winBounds && beta <= 0;
            const int opCap = lbMe - (needWin ? 1 : 0);  // ubOp <= opCap is what can cut
            if (opCap >= lbOp) {  // only ubOp <= opCap can cut
                int l2;
                // SOLVER_INHFIRST: the parent's exact bound first, if it already decides the window
                int cu;  // SOLVER_CHEAPFIRST: the flood-free bound first, if it already decides the window
                if (inhFirst && (inhOp < lbMe || (inhOp <= lbMe && beta <= 0))) ubOp = inhOp;
                else if (cheapFirst && ((cu = cheapUpper(s, s.stm ^ 1)) < lbMe || (cu <= lbMe && beta <= 0))) { ubOp = cu; knOp = std::min(knOp, cu); pCheap++; }
                else if (useOnePiece && onePieceBounds(s, s.stm ^ 1, l2, ubOp)) knOp = std::min(knOp, ubOp);
                else if (inhOp < lbMe) ubOp = inhOp;  // a win already: no flood needed
                else if (cheapUB && cheapUpper(s, s.stm ^ 1) <= opCap && (needWin || cheapUpper(s, s.stm ^ 1) < lbMe)) { ubOp = cheapUpper(s, s.stm ^ 1); knOp = std::min(knOp, ubOp); pCheap++; }
                else if (capBounds) {
                    snakeBoundsCap(s, s.stm ^ 1, opCap, l2, ubOp);
                    if (ubOp <= opCap) knOp = std::min(knOp, ubOp);
                    else if (ubOp < inhOp) opExact = false;  // a lower estimate only
                    ubOp = std::min(ubOp, inhOp);
                } else { snakeBounds(s, s.stm ^ 1, l2, ubOp); knOp = std::min(knOp, ubOp); }
                FLAP(1);
                // two moves ahead: after our best placement and any reply, the miai argument
                // guarantees our next placement
                const int needLB = ubOp + (needWin ? 1 : 0);  // lbMe >= needLB cuts
                // SOLVER_MOVELB2_TOP=k: more first placements where >= 7 pieces are left (the
                // eval-ordered plies, where a bound cutoff also saves the ordering)
                const int ml2 = (!moveLB2Leaf && pieceCount(s.avail[s.stm]) == 2 && pieceCount(s.avail[s.stm ^ 1]) == 1) ||
                                        pieceCount(s.avail[0]) + pieceCount(s.avail[1]) < moveLB2MinRem ? 0
                              : moveLB2Top > 0 && pieceCount(s.avail[0]) + pieceCount(s.avail[1]) >= 7 ? moveLB2Top : moveLB2;
                if (ml2 && lbMe < needLB + (needWin ? 0 : 1) && ubOp < big && s.ply > 0 && pieceCount(s.avail[s.stm]) >= 2) {
                    int gids[4];
                    int ng;
                    if (bestGid >= 0 && ml2 == 1) { gids[0] = bestGid; ng = 1; }
                    else ng = topPlacements(s.own[s.stm], s.comp[s.stm], s.ncomp[s.stm], s.L[s.stm], s.movable(), s.own[0] | s.own[1], gids, ml2);
                    for (int gi = 0; gi < ng && lbMe < needLB + (needWin ? 0 : 1); gi++) {
                        const int gid = gids[gi];
                        Pos c = s;
                        c.play(Move{(int16_t)gid});
                        const int m3 = miaiLower(c, s.stm, needLB - bonus) + bonus;
                        if (m3 >= needLB && m3 > lbMe) {
                            // the opponent's bound was computed (capped) for the old lbMe
                            if (!opExact) {
                                int l3;
                                snakeBoundsCap(s, s.stm ^ 1, 999, l3, ubOp);
                                knOp = std::min(knOp, ubOp);
                                ubOp = std::min(ubOp, inhOp);
                            }
                            lbMe = m3;
                            pML2++;
                        }
                    }
                }
                FLAP(3);
            }
            FRESET;
            bool ubExact = true;
            const int meCut = lbOp - (needLoss ? 1 : 0);  // ubMe <= meCut is what can cut
            if (!meDone && s.L[s.stm] + bonus <= meCut) {  // only ubMe <= meCut can cut
                int l2;
                if (inhMe + bonus < lbOp) ubMe = inhMe;  // a loss already: no flood needed
                else if (capBounds) {
                    // exact when the children can use it (they compare it with the opponent's best placement)
                    const int cap = exactMe && pieceCount(s.avail[s.stm ^ 1]) >= 2 ? 999 : meCut - bonus;
                    snakeBoundsCap(s, s.stm, cap, l2, ubMe);
                    ubExact = ubMe <= cap;  // otherwise only a lower estimate of the bound
                    if (ubExact) knMe = std::min(knMe, ubMe);
                    if (inhMe <= ubMe) { ubMe = inhMe; ubExact = true; }
                } else { snakeBounds(s, s.stm, l2, ubMe); knMe = std::min(knMe, ubMe); }
                ubMe += bonus;
                FLAP(4);
            }
            FRESET;
            // no cutoff from our upper bound yet: can the opponent force at least as much?
            const int needOp = ubMe + (needLoss ? 1 : 0);  // lbOp >= needOp cuts
            if (miaiLB && ubMe < big && lbOp < needOp + (needLoss ? 0 : 1) && ubMe >= lbOp && s.ply > 0 && s.avail[s.stm ^ 1]) {
                int m = miaiLower(s, s.stm ^ 1, needOp);
                if (m >= needOp) {
                    if (!ubExact) {  // the exact bound is needed now
                        int l2;
                        snakeBoundsCap(s, s.stm, 999, l2, ubMe);
                        knMe = std::min(knMe, ubMe);
                        ubMe = std::min(ubMe, inhMe) + bonus;
                    }
                    lbOp = std::max(lbOp, m);
                }
                FLAP(5);
            }
            ubKnown[s.ply][s.stm] = (int16_t)std::min(knMe, (int)big);
            ubKnown[s.ply][s.stm ^ 1] = (int16_t)std::min(knOp, (int)big);
        } else {
            if (!useOnePiece || !onePieceBounds(s, s.stm, lbMe, ubMe)) snakeBounds(s, s.stm, lbMe, ubMe);
            if (!useOnePiece || !onePieceBounds(s, s.stm ^ 1, lbOp, ubOp)) snakeBounds(s, s.stm ^ 1, lbOp, ubOp);
            lbMe += bonus;
            ubMe += bonus;
        }
        int blo = lbMe > ubOp ? 1 : lbMe >= ubOp ? 0 : -1;
        int bhi = ubMe < lbOp ? -1 : ubMe <= lbOp ? 0 : 1;
        lo = std::max(lo, blo);
        hi = std::min(hi, bhi);
        TLAP(2);
#ifdef SOLVER_TIMING
        {   // which side of the bounds decides: [win asked][0 lo cut, 1 hi cut, 2 none]
            const int w = beta > 0 ? 1 : 0;
            cutDir[pp][w][lo >= beta ? 0 : hi <= alpha ? 1 : lo >= hi ? 0 : 2]++;
        }
#endif
        if (lo >= hi || lo >= beta) { pBound[pp]++; return lo; }
        if (hi <= alpha) { pBound[pp]++; return hi; }
        if (ttLate) {
            k = s.key(t);
            if (tt->probe(k, eLo, eHi, eBest)) {
                lo = std::max(lo, eLo); hi = std::min(hi, eHi);
                if (lo >= hi || lo >= beta) { pTT[pp]++; return lo; }
                if (hi <= alpha) { pTT[pp]++; return hi; }
                if (eBest >= 0) ttMove = T.gidSym[eBest][T.inv[t]];
            }
        }
        pExp[pp]++;
        alpha = std::max(alpha, lo);
        beta = std::min(beta, hi);
        int a0 = alpha, b0 = beta;
        // the opponent's best placements, for the children (see the moveLB step). SOLVER_LAZYBSLOT:
        // only once a second child is searched - where the first child already cuts, its own
        // bestValueAny is cheaper (parents of PRECUT children need them for every child: eager)
        auto computeBslot = [&]() {
            BSlot& bs = bslot[s.ply];
            const int op = s.stm ^ 1;
            FSTART;
            int nn;
            const int V = bestPlacementsKill(s.own[op], s.comp[op], s.ncomp[op], s.L[op], s.avail[op], s.own[0] | s.own[1], bs.kill, bs.gid, nn);
            bs.n = (int8_t)nn; bs.V = (int8_t)V; bs.player = (int8_t)op; bs.moverOwn = s.own[s.stm];
            bs.valid = nn > 0;
            FLAP(6);
        };
        bool bsPending = false;
        int nChild = 0;
        {
            bslot[s.ply].valid = false;
            const int op = s.stm ^ 1;
            if (inheritBest && moveLB && s.ply > 0 && pieceCount(s.avail[op]) >= 2) {
                if (lazyBslot && (lazyBslotMode == 2 || !(preCut && pieceCount(s.avail[s.stm]) == 2 && pieceCount(s.avail[op]) == 2))) bsPending = true;
                else computeBslot();
            }
        }
        auto beforeChild = [&]() {
            if (bsPending && nChild >= 1) { computeBslot(); bsPending = false; }
            nChild++;
        };
        int best = -2, bestGid = -1;

        // staged generation (deep nodes): the TT move and the killers are tried before the
        // other moves are generated - at cut nodes one of them usually refutes
        int preG[6], npre = 0;
        // SOLVER_STAGEDEVAL=1: also at the eval-ordered nodes where a win is asked (beta > 0), 2: at all
        // of them. The eval ordering keeps the TT move and the killers in front anyway, so the tree is
        // the same - only the learned eval of every child is skipped when one of them cuts (R1: the
        // ordering at ply 8 was 36 % of all cycles, 98 % of those nodes cut at the first child)
        const int remAll = pieceCount(s.avail[0]) + pieceCount(s.avail[1]);
        if (stagedGen && s.ply > 0 && (remAll < evalOrderRem || (stagedEval && (stagedEval == 2 || beta > 0))) &&
            !(fastLeaf && pieceCount(s.avail[s.stm]) == 2 && pieceCount(s.avail[s.stm ^ 1]) == 1) &&
            (!spec || (s.ply != specPly && s.ply != taskPly - 1))) {
            const int ply0 = std::min<int>(s.ply, 31);
            // SOLVER_COUNTER: also the move that last refuted the opponent's previous move here
            // (SOLVER_COUNTER=2: only where "at least a draw" is asked, beta <= 0 - many moves hold there)
            const int cm = useCounter && (counterMode != 2 || beta <= 0) && s.ply > 0 && lastGid[s.ply - 1] >= 0 ? cmove[s.stm][lastGid[s.ply - 1]] : -1;
            // at the roots of the root moves' subproblems (gRefPly): also the two replies that refuted
            // the most root moves so far (SOLVE_ROOTREF, shared by all threads)
            int g1 = -1, g2 = -1;
            if (gRef && s.ply == gRefPly) {
                int c1 = 0, c2 = 0;
                for (int g = 0; g < T.NGID; g++) {
                    const int c = gRef[g].load(std::memory_order_relaxed);
                    if (c > c1) { c2 = c1; g2 = g1; c1 = c; g1 = g; } else if (c > c2) { c2 = c; g2 = g; }
                }
            }
            const int cand[6] = {ttMove, killer[ply0][0], killer[ply0][1], cm, g1, g2};
            const uint8_t mov = s.movable();
            for (int ci = 0; ci < 6 && alpha < beta; ci++) {
                const int gid = cand[ci];
                if (gid < 0) continue;
                bool dup = false;
                for (int q = 0; q < npre; q++) dup |= preG[q] == gid;
                if (dup) continue;
                Move m{(int16_t)gid};
                if (!(mov >> m.piece() & 1) || !s.legal(m.pl())) continue;
                preG[npre++] = gid;
                Pos c = s;
                c.play(m);
                TLAP(5);
                lastGid[s.ply] = (int16_t)gid;
                beforeChild();
                int v = -search(c, -beta, -alpha);
                TRESET;
                if (aborted) return 0;
                if (v > best) { best = v; bestGid = gid; }
                if (v > alpha) alpha = v;
                if (alpha >= beta) {
                    if (useCounter && s.ply > 0 && lastGid[s.ply - 1] >= 0) cmove[s.stm][lastGid[s.ply - 1]] = (int16_t)gid;
                    hist[s.stm][gid] += 1 + 4 * s.ply;
                    if (killer[ply0][0] != gid) { killer[ply0][1] = killer[ply0][0]; killer[ply0][0] = (int16_t)gid; }
                }
            }
        }
        if (alpha < beta) {

        // generate + order
        // per-depth move buffers (no per-node initialisation of 3000-element arrays)
        Move* moves = mvBuf[s.ply & 31].data();
        int* score = scBuf[s.ply & 31].data();
        int n = 0;
        int ply = std::min<int>(s.ply, 31);
        int k1 = killer[ply][0], k2 = killer[ply][1];
        auto& H = hist[s.stm];
        // leaf node (2 pieces against 1): children evaluated in place from the opponent's
        // precomputed placements (lastply.hpp); only the TT move and killers are ordered
        const bool leaf = fastLeaf && s.ply > 0 && pieceCount(s.avail[s.stm]) == 2 && pieceCount(s.avail[s.stm ^ 1]) == 1;
        if (leaf) {
            s.forEachLegal([&](Move m) {
                moves[n] = m;
                score[n] = m.gid == ttMove ? 3 : m.gid == k1 ? 2 : m.gid == k2 ? 1 : 0;
                n++;
            });
        } else {
        BB me = s.own[s.stm];
        BB opp = s.own[s.stm ^ 1];
        const Shape oppSh = shapeOf(opp);
        const int bw = beta > 0 && blockWWin >= 0 ? blockWWin : blockW;  // SOLVER_BLOCKW_WIN: where a win is asked
        BB oppFront = expand(oppSh.ends) & s.empty();
        // SOLVER_BLOCKMODE: 0 flat bonus for touching the front, 1 per front cell taken,
        // 2 flat but only the front of the opponent's longest snake(s), 3 per cell of that front
        if (blockMode >= 2) {
            BB lng = 0;
            for (int j = 0; j < s.ncomp[s.stm ^ 1]; j++)
                if (popc(s.comp[s.stm ^ 1][j]) == s.L[s.stm ^ 1]) lng |= s.comp[s.stm ^ 1][j];
            oppFront = expand(oppSh.ends & lng) & s.empty();
        }
        s.forEachLegal([&](Move m) {
            const Placement& pl = m.pl();
            int sc = H[m.gid] + PIECE_CELLS[m.piece()] * 64;
            if (pl.nb & me) sc += 256;
            if (pl.m & oppFront) sc += (blockMode & 1) ? bw * popc(pl.m & oppFront) : bw;  // blocking the opponent's growth
            else if (pl.nb & opp) sc += 32;
            if (m.gid == ttMove) sc = 1 << 30;
            else if (m.gid == k1) sc = 1 << 29;
            else if (m.gid == k2) sc = 1 << 28;
            moves[n] = m; score[n] = sc; n++;
        });
        }
        TLAP(3);
        int rem = pieceCount(s.avail[0]) + pieceCount(s.avail[1]);
        // SOLVER_EVALWINONLY: the conditional (deeper) eval ordering only where a win is asked for
        // (beta > 0); "at least a draw" usually has many good moves, so the cheap order will do
        // SOLVER_EVALDRAW_THR=T: where only "at least a draw" is asked (beta <= 0), the deeper
        // eval ordering is skipped once the side to move looks clearly winning (logit > T/100):
        // then most moves hold the draw and the cheap order does
        // SOLVER_EVAL_LOSE_THR=T: even where ordering is unconditional (>= EVAL_REM pieces left), skip
        // it when the side to move looks clearly lost (logit < -T/100): it has to try every move anyway
        bool doEval = n > 1 && rem >= evalOrderRem && (evalLoseThr >= 1e6 || learnedLogit(s) > -evalLoseThr);
        if (n > 1 && !doEval && rem >= evalOrderRemWin) {
            const double z = learnedLogit(s);
            doEval = z > evalWinThreshold && (!evalWinOnly || beta > 0) && (beta > 0 || rem < evalDrawRem || z <= evalDrawThr);
        }
        bool doEtc = n > 0 && rem >= etcRem;
        bool evalDone = false;  // set when the two-stage ordering ran
        // SOLVER_EVALTOPK=K: only the K best children by the cheap score get the learned
        // evaluation; they are ordered first, the rest follow in cheap-score order
        int evalFloor = INT32_MIN;
        if (doEval && evalTopK > 0 && n > evalTopK) {
            int* tmp = scBuf[(s.ply + 16) & 31].data();
            std::copy(score, score + n, tmp);
            std::nth_element(tmp, tmp + (evalTopK - 1), tmp + n, std::greater<int>());
            evalFloor = tmp[evalTopK - 1];
        }
        // SOLVER_EVAL2=M: two-stage ordering - the partial logit (no cluster / ub / fresh terms) for
        // every child, the full learned eval only for the M best of them; the full eval's best child
        // is among the partial top 24 in ~99% of the nodes
        if (doEval && !doEtc && eval2 > 0 && n > eval2) {
            int* tmp = scBuf[(s.ply + 16) & 31].data();
            int nc = 0;
            for (int i = 0; i < n; i++) {
                if (score[i] >= (1 << 28)) continue;  // TT move / killers keep their place
                Pos c = s; c.play(moves[i]);
                int p = c.over() ? -terminalValue(c) * 100000 : -(int)(1000 * partialLogit(c));
                score[i] = std::clamp(p, -(1 << 25), 1 << 25);
                tmp[nc++] = score[i];
            }
            int thr = INT32_MIN;
            if (nc > eval2) { std::nth_element(tmp, tmp + (eval2 - 1), tmp + nc, std::greater<int>()); thr = tmp[eval2 - 1]; }
            for (int i = 0; i < n; i++) {
                if (score[i] >= (1 << 28) || score[i] < thr) continue;
                Pos c = s; c.play(moves[i]);
                int e = c.over() ? -terminalValue(c) * 100000 : -(int)(1000 * cachedLogit(c));
                score[i] = (1 << 27) + std::clamp(e, -(1 << 25), 1 << 25);
            }
            doEval = false;  // ordered; skip the one-stage loop below
            evalDone = true;
        }
        if (doEval || doEtc) {
            for (int i = 0; i < n; i++) {
                if (!doEtc && score[i] < evalFloor) continue;
                Pos c = s; c.play(moves[i]);
                if (doEtc) {
                    // enhanced transposition cutoff: a child already known to be good enough
                    int cv = -9;
                    if (c.over()) cv = -terminalValue(c);
                    else { int ct, clo, chi, cb; if (tt->probe(c.key(ct), clo, chi, cb)) cv = -chi; }  // our value >= -chi
                    if (cv >= beta) {
                        tt->store(k, std::max(lo, cv), hi, T.gidSym[moves[i].gid][t], 0);
                        return cv;
                    }
                }
                if (doEval && score[i] < (1 << 28) && score[i] >= evalFloor) {
                    int e;
                    if (c.over()) e = -terminalValue(c) * 100000;
                    else if (evalUbW > 0) {
                        // SOLVER_EVAL_UBW=W: also prefer moves that cut the opponent's potential (its
                        // upper bound, free from the eval's features): cheaper proofs (offline: D1 -10%)
                        int ub = 0;
                        const double z = learnedLogit(c, &ub);
                        e = -(int)(1000 * z) - 10 * evalUbW * ub;
                    } else e = -(int)(1000 * cachedLogit(c));
                    score[i] = evalFloor == INT32_MIN ? e : (1 << 26) + std::clamp(e, -(1 << 24), 1 << 24);
                }
            }
        }
        TLAP(4);
#ifdef SOLVER_TIMING
        // which kind of move the eval puts first: touching our snake (A), taking the opponent's
        // front (B) - and how many children of each kind there are
        if (doEval && !leaf && n > 1) {
            const BB me2 = s.own[s.stm];
            const BB front2 = expand(shapeOf(s.own[s.stm ^ 1]).ends) & s.empty();
            int bi = 0, nA = 0, nB = 0, nAB = 0;
            for (int i = 0; i < n; i++) {
                if (score[i] > score[bi]) bi = i;
                const Placement& pl = moves[i].pl();
                const bool a = (pl.nb & me2) != 0, b = (pl.m & front2) != 0;
                nA += a; nB += b; nAB += a || b;
            }
            const Placement& pb = moves[bi].pl();
            const bool a = (pb.nb & me2) != 0, b = (pb.m & front2) != 0;
            uint64_t* c = cs[pp][beta > 0 ? 1 : 0];
            c[0]++; c[1] += n; c[2] += nA; c[3] += nB; c[4] += nAB; c[5] += a; c[6] += b; c[7] += a || b;
        }
#endif
        if (spec && s.ply == taskPly - 1 && n > 1) spec->lateBundle(*this, s, moves, score, n);
        if (n == 0) {
            Pos c = s;
            c.play(Move{});
            lastGid[s.ply] = -1;
            best = -search(c, -beta, -alpha);
            TRESET;
        } else {
            LeafPre pre;
            if (leaf) leafPre(s, pre);
            // SOLVER_PRECUT: children that are "2 pieces against 1" nodes are mostly cut by two cheap
            // bounds - the mover's best placement (known here: bslot) against the flood-free upper bound
            // of our final snake. Test that here, without building the child.
            const int op = s.stm ^ 1;
            const BSlot& pbs = bslot[s.ply];
            const bool preBase = preCut && !leaf && pieceCount(s.avail[s.stm]) == 2 && pieceCount(s.avail[op]) == 2;
            const int ubOpHere = preBase && preCutHi ? std::min<int>(ubKnown[s.ply][op], cheapUpper(s, op)) : 1000;
            const int bonusOp = op == 0 ? RED_BONUS : -RED_BONUS;
            bool chunkPending = evalChunk && evalFloor != INT32_MIN;
            for (int i = 0; i < n; i++) {
                // SOLVER_EVALCHUNK=1: the top-K chunk did not cut - now evaluate the rest too
                if (chunkPending && i >= evalTopK) {
                    for (int j = i; j < n; j++)
                        if (score[j] < (1 << 26)) {
                            Pos c = s; c.play(moves[j]);
                            int e = c.over() ? -terminalValue(c) * 100000 : -(int)(1000 * cachedLogit(c));
                            score[j] = (1 << 26) + std::clamp(e, -(1 << 24), 1 << 24);
                        }
                    chunkPending = false;
                }
                if (leaf ? i < 3 : (i < selPicks || rem >= evalOrderRemWin)) {
                    int bi = i;
                    for (int j = i + 1; j < n; j++) if (score[j] > score[bi]) bi = j;
                    std::swap(moves[i], moves[bi]); std::swap(score[i], score[bi]);
                }
                if (npre) {  // searched in the first stage
                    bool dup = false;
                    for (int q = 0; q < npre; q++) dup |= preG[q] == moves[i].gid;
                    if (dup) continue;
                }
                if (spec && s.ply == specPly) spec->beforeChild(*this, s, moves, score, i, n);
                int v;
                if (leaf) {
                    nodes++;
                    // the child's hash without building it; small per-thread cache of results
                    const Placement& pl = moves[i].pl();
                    uint64_t hc = s.h[0] ^ T.zPiece[s.stm][moves[i].piece()] ^ T.zStm;
                    for (int q = 0; q < pl.n; q++) hc ^= T.zCell[s.stm][pl.c[q]];
                    uint64_t& slot = hcache[hc & HCMASK];
                    int hv;
                    if (useHCache && (slot & ~uint64_t(3)) == (hc & ~uint64_t(3)) && slot) { hv = (int)(slot & 3) - 1; pHC++; }
                    else {
                        hv = horizonAfter(s, moves[i], pre, RED_BONUS);
                        if (hv != -9 && useHCache) slot = (hc & ~uint64_t(3)) | (uint64_t)(hv + 1);
                    }
                    if (hv != -9) { pHor[std::min(pp + 1, 15)]++; v = -hv; }
                    else { Pos c = s; c.play(moves[i]); lastGid[s.ply] = (int16_t)moves[i].gid; v = -search(c, -beta, -alpha); }
                } else {
                    int pv = -9;
                    // (SOLVER_LAZYBSLOT=2: the slot arrives with the second child)
                    if (preBase && bsPending && nChild >= 1) { computeBslot(); bsPending = false; }
                    // SOLVER_PRECUT_HI: the other direction - the opponent's upper bound here (exact one
                    // from our bounds, or the flood-free one; our move can only lower it) is below our
                    // snake after the move: the child is lost for the opponent
                    if (preBase && preCutHi) {
                        const Placement& pl = moves[i].pl();
                        int joinedH = pl.n;
                        for (int j = 0; j < s.ncomp[s.stm]; j++)
                            if (s.comp[s.stm][j] & pl.nb) joinedH += popc(s.comp[s.stm][j]);
                        const int lbUs = std::max<int>(s.L[s.stm], joinedH);
                        const int ubThem = ubOpHere + bonusOp;
                        const int bhi = ubThem < lbUs ? -1 : ubThem <= lbUs ? 0 : 1;
                        if (bhi <= -beta) { nodes++; pBound[std::min(pp + 1, 15)]++; pPreHi++; pv = -bhi; }
                    }
                    if (pv == -9 && preBase && pbs.valid && pbs.player == op) {
                        const Placement& pl = moves[i].pl();
                        bool keep = false;  // one of the opponent's best placements survives our move
                        for (int q = 0; q < pbs.n && !keep; q++) keep = !(pbs.kill[q] & pl.m);
                        // SOLVER_PRECUT2: otherwise recompute the opponent's best placement here, with
                        // our piece added to the occupied cells (its own snakes are unchanged)
                        int vC = keep ? pbs.V : -1;
                        if (!keep && preCut2)
                            vC = bestValueAny(s.own[op], s.comp[op], s.ncomp[op], s.L[op], s.avail[op], s.own[0] | s.own[1] | pl.m);
                        if (vC >= 0) {
                            const int lbC = std::max<int>(s.L[op], vC) + bonusOp;  // the child's lbMe
                            int joined = pl.n, t1 = 0, t2 = 0;
                            auto push = [&](int l) { if (l > t1) { t2 = t1; t1 = l; } else if (l > t2) t2 = l; };
                            for (int j = 0; j < s.ncomp[s.stm]; j++) {
                                const int l = popc(s.comp[s.stm][j]);
                                if (s.comp[s.stm][j] & pl.nb) joined += l; else push(l);
                            }
                            push(joined);
                            const int ubC = std::max(std::max<int>(s.L[s.stm], joined),
                                                     t1 + t2 + remainingCells(s.avail[s.stm] & ~(1 << moves[i].piece())));  // cheapUpper
                            const int blo = lbC > ubC ? 1 : lbC >= ubC ? 0 : -1;
                            if (blo >= -alpha) { nodes++; pBound[std::min(pp + 1, 15)]++; pPre++; pv = -blo; }
                        }
                    }
                    if (pv != -9) v = pv;
                    else {
                        Pos c = s;
                        c.play(moves[i]);
                        TLAP(5);
                        lastGid[s.ply] = (int16_t)moves[i].gid;
                        beforeChild();
                        v = -search(c, -beta, -alpha);
                    }
                }
                TRESET;
                if (aborted) return 0;
                if (v > best) { best = v; bestGid = moves[i].gid; }
                if (v > alpha) alpha = v;
                if (alpha >= beta) {
                    if (useCounter && s.ply > 0 && lastGid[s.ply - 1] >= 0) cmove[s.stm][lastGid[s.ply - 1]] = (int16_t)moves[i].gid;
                    if (doEval || evalDone) { eo[pp][0]++; eo[pp][i == 0 ? 1 : i < 3 ? 2 : 3]++; eo[pp][5] += i + 1; }
                    H[moves[i].gid] += 1 + i / 8 + 4 * s.ply;
                    if (killer[ply][0] != moves[i].gid) { killer[ply][1] = killer[ply][0]; killer[ply][0] = moves[i].gid; }
                    break;
                }
            }
            if ((doEval || evalDone) && alpha < beta) { eo[pp][0]++; eo[pp][4]++; eo[pp][5] += n; }
        }
        }  // alpha < beta after the first stage
        if (aborted) return 0;
        // store
        int nlo = lo, nhi = hi;
        if (best <= a0) nhi = std::min(nhi, best);
        else if (best >= b0) nlo = std::max(nlo, best);
        else nlo = nhi = best;
        uint64_t work = nodes - n0 + 1;
        int w = 63 - __builtin_clzll(work);
        tt->store(k, nlo, nhi, bestGid >= 0 ? T.gidSym[bestGid][t] : eBest, w);
        TLAP(6);
        return best;
    }

    // Returns value for the side to move, or -9 if aborted.
    // (parallel version: see parallelSolve below)
    int solve(const Pos& s, double seconds = 0, uint64_t maxNodes = UINT64_MAX, int alpha = -1, int beta = 1) {
        aborted = false;
        nodes = 0;
        nodeLimit = maxNodes;
        useDeadline = seconds > 0;
        if (useDeadline)
            deadline = std::chrono::steady_clock::now() +
                       std::chrono::microseconds((long long)(seconds * 1e6));
        for (auto& u : ubKnown) u[0] = u[1] = 1000;
        for (auto& b : bslot) b.valid = false;  // the root has no parent here
        std::memset(lastGid, -1, sizeof lastGid);
        int v = search(s, alpha, beta);
        return aborted ? -9 : v;
    }
};

// Parallel exact solve: root moves (symmetry-reduced, ordered by the learned
// evaluation) are distributed over threads, each with its own solver/TT. Stops
// as soon as a winning move is found. perMove receives the value of every root
// move for the side to move (-9 = unknown).
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
struct ParallelSolveResult {
    int value = -9;  // for the side to move, -9 unknown
    uint64_t nodes = 0;
    double seconds = 0;
    std::vector<std::pair<Move, int>> perMove;
};
// known: root moves whose value is already known (e.g. from an interrupted run) - skipped.
// onResult: called (under a lock) for every newly solved root move.
// called (under the lock, before onResult) with a solved root move and the opponent's reply that
// decided it, when known (YB scheduler): for refuted moves the refutation, i.e. the winning strategy
inline std::function<void(Move, Move)> g_rootReply;
inline ParallelSolveResult parallelSolve(const Pos& root, double seconds, int threads, int ttLog2 = 22,
                                         const std::vector<std::pair<Move, int>>& known = {},
                                         std::function<void(Move, int, uint64_t, double)> onResult = nullptr) {
    auto t0 = std::chrono::steady_clock::now();
    ParallelSolveResult res;
    std::vector<Move> all, mv;
    root.genMoves(all);
    if (all.empty()) all.push_back(Move{});
    std::vector<uint64_t> seen;
    for (Move m : all) {
        Pos c = root; c.play(m);
        int t; uint64_t k = c.key(t);
        if (std::find(seen.begin(), seen.end(), k) == seen.end()) { seen.push_back(k); mv.push_back(m); }
    }
    std::vector<double> ord(mv.size());
    for (size_t i = 0; i < mv.size(); i++) {
        Pos c = root; c.play(mv[i]);
        ord[i] = c.over() ? -Solver::terminalValue(c) * 1000.0 : -learnedLogit(c);
    }
    std::vector<size_t> idx(mv.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return ord[a] > ord[b]; });
    // SOLVE_ROOTORDER=<file>: the root moves named in the file (first token of each line) go first,
    // in file order, the rest after them in eval order. For a ranking by a deeper search: the only
    // winning reply to 1:60 4B:16,27,28,39 was 1409th of 1589 by the 1-ply eval, 24th by depth-3 alpha-beta
    if (const char* fn = std::getenv("SOLVE_ROOTORDER")) {
        if (FILE* f = std::fopen(fn, "r")) {
            std::vector<size_t> front;
            std::vector<char> used(mv.size(), 0);
            char line[512];
            while (std::fgets(line, sizeof line, f)) {
                char tok[256];
                Move m;
                if (std::sscanf(line, "%255s", tok) != 1 || !parseMove(tok, m)) continue;
                Pos c = root; c.play(m);
                int t; const uint64_t k = c.key(t);
                for (size_t i = 0; i < mv.size(); i++)
                    if (!used[i] && seen[i] == k) { used[i] = 1; front.push_back(i); break; }
            }
            std::fclose(f);
            for (size_t i : idx) if (!used[i]) front.push_back(i);
            idx.swap(front);
        }
    }
    // SOLVE_ROOTMIX=1: take the root moves alternately from both ends of the eval order. In the
    // hardest "find the winning move" problems the only winning move was ranked near the bottom
    // (918th of 1032, 637th of 881: quiet new snakes the eval dislikes); interleaving bounds the
    // wait by twice the distance from the nearer end
    // SOLVE_ROOTMIX_K=K: the first K moves in plain eval order (easy problems end there), only then
    // the interleaving
    if (envInt("SOLVE_ROOTMIX", 0) == 1 && idx.size() > 2) {
        std::vector<size_t> mix;
        mix.reserve(idx.size());
        const size_t K = std::min<size_t>(idx.size(), (size_t)envInt("SOLVE_ROOTMIX_K", 0));
        for (size_t i = 0; i < K; i++) mix.push_back(idx[i]);
        for (size_t a = K, b = idx.size(); a < b;) {
            mix.push_back(idx[a++]);
            if (a < b) mix.push_back(idx[--b]);
        }
        idx.swap(mix);
    }
    std::vector<int> val(mv.size(), -9);
    std::atomic<bool> stop{false};
    for (auto& [m, v] : known)
        for (size_t i = 0; i < mv.size(); i++)
            if (mv[i] == m) { val[i] = v; if (v == 1) stop = true; }
    std::atomic<uint64_t> nodes{0};
    std::mutex mu;
    threads = std::max(1, std::min<int>(threads, (int)mv.size()));
    std::vector<std::thread> th;
    // SOLVER_SHARED=<log2 entries>: one transposition table shared by all threads
    // production null-window root solves (SOLVE_WINONLY / SOLVE_DRAWTEST) share one 2^26-entry TT
    // by default: the threads work on sibling subtrees with many transpositions (-14% on real units)
    const bool nullRoot = envInt("SOLVE_WINONLY", 0) != 0 || envInt("SOLVE_DRAWTEST", 0) != 0;
    int sharedLog2 = envInt("SOLVER_SHARED", nullRoot && threads > 1 ? 26 : 0);
    // SOLVE_PERSIST=1: keep the solvers (and the shared TT) between calls - a batch of related
    // problems (e.g. all replies to one move) then reuses each other's results
    static std::shared_ptr<LTT> persistShared;
    static std::vector<std::unique_ptr<Solver>> persistSolvers;
    const bool persist = envInt("SOLVE_PERSIST", 0) != 0;
    std::shared_ptr<LTT> shared = persist && persistShared ? persistShared
                                : sharedLog2 ? std::make_shared<LTT>(sharedLog2 - 1) : nullptr;
    if (persist && sharedLog2 && !persistShared) persistShared = shared;
    const bool winOnly = envInt("SOLVE_WINONLY", 0) != 0;
    // SOLVE_MAXROOT=N: only the first N root moves in order (production-like benchmark)
    const size_t maxRoot = (size_t)envInt("SOLVE_MAXROOT", 1 << 30);
    std::atomic<size_t> next{(size_t)envInt("SOLVE_ROOTFROM", 0)};  // SOLVE_ROOTFROM=a: start at root move a
    const bool drawTest = envInt("SOLVE_DRAWTEST", 0) != 0;
    // SOLVE_SPLIT=1 (with SOLVE_WINONLY or SOLVE_DRAWTEST): two phases. Phase 1 is the usual
    // one-thread-per-root-move loop, but with a node budget per move (SOLVE_BUDGET, millions,
    // default 30); moves over budget are deferred. Phase 2 takes the deferred moves in root order
    // and proves each with all threads: the opponent's replies are split over the threads (all of
    // them must fail for the opponent), the first reply that holds refutes the move. Otherwise the
    // proof of the one good move runs on a single thread while the others refute hopeless moves.
    const bool split = (winOnly || drawTest) && envInt("SOLVE_SPLIT", 2) != 0;  // default: the task queue (2)
    const uint64_t budget = split ? (uint64_t)envInt("SOLVE_BUDGET", 30) * 1000000ull : UINT64_MAX;
    std::vector<size_t> deferred;
    std::atomic<bool> pause{false};  // a move went over budget: prove it with all threads first
    std::vector<std::unique_ptr<Solver>> solvers(threads);
    for (int w = 0; w < threads; w++) {
        if (persist && w < (int)persistSolvers.size() && persistSolvers[w]) solvers[w] = std::move(persistSolvers[w]);
        else solvers[w].reset(shared ? new Solver(shared) : new Solver(ttLog2));
    }
    // SOLVE_LEVELB=<file>: the strategy one level below every decided root move, read from the TT
    // (for a solution file with instant play). Binary records, appended (under the lock):
    //   uint16 'B', uint8 type, int16 root move gid, int16 reply gid (-1: none), uint16 n, n x int16
    //   type 1 - root move refuted by `reply`: for each of our moves b after (move, reply) in
    //            genMoves order, the opponent's best answer (-1 unknown, -2 game over);
    //   type 2 - root move holds: for each opponent reply r after the move (genMoves order), our
    //            best answer to it.
    // Entries missing from the TT (decided by bounds, or replaced) are cheap to recompute.
    FILE* lvb = nullptr;
    if (const char* f = std::getenv("SOLVE_LEVELB")) lvb = std::fopen(f, "ab");
    auto ttBest = [&](const Pos& p) -> int16_t {
        if (p.over()) return -2;
        int t, lo, hi, b;
        const uint64_t k = p.key(t);
        LTT& tt = shared ? *shared : *solvers[0]->tt;
        if (tt.probe(k, lo, hi, b) && b >= 0) return (int16_t)T.gidSym[b][T.inv[t]];
        return -1;
    };
    auto levelB = [&](Move m, int type, Move rep) {  // under the lock
        if (!lvb) return;
        Pos p = root; p.play(m);
        if (type == 1) { if (rep.isPass() || rep.gid < 0) return; p.play(rep); }
        std::vector<Move> ms;
        p.genMoves(ms);
        std::vector<int16_t> e(ms.size());
        for (size_t i = 0; i < ms.size(); i++) { Pos q = p; q.play(ms[i]); e[i] = ttBest(q); }
        const uint16_t tag = 'B', n = (uint16_t)ms.size();
        const uint8_t ty = (uint8_t)type;
        const int16_t mg = m.gid, rg = type == 1 ? rep.gid : (int16_t)-1;
        std::fwrite(&tag, 2, 1, lvb); std::fwrite(&ty, 1, 1, lvb); std::fwrite(&mg, 2, 1, lvb);
        std::fwrite(&rg, 2, 1, lvb); std::fwrite(&n, 2, 1, lvb);
        if (n) std::fwrite(e.data(), 2, n, lvb);
        std::fflush(lvb);
    };
    struct CloseLvb { FILE*& f; ~CloseLvb() { if (f) std::fclose(f); } } closeLvb{lvb};
    auto work = [&](int w) {
        Solver& S = *solvers[w];
        S.stop = &stop;
        for (;;) {
            if (stop || pause) break;
            size_t k = next++;
            if (k >= idx.size() || k >= maxRoot) break;
            size_t i = idx[k];
            if (val[i] != -9) continue;  // known from a previous run
            Pos c = root; c.play(mv[i]);
            auto ts = std::chrono::steady_clock::now();
            double left = seconds - std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (left <= 0) break;
            // SOLVE_WINONLY=1: only "is this root move a win?" (child searched with the null window
            // (-1, 0)); a non-win comes back as 0 = "draw or loss", a proven loss as -1.
            // SOLVE_DRAWTEST=1: only "does this root move reach at least a draw?" (child window
            // (0, 1)); stops at the first such move. 0 = "draw or win", -1 = proven loss.
            int v = winOnly ? S.solve(c, left, budget, -1, 0)
                  : drawTest ? S.solve(c, left, budget, 0, 1) : S.solve(c, left);
            nodes += S.nodes;
            if (v == -9) {
                if (split && !stop && S.nodes > budget) { std::lock_guard<std::mutex> lk(mu); deferred.push_back(k); pause = true; }
                continue;
            }
            std::lock_guard<std::mutex> lk(mu);
            val[i] = -v;
            if (-v == 1 || (drawTest && -v >= 0)) stop = true;
            if (onResult) onResult(mv[i], -v, S.nodes, std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count());
        }
    };
    // SOLVE_SPLIT=2: one task queue, "young brothers wait". Tasks in priority order (earlier root
    // moves first): (a) the remaining replies of a move whose eldest reply did not refute it, one per
    // thread; (b) the eldest (the opponent's most promising) reply of a move that went over budget,
    // on one thread; (c) the next root move with the node budget. Hopeless moves are refuted by one
    // thread as before, the proof of a good move is spread over all threads, and no thread waits
    // while there is work.
    if (split && envInt("SOLVE_SPLIT", 2) == 2) {
        struct Cand {
            size_t k, i;
            int state = 1;  // 1 eldest pending, 2 eldest running, 3 fan-out, 4 done
            std::vector<Move> rep;
            size_t rnext = 1;
            int outstanding = 0;
            bool refuted = false;
            std::atomic<bool> cstop{false};
            uint64_t cn = 0;
            std::chrono::steady_clock::time_point ts;
            int refIdx = -1;  // the reply that refuted the move (index in rep)
        };
        // SOLVE_DEFER_KILLERS=1: a deferred move's replies start with the killers of the thread that
        // deferred it - what its budgeted search was working on (the work is in the shared TT)
        const bool deferKillers = envInt("SOLVE_DEFER_KILLERS", 1) != 0;
        const bool ybDebug = envInt("SOLVE_YBDEBUG", 0) != 0;
        // SOLVE_DEFER_REFUT=1: count for every reply how many root moves it refuted so far; a deferred
        // move's replies start with the most frequent refutations (siblings are refuted by the same
        // few replies, which plain eval order put at #6-#53 of ~650)
        const bool deferRefut = envInt("SOLVE_DEFER_REFUT", 1) != 0;
        // SOLVE_ROOTREF=1: the subproblems' searches also try the two most frequent refutations first
        const bool rootRef = envInt("SOLVE_ROOTREF", 1) != 0;
        std::unique_ptr<std::atomic<int>[]> refCount(new std::atomic<int>[T.NGID]);
        for (int g = 0; g < T.NGID; g++) refCount[g].store(0, std::memory_order_relaxed);
        auto noteRefut = [&](Move rep) { if (!rep.isPass() && rep.gid >= 0 && rep.gid < T.NGID) refCount[rep.gid].fetch_add(1, std::memory_order_relaxed); };
        const int need = winOnly ? 1 : 0;  // the side to move needs >= need after every reply
        std::vector<std::unique_ptr<Cand>> active;  // in root order
        std::condition_variable cv;
        int running = 0;
        auto finishCand = [&](Cand& C, bool won) {  // under mu
            const bool fanned = C.state == 3;
            C.state = 4;
            val[C.i] = won ? need : (winOnly ? 0 : -1);
            if (won) { stop = true; for (auto& a : active) a->cstop = true; }
            if (ybDebug)
                fprintf(stderr, "YB %-20s %s %s refuting reply #%d of %zu (%s), %llu nodes\n", moveStr(mv[C.i]).c_str(),
                        won ? "holds  " : "refuted", fanned ? "fan-out" : "eldest ", C.refIdx, C.rep.size(),
                        C.refIdx >= 0 ? moveStr(C.rep[C.refIdx]).c_str() : "-", (unsigned long long)C.cn);
            if (C.refIdx >= 0) noteRefut(C.rep[C.refIdx]);
            if (g_rootReply && C.refIdx >= 0) g_rootReply(mv[C.i], C.rep[C.refIdx]);
            if (won) levelB(mv[C.i], 2, Move{});
            else if (C.refIdx >= 0) levelB(mv[C.i], 1, C.rep[C.refIdx]);
            if (onResult) onResult(mv[C.i], val[C.i], C.cn, std::chrono::duration<double>(std::chrono::steady_clock::now() - C.ts).count());
        };
        auto worker = [&](int w) {
            Solver& S = *solvers[w];
            if (rootRef) { S.gRef = refCount.get(); S.gRefPly = root.ply + 1; }
            struct ResetRef { Solver& S; ~ResetRef() { S.gRef = nullptr; } } resetRef{S};
            std::unique_lock<std::mutex> lk(mu);
            for (;;) {
                if (stop) break;
                if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= seconds) break;
                // pick a task
                int type = 0; Cand* C = nullptr; size_t j = 0, k = 0;
                for (auto& a : active)
                    if (a->state == 3 && a->rnext < a->rep.size() && !a->refuted) { C = a.get(); j = C->rnext++; C->outstanding++; type = 3; break; }
                if (!type)
                    for (auto& a : active)
                        if (a->state == 1) { C = a.get(); C->state = 2; type = 2; break; }
                if (!type && next < idx.size() && next < maxRoot) {
                    k = next++;
                    if (val[idx[k]] != -9) continue;  // known from a previous run
                    type = 1;
                }
                if (!type) {
                    if (running == 0) break;  // nothing left anywhere
                    cv.wait(lk);
                    continue;
                }
                running++;
                lk.unlock();
                const double left = seconds - std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (type == 1) {
                    const size_t i = idx[k];
                    Pos c = root; c.play(mv[i]);
                    auto ts = std::chrono::steady_clock::now();
                    S.stop = &stop;
                    int v = left <= 0 ? -9 : winOnly ? S.solve(c, left, budget, -1, 0) : S.solve(c, left, budget, 0, 1);
                    nodes += S.nodes;
                    std::unique_ptr<Cand> nc;
                    if (v == -9 && !stop && S.nodes > budget) {
                        nc.reset(new Cand);
                        nc->k = k; nc->i = i; nc->ts = ts; nc->cn = S.nodes;
                        std::vector<Move> rall; c.genMoves(rall);
                        if (rall.empty()) rall.push_back(Move{});
                        std::vector<uint64_t> rseen;
                        std::vector<std::pair<double, Move>> ro;
                        for (Move m : rall) {
                            Pos g = c; g.play(m);
                            int t; uint64_t kk = g.key(t);
                            if (std::find(rseen.begin(), rseen.end(), kk) != rseen.end()) continue;
                            rseen.push_back(kk);
                            ro.push_back({g.over() ? -Solver::terminalValue(g) * 1000.0 : -learnedLogit(g), m});
                        }
                        std::stable_sort(ro.begin(), ro.end(), [](auto& a, auto& b) { return a.first > b.first; });
                        for (auto& r : ro) nc->rep.push_back(r.second);
                        if (deferRefut) {  // replies that refuted more root moves first (then eval order)
                            std::stable_sort(nc->rep.begin(), nc->rep.end(), [&](Move a, Move b) {
                                return refCount[a.gid].load(std::memory_order_relaxed) > refCount[b.gid].load(std::memory_order_relaxed);
                            });
                        }
                        if (deferKillers) {
                            const int kp = std::min<int>(c.ply, 31);
                            for (int q = 1; q >= 0; q--) {  // killer 0 ends up first
                                const int g = S.killer[kp][q];
                                if (g < 0) continue;
                                for (size_t a = 0; a < nc->rep.size(); a++)
                                    if (nc->rep[a].gid == g) { std::rotate(nc->rep.begin(), nc->rep.begin() + a, nc->rep.begin() + a + 1); break; }
                            }
                        }
                    }
                    lk.lock();
                    running--;
                    if (nc) {
                        auto pos = std::find_if(active.begin(), active.end(), [&](auto& a) { return a->k > nc->k; });
                        active.insert(pos, std::move(nc));
                    } else if (v != -9) {
                        val[i] = -v;
                        if (-v == 1 || (drawTest && -v >= 0)) { stop = true; for (auto& a : active) a->cstop = true; }
                        const bool rootLost = -v < need || (drawTest && -v < 0);
                        if (g_rootReply || deferRefut || rootRef || lvb) {  // the opponent's best reply, from the TT entry of the child
                            int t, lo, hi, b;
                            const uint64_t kk = c.key(t);
                            if (S.tt->probe(kk, lo, hi, b) && b >= 0) {
                                const Move rep{(int16_t)T.gidSym[b][T.inv[t]]};
                                if (rootLost) noteRefut(rep);
                                if (g_rootReply) g_rootReply(mv[i], rep);
                                if (rootLost) levelB(mv[i], 1, rep);
                            }
                        }
                        if (!rootLost) levelB(mv[i], 2, Move{});
                        if (onResult) onResult(mv[i], -v, S.nodes, std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count());
                    }
                } else {
                    // a reply of a deferred move: the side to move needs >= need after it
                    Pos g = root; g.play(mv[C->i]); g.play(C->rep[j]);
                    S.stop = &C->cstop;
                    int v = g.over() ? Solver::terminalValue(g) : left <= 0 ? -9 : S.solve(g, left, UINT64_MAX, need - 1, need);
                    nodes += S.nodes;
                    S.stop = &stop;
                    lk.lock();
                    running--;
                    C->cn += S.nodes;
                    if (type == 3) C->outstanding--;
                    if (C->state != 4 && !stop) {
                        if (v == -9) {
                            // aborted: refuted by another reply (handled below) or out of time
                        } else if (v < need) {
                            if (!C->refuted) C->refIdx = (int)j;
                            C->refuted = true; C->cstop = true;
                            if (C->outstanding == 0) finishCand(*C, false);
                        } else if (type == 2) {
                            if (C->rep.size() == 1) finishCand(*C, true);
                            else C->state = 3;
                        } else if (C->rnext >= C->rep.size() && C->outstanding == 0 && !C->refuted) finishCand(*C, true);
                    }
                    if (C->state == 3 && C->refuted && C->outstanding == 0) finishCand(*C, false);
                }
                cv.notify_all();
            }
            cv.notify_all();
        };
        for (int w = 0; w < threads; w++) th.emplace_back(worker, w);
        for (auto& t : th) t.join();
        th.clear();
    } else
    for (;;) {  // phase 1 until a move is deferred, then phase 2 for it (SOLVE_SPLIT), and again
    pause = false;
    deferred.clear();
    th.clear();
    if (threads == 1) work(0);  // inline, so it also runs where there are no threads (WebAssembly)
    else {
        for (int w = 0; w < threads; w++) th.emplace_back(work, w);
        for (auto& t : th) t.join();
    }
    if (!split || stop || deferred.empty()) break;
    {
        std::sort(deferred.begin(), deferred.end());
        const int need = winOnly ? 1 : 0;  // the side to move needs >= need after every reply
        for (size_t k : deferred) {
            if (stop) break;
            double left = seconds - std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (left <= 0) break;
            const size_t i = idx[k];
            Pos c = root; c.play(mv[i]);  // the opponent to move
            std::vector<Move> rep, rall;
            c.genMoves(rall);
            if (rall.empty()) rall.push_back(Move{});
            std::vector<uint64_t> rseen;
            for (Move m : rall) {
                Pos g = c; g.play(m);
                int t; uint64_t kk = g.key(t);
                if (std::find(rseen.begin(), rseen.end(), kk) == rseen.end()) { rseen.push_back(kk); rep.push_back(m); }
            }
            // the opponent's most promising replies first: a refutation ends the move early
            std::vector<std::pair<double, Move>> ro;
            for (Move m : rep) { Pos g = c; g.play(m); ro.push_back({g.over() ? -Solver::terminalValue(g) * 1000.0 : -learnedLogit(g), m}); }
            std::stable_sort(ro.begin(), ro.end(), [](auto& a, auto& b) { return a.first > b.first; });
            auto ts = std::chrono::steady_clock::now();
            std::atomic<size_t> rnext{0};
            std::atomic<bool> refuted{false}, candStop{false}, timedOut{false};
            std::atomic<uint64_t> cn{0};
            auto helper = [&](int w) {
                Solver& S = *solvers[w];
                S.stop = &candStop;
                for (;;) {
                    size_t j = rnext++;
                    if (j >= ro.size() || candStop) break;
                    Pos g = c; g.play(ro[j].second);  // the side to move again
                    double l2 = seconds - std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    if (l2 <= 0) { timedOut = true; candStop = true; break; }
                    int v = g.over() ? Solver::terminalValue(g) : S.solve(g, l2, UINT64_MAX, need - 1, need);
                    cn += S.nodes;
                    if (v == -9) { if (!candStop) { timedOut = true; candStop = true; } break; }
                    if (v < need) { refuted = true; candStop = true; break; }
                }
                S.stop = &stop;
            };
            std::vector<std::thread> hth;
            for (int w = 0; w < threads; w++) hth.emplace_back(helper, w);
            for (auto& t : hth) t.join();
            nodes += cn;
            if (timedOut && !refuted) break;
            std::lock_guard<std::mutex> lk(mu);
            val[i] = refuted ? (winOnly ? 0 : -1) : need;
            if (!refuted) stop = true;
            if (onResult) onResult(mv[i], val[i], cn, std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count());
        }
    }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() >= seconds) break;
    }
    if (persist) {
        if ((int)persistSolvers.size() < threads) persistSolvers.resize(threads);
        for (int w = 0; w < threads; w++) persistSolvers[w] = std::move(solvers[w]);
    }
    int best = -2;
    bool allKnown = true;
    for (size_t i = 0; i < mv.size(); i++) {
        res.perMove.push_back({mv[i], val[i]});
        if (val[i] == -9) allKnown = false; else best = std::max(best, val[i]);
    }
    if (best == 1 || (allKnown && best > -2)) res.value = best;
    res.nodes = nodes;
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}
