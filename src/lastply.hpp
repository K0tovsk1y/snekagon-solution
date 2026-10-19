// Exact evaluation of the last two plies without search.
//
// When the side to move has one piece left and the opponent at most one, the game value
// follows from bitboard arithmetic:
//  - a player's final longest snake after placing its last piece p is
//    max(L, |p| + sum of the (at most two) own snakes the placement touches): a piece cell
//    with an outside own neighbour must be an end of the piece, so at most two snakes join;
//  - the opponent's moves only remove cells and isolation, so they can only shrink a
//    player's options (monotone): a growth placement stays legal as long as its cells stay
//    empty, a new-snake placement as long as its cells and their neighbours stay empty.
#pragma once
#include "core.hpp"
#ifdef HSTATS
inline uint64_t HST[8];  // exits: 0 lo==hi 1 lo==1 2 after free 3 after blocking 4 entries 5 blocking candidates 6 calls(1,1)
#define HS(i, n) (HST[i] += (n))
#else
#define HS(i, n)
#endif

inline void genCtxFor(BB me, BB occ, GenCtx& g) {
    BB E = T.VALID & ~occ;
    Shape sh = shapeOf(me);
    g.iso = E & ~expand(occ);
    BB base = E & ~expand(sh.full);
    g.allow[0] = base & ~sh.ge3 & ~sh.tri2;
    g.allow[1] = base & ~sh.ge2;
    g.allow[2] = base & ~sh.ge1;
    g.adjOwn = E & sh.ge1;
    g.me = sh.own;
    g.meDeg1 = sh.own & sh.ge1;
    g.ends = sh.ends;
}

inline int onlyPiece(uint8_t av) { return __builtin_ctz(av); }

// Live snakes of a player, longest first. A piece touching a snake attaches to one of its
// ends with a piece end cell, which lies in allow[0] | allow[1]; snakes without such a
// cell next to an end can never grow.
struct SnakeSet {
    BB nb[8];   // the cells next to the snake's ends where a piece may touch it
    int len[8];
    int n;
    void init(const BB* comp, int ncomp, const GenCtx& g) {
        BB ok = g.allow[0] | g.allow[1];
        n = 0;
        for (int j = 0; j < ncomp; j++) {
            BB e = comp[j] & g.ends, nbj = 0;
            while (e) { nbj |= T.NB[lsb(e)]; e &= e - 1; }
            nbj &= ok;
            if (!nbj) continue;
            int l = popc(comp[j]), i = n++;
            while (i > 0 && len[i - 1] < l) { nb[i] = nb[i - 1]; len[i] = len[i - 1]; i--; }
            nb[i] = nbj; len[i] = l;
        }
    }
    int maxJoin(int sp) const { return sp + (n > 0 ? len[0] : 0) + (n > 1 ? len[1] : 0); }
};

// best value (final length) among anchors lg of orientation o; fills tj = anchors touching each snake
inline int bestOfAnchors(int o, const SnakeSet& S, BB lg, int sp, int base, BB* tj) {
    int v = base;
    for (int j = 0; j < S.n; j++) {
        tj[j] = olTouch(o, S.nb[j]) & lg;
        if (tj[j]) v = std::max(v, sp + S.len[j]);
    }
    for (int i = 0; i < S.n; i++)
        if (tj[i])
            for (int j = i + 1; j < S.n; j++)
                if (tj[i] & tj[j]) { v = std::max(v, sp + S.len[i] + S.len[j]); break; }  // len sorted desc
    return v;
}

// Best final longest snake of a player (cells me, snakes comp, longest L) placing its last
// piece p now with occupancy occ.
inline int bestValue(BB me, const BB* comp, int ncomp, int L, int p, BB occ) {
    GenCtx g;
    genCtxFor(me, occ, g);
    SnakeSet S;
    S.init(comp, ncomp, g);
    const int sp = PIECE_CELLS[p];
    const bool fresh = sp > L;  // a new snake only matters if it beats the longest one
    const int top = std::max(L, S.maxJoin(sp));
    if (top <= L) return L;
    int best = L;
    BB tj[8];
    for (int o = T.orBegin[p]; o < T.orBegin[p + 1]; o++) {
        BB lg = fresh ? olFull(o, g) : olGrow(o, g);
        if (!lg) continue;
        best = std::max(best, bestOfAnchors(o, S, lg, sp, fresh ? sp : L, tj));
        if (best >= top) break;
    }
    return best;
}

// best value over all pieces in av (one context for all; larger pieces first, a piece is
// skipped when even joining the two longest snakes cannot beat the best so far)
inline int bestValueAny(BB me, const BB* comp, int ncomp, int L, uint8_t av, BB occ) {
    GenCtx g;
    genCtxFor(me, occ, g);
    SnakeSet S;
    S.init(comp, ncomp, g);
    int best = L;
    BB tj[8];
    for (int p = NPIECE - 1; p >= 0; p--) {
        if (!(av >> p & 1)) continue;
        const int sp = PIECE_CELLS[p];
        const bool fresh = sp > best;
        const int top = std::max(best, S.maxJoin(sp));
        if (top <= best) continue;
        for (int o = T.orBegin[p]; o < T.orBegin[p + 1]; o++) {
            BB lg = fresh ? olFull(o, g) : olGrow(o, g);
            if (!lg) continue;
            best = std::max(best, bestOfAnchors(o, S, lg, sp, fresh ? sp : best, tj));
            if (best >= top) break;
        }
    }
    return best;
}

// bestValueAny that also returns a placement achieving the value (gid -1: nothing beats L)
inline int bestPlacementAny(BB me, const BB* comp, int ncomp, int L, uint8_t av, BB occ, int& gid) {
    GenCtx g;
    genCtxFor(me, occ, g);
    SnakeSet S;
    S.init(comp, ncomp, g);
    int best = L;
    gid = -1;
    BB tj[8];
    for (int p = NPIECE - 1; p >= 0; p--) {
        if (!(av >> p & 1)) continue;
        const int sp = PIECE_CELLS[p];
        const bool fresh = sp > best;
        const int top = std::max(best, S.maxJoin(sp));
        if (top <= best) continue;
        for (int o = T.orBegin[p]; o < T.orBegin[p + 1]; o++) {
            BB lg = fresh ? olFull(o, g) : olGrow(o, g);
            if (!lg) continue;
            BB anyT = 0;
            for (int j = 0; j < S.n; j++) { tj[j] = olTouch(o, S.nb[j]) & lg; anyT |= tj[j]; }
            // best anchor of this orientation
            int v = -1, a = -1;
            if (fresh && (lg & ~anyT)) { v = sp; a = lsb(lg & ~anyT); }
            for (int j = 0; j < S.n; j++) if (tj[j] && sp + S.len[j] > v) { v = sp + S.len[j]; a = lsb(tj[j]); }
            for (int i = 0; i < S.n; i++)
                if (tj[i])
                    for (int j = i + 1; j < S.n; j++)
                        if ((tj[i] & tj[j]) && sp + S.len[i] + S.len[j] > v) { v = sp + S.len[i] + S.len[j]; a = lsb(tj[i] & tj[j]); }
            if (v > best) { best = v; gid = T.OR[o].gidAt[a]; }
            if (best >= top) break;
        }
    }
    return best;
}

// The best value V of a player's single placements (> L, else returns L with n = 0) and up to
// 4 placements achieving it, with the cells an opponent piece must touch to spoil each (its
// cells; for a new snake also its neighbours).
inline int bestPlacementsKill(BB me, const BB* comp, int ncomp, int L, uint8_t av, BB occ, BB* kill, int16_t* gids, int& n) {
    GenCtx g;
    genCtxFor(me, occ, g);
    SnakeSet S;
    S.init(comp, ncomp, g);
    int best = L;
    n = 0;
    BB tj[8];
    for (int p = NPIECE - 1; p >= 0; p--) {
        if (!(av >> p & 1)) continue;
        const int sp = PIECE_CELLS[p];
        const bool fresh = sp > L;
        if (std::max(sp, S.maxJoin(sp)) < std::max(best, L + 1)) continue;
        for (int o = T.orBegin[p]; o < T.orBegin[p + 1]; o++) {
            BB lg = fresh ? olFull(o, g) : olGrow(o, g);
            if (!lg) continue;
            const Orient& od = T.OR[o];
            BB anyT = 0;
            for (int j = 0; j < S.n; j++) { tj[j] = olTouch(o, S.nb[j]) & lg; anyT |= tj[j]; }
            auto add = [&](int cell, int v, bool isFresh) {
                if (v < best || v <= L) return;
                if (v > best) { best = v; n = 0; }
                if (n == 4) return;
                const Placement* pl = T.plOfGid[od.gidAt[cell]];
                kill[n] = isFresh ? (pl->m | pl->nb) : pl->m;
                gids[n] = (int16_t)od.gidAt[cell];
                n++;
            };
            for (BB a = anyT; a; a &= a - 1) {
                int cell = lsb(a), v = sp;
                for (int j = 0; j < S.n; j++) if (tj[j] >> cell & 1) v += S.len[j];
                add(cell, v, false);
            }
            if (fresh && sp >= best)
                for (BB a = lg & ~anyT; a; a &= a - 1) add(lsb(a), sp, true);
        }
    }
    return best;
}

// up to k placements of distinct orientations with the highest values (best first) beating L
inline int topPlacements(BB me, const BB* comp, int ncomp, int L, uint8_t av, BB occ, int* gids, int k) {
    if (k <= 1) {
        int gid;
        bestPlacementAny(me, comp, ncomp, L, av, occ, gid);
        gids[0] = gid;
        return gid >= 0 ? 1 : 0;
    }
    GenCtx g;
    genCtxFor(me, occ, g);
    SnakeSet S;
    S.init(comp, ncomp, g);
    int vs[4], n = 0;
    k = std::min(k, 4);
    BB tj[8];
    for (int p = NPIECE - 1; p >= 0; p--) {
        if (!(av >> p & 1)) continue;
        const int sp = PIECE_CELLS[p];
        const bool fresh = sp > L;
        if (std::max(sp, S.maxJoin(sp)) <= L) continue;
        for (int o = T.orBegin[p]; o < T.orBegin[p + 1]; o++) {
            BB lg = fresh ? olFull(o, g) : olGrow(o, g);
            if (!lg) continue;
            BB anyT = 0;
            for (int j = 0; j < S.n; j++) { tj[j] = olTouch(o, S.nb[j]) & lg; anyT |= tj[j]; }
            int v = -1, a = -1;
            if (fresh && (lg & ~anyT)) { v = sp; a = lsb(lg & ~anyT); }
            for (int j = 0; j < S.n; j++) if (tj[j] && sp + S.len[j] > v) { v = sp + S.len[j]; a = lsb(tj[j]); }
            for (int i = 0; i < S.n; i++)
                if (tj[i])
                    for (int j = i + 1; j < S.n; j++)
                        if ((tj[i] & tj[j]) && sp + S.len[i] + S.len[j] > v) { v = sp + S.len[i] + S.len[j]; a = lsb(tj[i] & tj[j]); }
            if (v <= L) continue;
            // insert into the top-k list
            int pos = n < k ? n++ : k;
            if (pos == k && v <= vs[k - 1]) continue;
            if (pos == k) pos = k - 1;
            while (pos > 0 && vs[pos - 1] < v) { vs[pos] = vs[pos - 1]; gids[pos] = gids[pos - 1]; pos--; }
            vs[pos] = v; gids[pos] = T.OR[o].gidAt[a];
        }
    }
    return n;
}

// A sure lower bound for player c who is NOT to move: the side to move places one piece (at
// most 4 cells within distance 3 of each other) before c moves. If two of c's placements worth
// >= v are >= 4 cells apart (a new snake: counting its neighbours, which must stay empty), no
// single piece can spoil both, and c ends with >= v. Returns max(L, best such v); only values
// >= need are looked for (need <= L: all).
inline int miaiLower(const Pos& s, int c, int need = 0) {
    const int L = s.L[c];
    const int floorV = std::max(L + 1, need);
    const uint8_t av = s.avail[c];
    if (!av) return L;
    const BB occ = s.own[0] | s.own[1];
    GenCtx g;
    genCtxFor(s.own[c], occ, g);
    SnakeSet S;
    S.init(s.comp[c], s.ncomp[c], g);
    struct E { BB kill; int v; };
    E e[64];
    int n = 0;
    BB tj[8];
    for (int p = NPIECE - 1; p >= 0; p--) {
        if (!(av >> p & 1)) continue;
        const int sp = PIECE_CELLS[p];
        if (std::max(sp, S.maxJoin(sp)) < floorV) continue;
        const bool fresh = sp > L;
        for (int o = T.orBegin[p]; o < T.orBegin[p + 1]; o++) {
            BB lg = fresh ? olFull(o, g) : olGrow(o, g);
            if (!lg) continue;
            const Orient& od = T.OR[o];
            BB anyT = 0;
            for (int j = 0; j < S.n; j++) { tj[j] = olTouch(o, S.nb[j]) & lg; anyT |= tj[j]; }
            for (BB a = anyT; a; a &= a - 1) {
                int cell = lsb(a), v = sp;
                for (int j = 0; j < S.n; j++) if (tj[j] >> cell & 1) v += S.len[j];
                if (v < floorV) continue;
                if (n == 64) goto full;
                e[n++] = {T.plOfGid[od.gidAt[cell]]->m, v};
            }
            if (fresh && sp >= floorV)
                for (BB a = lg & ~anyT; a; a &= a - 1) {
                    if (n == 64) goto full;
                    const Placement* pl = T.plOfGid[od.gidAt[lsb(a)]];
                    e[n++] = {pl->m | pl->nb, sp};
                }
        }
    }
full:
    if (n < 2) return L;
    // best first
    for (int i = 1; i < n; i++) {
        E x = e[i];
        int j = i;
        while (j > 0 && e[j - 1].v < x.v) { e[j] = e[j - 1]; j--; }
        e[j] = x;
    }
    // e[i] can be paired with an earlier (better or equal) e[j] far from it: value e[i].v
    for (int i = 1; i < n; i++) {
        if (e[i].v <= L) break;
        BB near3 = expand(expand(expand(e[i].kill)));
        for (int j = 0; j < i; j++)
            if (!(near3 & e[j].kill)) return e[i].v;
    }
    return L;
}

inline int lengthAfter(const SnakeSet& S, int L, const Placement& pl, int sp) {
    int v = sp;
    for (int j = 0; j < S.n; j++) if (S.nb[j] & pl.m) v += S.len[j];
    return std::max(L, v);
}

// outcome for player x with final lengths lx, ly (redBonus = extra cells counted for red)
inline int outcomeFor(int x, int lx, int ly, int redBonus) {
    int d = lx - ly + (x == 0 ? redBonus : -redBonus);
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

// Exact value for the side to move when it has <= 1 piece and the opponent <= 1 piece
// (not both 0). Returns -9 when not applicable (falls back to search).
inline int horizonValue(const Pos& s, int redBonus) {
    int X = s.stm, Y = X ^ 1;
    int nX = pieceCount(s.avail[X]), nY = pieceCount(s.avail[Y]);
    if (nX > 1 || nY > 1 || s.ply == 0) return -9;
    BB occ = s.own[0] | s.own[1];
    if (nX == 0) {
        if (nY == 0) return -9;
        // X passes, Y places its last piece (or cannot), game over
        int ry = bestValue(s.own[Y], s.comp[Y], s.ncomp[Y], s.L[Y], onlyPiece(s.avail[Y]), occ);
        return outcomeFor(X, s.L[X], ry, redBonus);
    }
    const int px = onlyPiece(s.avail[X]);
    if (nY == 0) {
        int rx = bestValue(s.own[X], s.comp[X], s.ncomp[X], s.L[X], px, occ);
        return outcomeFor(X, rx, s.L[Y], redBonus);
    }
    // nX == 1, nY == 1: X places, then Y places its last piece.
    const int py = onlyPiece(s.avail[Y]), LY = s.L[Y], LX = s.L[X];
    const int sy = PIECE_CELLS[py], sx = PIECE_CELLS[px];
    const int oy0 = T.orBegin[py], noy = T.orBegin[py + 1] - oy0;
    const int ox0 = T.orBegin[px], nox = T.orBegin[px + 1] - ox0;

    GenCtx gy, gx;
    genCtxFor(s.own[Y], occ, gy);
    genCtxFor(s.own[X], occ, gx);
    SnakeSet SY, SX;
    SY.init(s.comp[Y], s.ncomp[Y], gy);
    SX.init(s.comp[X], s.ncomp[X], gx);
    const int ubY = std::max(LY, SY.maxJoin(sy)), ubX = std::max(LX, SX.maxJoin(sx));
    HS(6, 1);
    // X ends in [LX, ubX], Y in [LY, ubY] whatever happens (passes included)
    {
        int a = outcomeFor(X, LX, ubY, redBonus), b = outcomeFor(X, ubX, LY, redBonus);
        if (a == b) { HS(7, 1); return a; }
    }
    // Y's best value R if X does not block (per-orientation masks kept for later)
    const bool freshY = sy > LY;
    BB lgY[12], tjY[12][8];
    int R = LY, doneY = 0;
    auto scanY = [&](int upto, bool stopAtTop) {
        for (; doneY < upto; doneY++) {
            int i = doneY;
            lgY[i] = ubY > LY ? (freshY ? olFull(oy0 + i, gy) : olGrow(oy0 + i, gy)) : 0;
            if (lgY[i]) R = std::max(R, bestOfAnchors(oy0 + i, SY, lgY[i], sy, freshY ? sy : LY, tjY[i]));
            if (stopAtTop && R >= ubY) { doneY++; return; }
        }
    };
    scanY(noy, true);
    // X's placements: stop as soon as one wins outright against R
    BB lgX[12], tjX[12][8];
    int maxAll = -1, doneX = 0;
    bool anyX = false;
    auto scanX = [&](int upto, bool early) {
        for (; doneX < upto; doneX++) {
            int i = doneX;
            lgX[i] = olFull(ox0 + i, gx);
            if (!lgX[i]) continue;
            anyX = true;
            maxAll = std::max(maxAll, bestOfAnchors(ox0 + i, SX, lgX[i], sx, sx, tjX[i]));
            if (early && (maxAll >= ubX || outcomeFor(X, maxAll, R, redBonus) == 1)) { doneX++; return; }
        }
    };
    scanX(nox, true);
    if (anyX && outcomeFor(X, std::max(maxAll, LX), R, redBonus) == 1) { HS(1, 1); return 1; }
    scanX(nox, false);
    if (!anyX) return -9;  // X must pass: rare, leave it to the search
    maxAll = std::max(maxAll, LX);
    // X's longest placement can only help by also blocking: value in [out(maxAll, R), out(maxAll, LY)]
    int lo = outcomeFor(X, maxAll, R, redBonus), hi = outcomeFor(X, maxAll, LY, redBonus);
    if (lo == hi) { HS(0, 1); return lo; }
    scanY(noy, false);  // the blocking analysis needs all of Y's orientations

    // Blocking analysis. Y's placements beating LY, best first; kill = cells X must touch.
    struct Entry { BB kill; int v; };
    Entry ent[192];
    int ne = 0;
    for (int i = 0; i < noy; i++) {
        BB lg = lgY[i];
        if (!lg) continue;
        const Orient& od = T.OR[oy0 + i];
        BB anyT = 0;
        for (int j = 0; j < SY.n; j++) anyT |= tjY[i][j];
        for (BB a = anyT; a; a &= a - 1) {
            int c = lsb(a), v = sy;
            for (int j = 0; j < SY.n; j++) if (tjY[i][j] >> c & 1) v += SY.len[j];
            if (v <= LY) continue;
            if (ne == 192) return -9;
            ent[ne++] = {T.plOfGid[od.gidAt[c]]->m, v};
        }
        if (freshY)
            for (BB a = lg & ~anyT; a; a &= a - 1) {
                if (ne == 192) return -9;
                const Placement* pl = T.plOfGid[od.gidAt[lsb(a)]];
                ent[ne++] = {pl->m | pl->nb, sy};
            }
    }
    for (int i = 1; i < ne; i++) {
        Entry e = ent[i];
        int j = i;
        while (j > 0 && ent[j - 1].v < e.v) { ent[j] = ent[j - 1]; j--; }
        ent[j] = e;
    }
    HS(4, ne);
    BB U = 0;  // X must touch these cells to lower Y's best below R
    for (int i = 0; i < ne && ent[i].v == R; i++) U |= ent[i].kill;

    int best = lo;
    // X placements avoiding U leave Y at R: only X's own length matters
    int bestFree = -1;
    BB avs[12];
    for (int i = 0; i < nox; i++) {
        BB lg = lgX[i];
        avs[i] = 0;
        if (!lg) continue;
        BB av = olTouch(ox0 + i, U) & lg;
        avs[i] = av;
        BB fr = lg & ~av;
        if (!fr) continue;
        int vFree = sx;
        for (int j = 0; j < SX.n; j++) if (tjX[i][j] & fr) vFree = std::max(vFree, sx + SX.len[j]);
        for (int a = 0; a < SX.n; a++)
            if (tjX[i][a] & fr)
                for (int b = a + 1; b < SX.n; b++)
                    if (tjX[i][a] & tjX[i][b] & fr) { vFree = std::max(vFree, sx + SX.len[a] + SX.len[b]); break; }
        bestFree = std::max(bestFree, vFree);
    }
    if (bestFree >= 0) best = std::max(best, outcomeFor(X, std::max(LX, bestFree), R, redBonus));
    if (best >= hi) { HS(2, 1); return best; }
    for (int i = 0; i < nox; i++) HS(5, popc(avs[i]));
    // placements touching U, one by one: Y's best is its first surviving entry
    for (int i = 0; i < nox; i++) {
        const Orient& od = T.OR[ox0 + i];
        for (BB bl = avs[i]; bl; bl &= bl - 1) {
            const Placement& pl = *T.plOfGid[od.gidAt[lsb(bl)]];
            int lx = lengthAfter(SX, LX, pl, sx);
            if (outcomeFor(X, lx, LY, redBonus) <= best) continue;
            int ry = LY;
            for (int e = 0; e < ne; e++) if (!(ent[e].kill & pl.m)) { ry = ent[e].v; break; }
            best = std::max(best, outcomeFor(X, lx, ry, redBonus));
            if (best >= hi) { HS(3, 1); return best; }
        }
    }
    HS(3, 1);
    return best;
}

// Sound bounds on the final longest snake of player c when it has exactly one piece left:
// it can never do better than its best placement now (the opponent only removes options);
// if it is to move it can also achieve that placement.
inline bool onePieceBounds(const Pos& s, int c, int& lb, int& ub) {
    if (pieceCount(s.avail[c]) != 1 || s.ply == 0) return false;
    ub = bestValue(s.own[c], s.comp[c], s.ncomp[c], s.L[c], onlyPiece(s.avail[c]), s.own[0] | s.own[1]);
    lb = c == s.stm ? ub : s.L[c];
    return true;
}

// ---- "leaf nodes": side to move with 2 pieces, opponent with 1 -------------------------
// Every child is a last-two-plies position whose side to move (our opponent X) has the same
// cells as now: its legal placements after our move m are the current ones minus those that
// cover a cell of m (or, for a new snake, a cell of m or next to it). LeafPre holds X's data
// once per node; horizonAfter() evaluates a child from it without building the child.
struct LeafPre {
    int X, px, ox0, nox, LX;
    SnakeSet SX;
    BB lgG[12], lgF[12], tj[12][8];  // per orientation: growth / new-snake anchors, anchors touching each snake
};
inline void leafPre(const Pos& s, LeafPre& p) {
    p.X = s.stm ^ 1;
    p.px = onlyPiece(s.avail[p.X]);
    p.ox0 = T.orBegin[p.px];
    p.nox = T.orBegin[p.px + 1] - p.ox0;
    p.LX = s.L[p.X];
    GenCtx g;
    genCtxFor(s.own[p.X], s.own[0] | s.own[1], g);
    p.SX.init(s.comp[p.X], s.ncomp[p.X], g);
    for (int i = 0; i < p.nox; i++) {
        const int o = p.ox0 + i;
        p.lgG[i] = olGrow(o, g);
        p.lgF[i] = olFull(o, g) & ~p.lgG[i];
        for (int j = 0; j < p.SX.n; j++) p.tj[i][j] = olTouch(o, p.SX.nb[j]) & p.lgG[i];
    }
}
// value of the child after our move mv for its side to move (X); -9 = not decided here
inline int horizonAfter(const Pos& s, Move mv, const LeafPre& p, int redBonus) {
    const int Y = s.stm, X = p.X;
    const Placement& pl = mv.pl();
    const uint8_t availY = s.avail[Y] & (uint8_t)~(1u << mv.piece());
    if (pieceCount(availY) != 1) return -9;
    const int py = onlyPiece(availY), sy = PIECE_CELLS[py], sx = PIECE_CELLS[p.px];
    const BB mc = pl.m, occ = s.own[0] | s.own[1] | mc;
    // our snakes after the move
    BB comp[8];
    int nc = 0;
    BB merged = mc;
    for (int j = 0; j < s.ncomp[Y]; j++) {
        if (s.comp[Y][j] & pl.nb) merged |= s.comp[Y][j];
        else comp[nc++] = s.comp[Y][j];
    }
    comp[nc++] = merged;
    const int LY = std::max<int>(s.L[Y], popc(merged)), LX = p.LX;
    // X's snakes that can still be reached (sorted by length already)
    int l1 = 0, l2 = 0;
    for (int j = 0, c = 0; j < p.SX.n && c < 2; j++)
        if (p.SX.nb[j] & ~mc) { (c == 0 ? l1 : l2) = p.SX.len[j]; c++; }
    const int ubX = std::max(LX, sx + l1 + l2);
    GenCtx gy;
    genCtxFor(s.own[Y] | mc, occ, gy);
    SnakeSet SY;
    SY.init(comp, nc, gy);
    const int ubY = std::max(LY, SY.maxJoin(sy));
    {
        int a = outcomeFor(X, LX, ubY, redBonus), b = outcomeFor(X, ubX, LY, redBonus);
        if (a == b) return a;
    }
    // our best last placement if X does not block
    int R = LY;
    if (ubY > LY) {
        const bool freshY = sy > LY;
        BB tjY[8];
        for (int o = T.orBegin[py]; o < T.orBegin[py + 1] && R < ubY; o++) {
            BB lg = freshY ? olFull(o, gy) : olGrow(o, gy);
            if (lg) R = std::max(R, bestOfAnchors(o, SY, lg, sy, freshY ? sy : LY, tjY));
        }
    }
    // X's longest own placement; stop as soon as it wins outright
    int maxAll = -1;
    const BB mn = mc | pl.nb;
    for (int i = 0; i < p.nox; i++) {
        BB gr = p.lgG[i], fr = p.lgF[i];
        if (!(gr | fr)) continue;
        const int o = p.ox0 + i;
        gr &= ~olTouch(o, mc);
        fr &= ~olTouch(o, mn);
        if (!(gr | fr)) continue;
        int v = sx;
        BB tc[8];
        for (int j = 0; j < p.SX.n; j++) {
            tc[j] = p.tj[i][j] & gr;
            if (tc[j]) v = std::max(v, sx + p.SX.len[j]);
        }
        for (int a = 0; a < p.SX.n; a++)
            if (tc[a])
                for (int b = a + 1; b < p.SX.n; b++)
                    if (tc[a] & tc[b]) { v = std::max(v, sx + p.SX.len[a] + p.SX.len[b]); break; }
        maxAll = std::max(maxAll, v);
        if (outcomeFor(X, std::max(maxAll, LX), R, redBonus) == 1) return 1;
        if (maxAll >= ubX) break;
    }
    if (maxAll < 0) return -9;  // X must pass
    maxAll = std::max(maxAll, LX);
    int lo = outcomeFor(X, maxAll, R, redBonus), hi = outcomeFor(X, maxAll, LY, redBonus);
    return lo == hi ? lo : -9;  // otherwise blocking matters: full analysis
}
