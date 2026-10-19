// proof.hpp - the stored proof that red (player 1) wins Snekagon with the centre move 1:60, and its checker.
//
// Proof data (directory, see README): index.txt with one line per symmetry-distinct blue reply Y to 1:60,
//   <Y> <Z> <file>
// Z is red's winning answer; <file> holds the level-B records written by `solvepos` with SOLVE_LEVELB while
// it proved "blue loses after 1:60 Y Z" (SOLVE_DRAWTEST, value -1). Binary records:
//   uint16 'B', uint8 type (1), int16 m, int16 rep, uint16 n, n x int16
// m is blue's 4th move (one per symmetry class of the position after 1:60 Y Z), rep red's winning 5th move,
// and the n entries are, for each blue 6th move b (genMoves order of the position after m rep), red's 7th move
// (gid; -1 not stored, -2 game over). So red's strategy is stored for plies 1, 3, 5, 7.
//
// Checking (`snek proofcheck`): the structure (every blue move covered, every red move legal) always in full;
// then the exact solver proves, for every record or k random ones, that blue loses after red's 5th move (this
// alone proves the win), or, as a spot check of the stored 7th moves, that blue loses at leaves below them.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include "core.hpp"
#include "solver.hpp"

struct ProofRec {
    int16_t m, rep;  // blue's 4th move, red's 5th move
    uint32_t off;    // first entry in ProofFile::ent
    uint16_t n;      // entries = blue's 6th moves
};
struct ProofFile {
    std::string Y, Z, file;
    Move y, z;
    std::vector<ProofRec> recs;
    std::vector<int16_t> ent;
    std::map<int16_t, size_t> byMove;  // m -> index in recs
};

inline Pos proofRoot() {  // after red's 1:60
    Pos s = Pos::start();
    Move m;
    parseMove("1:60", m);
    s.play(m);
    return s;
}

// Reads one level-B file (records appended by possibly several runs; a later record for the same m wins).
inline bool readLevelB(const std::string& path, ProofFile& pf, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { err = "cannot open " + path; return false; }
    std::vector<char> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    size_t i = 0;
    auto rd16 = [&](size_t at) { int16_t v; std::memcpy(&v, &buf[at], 2); return v; };
    while (i < buf.size()) {
        if (i + 9 > buf.size()) { err = path + ": truncated record"; return false; }
        uint16_t tag = (uint16_t)rd16(i);
        uint8_t ty = (uint8_t)buf[i + 2];
        int16_t m = rd16(i + 3), rep = rd16(i + 5);
        uint16_t n = (uint16_t)rd16(i + 7);
        i += 9;
        if (tag != 'B' || ty != 1 || i + 2ull * n > buf.size()) { err = path + ": bad record"; return false; }
        ProofRec r{m, rep, (uint32_t)pf.ent.size(), n};
        for (uint16_t k = 0; k < n; k++) pf.ent.push_back(rd16(i + 2 * k));
        i += 2ull * n;
        auto it = pf.byMove.find(m);
        if (it != pf.byMove.end()) pf.recs[it->second] = r;
        else { pf.byMove[m] = pf.recs.size(); pf.recs.push_back(r); }
    }
    return true;
}

inline bool loadProof(const std::string& dir, std::vector<ProofFile>& out, std::string& err) {
    std::ifstream idx(dir + "/index.txt");
    if (!idx) { err = "cannot open " + dir + "/index.txt"; return false; }
    std::string line;
    while (std::getline(idx, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        ProofFile pf;
        ls >> pf.Y >> pf.Z >> pf.file;
        if (!parseMove(pf.Y, pf.y) || !parseMove(pf.Z, pf.z)) { err = "bad move in index: " + line; return false; }
        if (!readLevelB(dir + "/" + pf.file, pf, err)) return false;
        out.push_back(std::move(pf));
    }
    return true;
}

// canonical keys of the positions after each legal move (symmetry classes of the moves)
inline std::unordered_set<uint64_t> moveClasses(const Pos& p) {
    std::vector<Move> ms;
    p.genMoves(ms);
    std::unordered_set<uint64_t> k;
    for (Move m : ms) { Pos c = p; c.play(m); int t; k.insert(c.key(t)); }
    return k;
}

struct ProofLeaf { uint32_t file, rec; int idx; };  // idx: entry index, -1 = blue has no 6th move, -2 = whole record

// The structure in full: every blue move covered, every stored red move legal. Returns the number of
// problems (printed); counts the leaves per record into leaves[file][rec].
inline long proofStructure(const std::vector<ProofFile>& P, int threads, std::vector<std::vector<uint32_t>>& leaves) {
    std::atomic<long> bad{0};
    std::mutex io;
    auto report = [&](const std::string& s) { std::lock_guard<std::mutex> g(io); std::printf("PROBLEM %s\n", s.c_str()); bad++; };
    const Pos s1 = proofRoot();
    {   // ply 2: the index covers every class of blue's reply exactly once
        auto cls = moveClasses(s1);
        std::unordered_set<uint64_t> seen;
        for (auto& pf : P) {
            Pos c = s1;
            if (!c.legalMove(pf.y)) { report("blue move " + pf.Y + " is not legal after 1:60"); continue; }
            c.play(pf.y);
            int t;
            uint64_t k = c.key(t);
            if (!seen.insert(k).second) report("blue move " + pf.Y + " listed twice (up to symmetry)");
            if (!c.legalMove(pf.z)) report("red move " + pf.Z + " is not legal after " + pf.Y);
        }
        for (uint64_t k : cls) if (!seen.count(k)) report("a blue reply to 1:60 has no proof");
        std::printf("ply 2: %zu classes of blue replies, %zu in the index\n", cls.size(), seen.size());
    }
    leaves.assign(P.size(), {});
    std::atomic<size_t> next{0};
    auto work = [&]() {
        std::vector<Move> ms;
        for (;;) {
            size_t f = next++;
            if (f >= P.size()) break;
            const ProofFile& pf = P[f];
            leaves[f].assign(pf.recs.size(), 0);
            Pos p3 = s1;
            if (!p3.legalMove(pf.y)) continue;
            p3.play(pf.y);
            if (!p3.legalMove(pf.z)) continue;
            p3.play(pf.z);
            std::unordered_set<uint64_t> have;
            for (size_t r = 0; r < pf.recs.size(); r++) {
                const ProofRec& R = pf.recs[r];
                Move m{R.m}, rep{R.rep};
                if (!p3.legalMove(m)) { report(pf.Y + " " + pf.Z + ": stored blue move not legal"); continue; }
                Pos p4 = p3; p4.play(m);
                int t;
                have.insert(p4.key(t));
                if (!p4.legalMove(rep)) { report(pf.Y + " " + pf.Z + " " + moveStr(m) + ": red reply not legal"); continue; }
                Pos p6 = p4; p6.play(rep);
                p6.genMoves(ms);
                if (ms.size() != R.n) { report(pf.Y + " " + pf.Z + " " + moveStr(m) + ": entry count differs"); continue; }
                leaves[f][r] = R.n ? R.n : 1;
                for (size_t i = 0; i < ms.size(); i++) {
                    int16_t e = pf.ent[R.off + i];
                    Pos q = p6; q.play(ms[i]);
                    if (e == -2 && !q.over()) report(pf.Y + ": entry -2 but the game goes on");
                    if (e >= 0 && (q.over() || !q.legalMove(Move{e}))) report(pf.Y + " " + pf.Z + ": stored 7th move not legal");
                }
            }
            auto cls = moveClasses(p3);
            for (uint64_t k : cls) if (!have.count(k)) report(pf.Y + " " + pf.Z + ": a blue 4th move has no record");
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < threads; i++) th.emplace_back(work);
    for (auto& t : th) t.join();
    return bad;
}

// Proves one record: after 1:60 Y Z m rep, blue (to move) cannot reach a draw. This is the original proof's
// work below red's 5th move and proves everything below it; the stored 7th moves are not needed for it.
inline bool proofRecord(const ProofFile& pf, const ProofRec& R, Solver& S) {
    Pos q = proofRoot();
    q.play(pf.y); q.play(pf.z); q.play(Move{R.m}); q.play(Move{R.rep});
    if (q.over()) return finalScore(q) > 0;
    return S.solve(q, 0, UINT64_MAX, -1, 0) <= -1;
}

// Proves one leaf; returns true if red wins there. Leaf positions: after red's 7th move (blue to move) or,
// with no stored 7th move, after blue's 6th move (red to move).
inline bool proofLeaf(const ProofFile& pf, const ProofRec& R, int idx, Solver& S) {
    Pos q = proofRoot();
    q.play(pf.y); q.play(pf.z); q.play(Move{R.m}); q.play(Move{R.rep});
    int16_t e = -1;
    if (idx < 0 || R.n == 0) {
        if (q.over()) return finalScore(q) > 0;
        q.play(Move{});  // blue has no move: passes
    } else {
        std::vector<Move> ms;
        q.genMoves(ms);
        q.play(ms[idx]);
        e = pf.ent[R.off + idx];
    }
    if (q.over()) return finalScore(q) > 0;
    if (e >= 0) {
        q.play(Move{e});
        if (q.over()) return finalScore(q) > 0;
        return S.solve(q, 0, UINT64_MAX, -1, 0) <= -1;  // blue to move cannot reach a draw
    }
    return S.solve(q, 0, UINT64_MAX, 0, 1) >= 1;  // red to move wins
}

struct ProofCheckResult {
    long structural = 0, failed = 0, checked = 0;
    uint64_t total = 0;
    double cpuSec = 0, wallSec = 0;
    bool complete = false;  // a full check of everything (this run plus, with PROOFCHECK_LOG, the earlier ones)
};

// What is proved: records (default) - the position after red's 5th move, one solve each, proves the whole
// strategy; or leaves - the positions below red's stored 7th move (also checks the stored 7th moves, but
// proving every leaf costs several times more than every record: leaf by leaf, the solver cannot share the
// work between sibling positions). k < 0: all of them (resumable with PROOFCHECK_LOG=<file>, one file per
// kind); otherwise k drawn uniformly at random with the given seed.
inline ProofCheckResult proofCheck(const std::vector<ProofFile>& P, int threads, long long k, uint64_t seed, bool leaves,
                                   int ttLog2 = 22) {
    ProofCheckResult res;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<uint32_t>> nLeaves;
    res.structural = proofStructure(P, threads, nLeaves);
    std::vector<uint64_t> cum;  // units before (file, rec), flattened
    std::vector<std::pair<uint32_t, uint32_t>> fr;
    uint64_t allLeaves = 0;
    for (uint32_t f = 0; f < P.size(); f++)
        for (uint32_t r = 0; r < nLeaves[f].size(); r++) {
            cum.push_back(res.total);
            fr.push_back({f, r});
            res.total += leaves ? nLeaves[f][r] : 1;
            allLeaves += nLeaves[f][r];
        }
    std::printf("structure: %ld problems; %zu records (blue's 4th move up to symmetry, red's answer), %llu leaves\n",
                res.structural, fr.size(), (unsigned long long)allLeaves);
    std::fflush(stdout);
    // work items: one record (idx -3), all leaves of a record (idx -2: they share one solver's TT) or one leaf
    std::vector<ProofLeaf> todo;
    uint64_t total = 0;
    std::FILE* log = nullptr;
    if (k < 0) {
        std::unordered_set<uint64_t> fin;
        if (const char* lf = std::getenv("PROOFCHECK_LOG")) {
            if (std::FILE* in = std::fopen(lf, "r")) {
                unsigned f, r;
                char st[8];
                while (std::fscanf(in, "%u %u %7s", &f, &r, st) == 3) {
                    if (std::string(st) == "ok") fin.insert((uint64_t)f << 32 | r);
                    else { std::printf("FAILED earlier: record %u %u\n", f, r); res.failed++; }
                }
                std::fclose(in);
            }
            log = std::fopen(lf, "a");
        }
        // PROOFCHECK_PART=i/n: only every n-th record from the i-th (0 <= i < n), to split a full check over
        // machines; the parts together cover everything
        unsigned pi = 0, pn = 1;
        if (const char* pp = std::getenv("PROOFCHECK_PART")) {
            if (std::sscanf(pp, "%u/%u", &pi, &pn) != 2 || pn == 0 || pi >= pn) { pi = 0; pn = 1; }
            else std::printf("part %u of %u (records with index %% %u == %u)\n", pi, pn, pn, pi);
        }
        for (size_t a = 0; a < fr.size(); a++) {
            auto [f, r] = fr[a];
            if (a % pn != pi) continue;
            if (fin.count((uint64_t)f << 32 | r)) continue;
            todo.push_back({f, r, leaves ? -2 : -3});
            total += leaves ? nLeaves[f][r] : 1;
        }
        if (!fin.empty()) std::printf("resuming: %zu records already checked\n", fin.size());
        res.complete = pn == 1;
    } else {
        std::mt19937_64 rng(seed);
        for (long long j = 0; j < k && res.total; j++) {
            uint64_t x = rng() % res.total;
            size_t a = std::upper_bound(cum.begin(), cum.end(), x) - cum.begin() - 1;
            auto [f, r] = fr[a];
            const ProofRec& R = P[f].recs[r];
            todo.push_back({f, r, !leaves ? -3 : R.n ? (int)(x - cum[a]) : -1});
        }
        total = todo.size();
    }
    std::atomic<size_t> next{0};
    std::atomic<long> failed{(long)res.failed}, done{0};
    std::atomic<uint64_t> cpuUs{0};
    std::mutex io;
    auto tStart = std::chrono::steady_clock::now();
    const char* unit = leaves ? "leaves" : "records";
    auto work = [&]() {
        Solver S(ttLog2);
        for (;;) {
            size_t j = next++;
            if (j >= todo.size()) break;
            const ProofLeaf& L = todo[j];
            const ProofFile& pf = P[L.file];
            const ProofRec& R = pf.recs[L.rec];
            std::vector<int> idxs;
            if (L.idx == -3 || L.idx >= -1) idxs.push_back(L.idx);
            else if (R.n == 0) idxs.push_back(-1);
            else for (int i = 0; i < R.n; i++) idxs.push_back(i);
            auto a = std::chrono::steady_clock::now();
            std::vector<int> bad;
            for (int i : idxs) if (!(i == -3 ? proofRecord(pf, R, S) : proofLeaf(pf, R, i, S))) bad.push_back(i);
            cpuUs += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - a).count();
            long before = done.fetch_add((long)idxs.size()), d = before + (long)idxs.size();
            std::lock_guard<std::mutex> g(io);
            for (int i : bad) {
                failed++;
                std::printf("FAILED %s: 1:60 %s %s %s %s", i == -3 ? "record" : "leaf", pf.Y.c_str(), pf.Z.c_str(),
                            moveStr(Move{R.m}).c_str(), moveStr(Move{R.rep}).c_str());
                if (i != -3) std::printf(" entry %d", i);
                std::printf("\n");
            }
            if (log) { std::fprintf(log, "%u %u %s\n", L.file, L.rec, bad.empty() ? "ok" : "FAILED"); std::fflush(log); }
            const long step = leaves ? 1000 : 100;
            if (d / step != before / step || d == (long)total) {
                double w = std::chrono::duration<double>(std::chrono::steady_clock::now() - tStart).count();
                std::fprintf(stderr, "\r%ld/%llu %s, %ld failed, %.0f s, ETA %.0f s   ", d, (unsigned long long)total, unit,
                             (long)failed, w, w / d * (total - d));
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < threads; i++) th.emplace_back(work);
    for (auto& t : th) t.join();
    std::fprintf(stderr, "\n");
    if (log) std::fclose(log);
    res.failed = failed;
    res.checked = done;
    res.cpuSec = cpuUs / 1e6;
    res.wallSec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}
