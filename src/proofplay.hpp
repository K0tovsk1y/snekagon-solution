// proofplay.hpp - playing the proof: red's move in any game that follows red's winning strategy.
// Red plays 1:60, then the stored answers (moves 3, 5, 7, see proof.hpp) and, below them, a winning move found
// by the exact solver. The stored proof covers one blue move per symmetry class; a game that reaches a
// symmetric image is mapped onto the stored one by a chain of board symmetries (the "frame": a list of
// transforms taking an actual move to the stored one).
#pragma once
#include <array>
#include "proof.hpp"

struct ProofPlayer {
    std::string dir = "proof";                     // index.txt and the level-B files
    std::vector<std::array<std::string, 3>> index;  // Y Z file
    std::map<std::string, ProofFile> files;
    Solver solver{20};

    bool loadIndex(std::string& err) {
        std::ifstream in(dir + "/index.txt");
        if (!in) { err = "no " + dir + "/index.txt"; return false; }
        std::string line;
        index.clear();
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            std::array<std::string, 3> e;
            ls >> e[0] >> e[1] >> e[2];
            index.push_back(e);
        }
        return !index.empty();
    }
    static int16_t mapMove(int16_t g, const std::vector<int>& frame) {
        for (int t : frame) g = (int16_t)T.gidSym[g][t];
        return g;
    }
    static int16_t unmapMove(int16_t g, const std::vector<int>& frame) {
        for (size_t i = frame.size(); i-- > 0;) g = (int16_t)T.gidSym[g][T.inv[frame[i]]];
        return g;
    }
    // appends the transforms taking position a (actual frame) onto the equivalent position d (stored frame)
    static bool frameStep(const Pos& a, const Pos& d, std::vector<int>& frame) {
        int ta, td;
        if (a.key(ta) != d.key(td)) return false;
        frame.push_back(ta);
        frame.push_back(T.inv[td]);
        return true;
    }

    // A winning move for red (to move) by the exact solver: the first move after which blue cannot draw,
    // trying the moves the evaluation likes first.
    Move solverMove(const Pos& s) {
        std::vector<Move> ms;
        s.genMoves(ms);
        if (ms.empty()) return Move{};
        std::vector<std::pair<double, Move>> order;
        std::unordered_set<uint64_t> seen;
        for (Move m : ms) {
            Pos c = s; c.play(m);
            int t;
            if (!seen.insert(c.key(t)).second) continue;
            if (c.over()) { if (finalScore(c) > 0) return m; continue; }
            order.push_back({learnedLogit(c), m});  // blue's view: low is good for red
        }
        std::stable_sort(order.begin(), order.end(), [](auto& x, auto& y) { return x.first < y.first; });
        for (auto& [e, m] : order) {
            Pos c = s; c.play(m);
            if (solver.solve(c, 0, UINT64_MAX, -1, 0) <= -1) return m;
        }
        return order.empty() ? ms[0] : order[0].second;  // not reached in a won position
    }

    // Red's move after the moves h (red to move). src: where it came from. Returns false with `need` set
    // when that level-B file has to be loaded first (the browser fetches it), with `need` empty when red
    // has left its strategy before move 7.
    bool redMove(const std::vector<Move>& h, Move& out, std::string& src, std::string& need) {
        need.clear();
        Pos s = Pos::start();
        for (Move m : h) s.play(m);
        Move c60;
        parseMove("1:60", c60);
        if (h.empty()) { out = c60; src = "proof"; return true; }
        // below the stored answers the solver finds the winning move fast; above them (a game in which red
        // left the strategy) it could take hours: refused (false with `need` empty)
        auto bySolver = [&]() {
            if (h.size() < 6) return false;
            out = solverMove(s);
            src = "solver";
            return true;
        };
        if (h[0].gid != c60.gid || h.size() > 6 || h[1].isPass()) return bySolver();
        // move 2: blue's reply -> its stored class
        Pos a = proofRoot(); a.play(h[1]);
        const std::array<std::string, 3>* E = nullptr;
        std::vector<int> frame;
        Move y0, z0;
        for (auto& e : index) {
            Move y;
            if (!parseMove(e[0], y)) continue;
            Pos d = proofRoot(); d.play(y);
            std::vector<int> f;
            if (frameStep(a, d, f)) { E = &e; frame = f; y0 = y; parseMove(e[1], z0); break; }
        }
        if (!E) return bySolver();
        if (h.size() == 2) { out = Move{unmapMove(z0.gid, frame)}; src = "proof"; return true; }
        if (h[2].isPass() || mapMove(h[2].gid, frame) != z0.gid) return bySolver();  // red left the strategy
        auto it = files.find((*E)[2]);
        if (it == files.end()) {
            ProofFile pf;
            std::string err;
            if (!readLevelB(dir + "/" + (*E)[2], pf, err)) { need = (*E)[2]; return false; }
            it = files.emplace((*E)[2], std::move(pf)).first;
        }
        const ProofFile& pf = it->second;
        // move 4: blue's move -> its stored class
        Pos p3 = proofRoot(); p3.play(y0); p3.play(z0);
        if (h[3].isPass()) return bySolver();
        int16_t m = mapMove(h[3].gid, frame);
        auto r = pf.byMove.find(m);
        if (r == pf.byMove.end()) {
            Pos am = p3; am.play(Move{m});
            for (auto q = pf.byMove.begin(); q != pf.byMove.end(); ++q) {
                Pos dm = p3; dm.play(Move{q->first});
                if (frameStep(am, dm, frame)) { r = q; break; }
            }
            if (r == pf.byMove.end()) return bySolver();
        }
        const ProofRec& R = pf.recs[r->second];
        if (h.size() == 4) { out = Move{unmapMove(R.rep, frame)}; src = "proof"; return true; }
        if (h[4].isPass() || mapMove(h[4].gid, frame) != R.rep) return bySolver();
        // move 6: blue's move -> the stored answer
        if (h[5].isPass()) return bySolver();
        Pos p6 = p3; p6.play(Move{R.m}); p6.play(Move{R.rep});
        std::vector<Move> ms;
        p6.genMoves(ms);
        int16_t b = mapMove(h[5].gid, frame);
        for (size_t i = 0; i < ms.size(); i++)
            if (ms[i].gid == b) {
                int16_t e = pf.ent[R.off + i];
                if (e >= 0) { out = Move{unmapMove(e, frame)}; src = "proof"; return true; }
                break;
            }
        return bySolver();
    }
};
