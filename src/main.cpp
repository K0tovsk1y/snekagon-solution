// snek - Snekagon engine CLI.
//   snek serve                 line protocol for the UI (see README)
//   snek perft <depth>         count move sequences (for cross-checking the rules)
//   snek solvebench <plies> <count> [seconds] [threads]
//                              solve random positions after <plies> random moves
//   snek selfplay <games> <seconds/move>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#ifdef __linux__
#include <sys/resource.h>
#endif
#include "core.hpp"
#include "eval.hpp"
#include "mcts.hpp"
#include "search.hpp"
#include "solver.hpp"
#include "proofplay.hpp"
#include "serve.hpp"
#include "worker.hpp"

static uint64_t perft(const Pos& s, int d) {
    if (d == 0 || s.over()) return 1;
    std::vector<Move> mv;
    s.genMoves(mv);
    if (mv.empty()) { Pos c = s; c.play(Move{}); return perft(c, d - 1); }
    if (d == 1) return mv.size();
    uint64_t n = 0;
    for (Move m : mv) { Pos c = s; c.play(m); n += perft(c, d - 1); }
    return n;
}

static Pos randomPosition(int plies, Rng& rng) {
    for (;;) {
        Pos s = Pos::start();
        std::vector<Move> buf;
        for (int i = 0; i < plies && !s.over(); i++) {
            s.genMoves(buf);
            if (buf.empty()) s.play(Move{});
            else s.play(buf[rng.below((int)buf.size())]);
        }
        if (!s.over()) return s;
    }
}

// Reference implementation of snakeBounds (floods over own cells + growable cells), used by
// `verify` to check the fast version.
static void snakeBoundsRef(const Pos& s, int c, int& lb, int& ub) {
    BB me = s.own[c];
    lb = longest(me);
    uint8_t av = s.avail[c];
    if (!av) { ub = lb; return; }
    Shape sh = shapeOf(me);
    BB G = s.empty() & ~expand(sh.full);
    BB U = me | G;
    ub = lb;
    while (U) {
        BB K = flood(bit(lsb(U)), U);
        U &= ~K;
        BB g = K & G;
        if (!g) continue;
        int cap = 0;
        while (g) { BB r = flood(bit(lsb(g)), g); g &= ~r; cap += T.fit[av][popc(r)]; }
        ub = std::max(ub, popc(K & me) + (int)T.fit[av][std::min(cap, NIDX)]);
    }
}

// ---- training data rows ----------------------------------------------------------
static std::string hex128(BB b) {
    char buf[40];
    snprintf(buf, sizeof buf, "%016llx%016llx", (unsigned long long)(uint64_t)(b >> 64), (unsigned long long)(uint64_t)b);
    return buf;
}
static BB parseHex128(const std::string& h) {
    BB r = 0;
    for (char c : h) r = r * 16 + (BB)(c >= '0' && c <= '9' ? c - '0' : c - 'a' + 10);
    return r;
}
// own0_own1_avail0_avail1_stm_ply_passes
static std::string posStr(const Pos& s) {
    return hex128(s.own[0]) + "_" + hex128(s.own[1]) + "_" + std::to_string(s.avail[0]) + "_" + std::to_string(s.avail[1]) +
           "_" + std::to_string(s.stm) + "_" + std::to_string(s.ply) + "_" + std::to_string(s.passes);
}
static bool parsePosStr(const std::string& str, Pos& s) {
    std::vector<std::string> f;
    size_t i = 0;
    while (i <= str.size()) {
        size_t j = str.find('_', i);
        if (j == std::string::npos) j = str.size();
        f.push_back(str.substr(i, j - i));
        i = j + 1;
    }
    if (f.size() != 7) return false;
    s = Pos{};
    s.own[0] = parseHex128(f[0]); s.own[1] = parseHex128(f[1]);
    s.avail[0] = (uint8_t)std::stoi(f[2]); s.avail[1] = (uint8_t)std::stoi(f[3]);
    s.stm = (uint8_t)std::stoi(f[4]); s.ply = (uint8_t)std::stoi(f[5]); s.passes = (uint8_t)std::stoi(f[6]);
    s.initHash();
    return true;
}
static void printDataHeader() {
    printf("game,ply,stm,value");
    const char* names[] = {"L", "rem", "pieces", "terr", "bestTerr", "bestOwn", "lb", "ub", "fresh", "ends", "liveEnds", "singleOwn", "singleTerr", "comps"};
    for (const char* side : {"s", "o"}) for (const char* n : names) printf(",%s_%s", n, side);
    printf(",eval,pos\n");
}
static std::string dataRow(int g, const Pos& s, int v, int head) {
    PlayerFeat f[2];
    features(s, f, head);
    char buf[512];
    std::string row;
    snprintf(buf, sizeof buf, "%d,%d,%d,%d", g, (int)s.ply, (int)s.stm, v);
    row += buf;
    for (int c : {(int)s.stm, (int)s.stm ^ 1}) {
        auto& x = f[c];
        snprintf(buf, sizeof buf, ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", x.L, x.rem, x.pieces, x.terr, x.bestTerr, x.bestOwn, x.lb, x.ub, x.fresh, x.ends, x.liveEnds, x.singleOwn, x.singleTerr, x.comps);
        row += buf;
    }
    snprintf(buf, sizeof buf, ",%d,", evaluateHand(s));
    row += buf;
    row += posStr(s) + "\n";
    return row;
}

static void serve(int threads) {
    ServeSession session(threads);
    // SERVE_PROGRESS=1 (set by ui/server.js): solve also prints its {"progress":...} lines
    if (getenv("SERVE_PROGRESS")) session.progress = [](const std::string& p) { std::cout << p << "\n" << std::flush; };
    std::string line;
    while (std::getline(std::cin, line)) {
        std::string out = session.handle(line);
        if (out.empty()) break;  // quit
        std::cout << out << "\n" << std::flush;
    }
}

int main(int argc, char** argv) {
    initTables();
    std::string mode = argc > 1 ? argv[1] : "serve";
    if (mode == "perft") {
        int d = argc > 2 ? std::atoi(argv[2]) : 3;
        Pos s = Pos::start();
        for (int i = 1; i <= d; i++) {
            auto t0 = std::chrono::steady_clock::now();
            uint64_t n = perft(s, i);
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("perft %d = %llu  (%.2fs)\n", i, (unsigned long long)n, el);
            fflush(stdout);
        }
    } else if (mode == "verify") {
        // fast generator == slow per-placement check (same set, same order);
        // snake bounds hold for the final result of every random game.
        int games = argc > 2 ? std::atoi(argv[2]) : 2000;
        Rng rng(99);
        long positions = 0, boundChecks = 0;
        std::vector<Move> fast, slow;
        for (int g = 0; g < games; g++) {
            Pos s = Pos::start();
            std::vector<Pos> hist;
            while (!s.over()) {
                s.genMoves(fast);
                slow.clear();
                uint8_t av = s.movable();
                for (int p = 0; p < NPIECE; p++)
                    if (av >> p & 1)
                        for (auto& pl : T.PL[p]) if (s.legal(pl)) slow.push_back(Move{(int16_t)pl.gid});
                if (fast.size() != slow.size() || !std::equal(fast.begin(), fast.end(), slow.begin())) {
                    printf("MOVEGEN MISMATCH game %d ply %d: fast %zu slow %zu\n", g, (int)s.ply, fast.size(), slow.size());
                    return 1;
                }
                if (s.hasMove() != !fast.empty()) { printf("HASMOVE MISMATCH\n"); return 1; }
                for (int c = 0; c < 2; c++) {
                    // incremental snakes == recomputed; fast bounds == reference (flood) bounds
                    if (s.L[c] != longest(s.own[c])) { printf("LONGEST MISMATCH\n"); return 1; }
                    BB u = 0;
                    for (int j = 0; j < s.ncomp[c]; j++) u |= s.comp[c][j];
                    if (u != s.own[c]) { printf("COMPONENT MISMATCH\n"); return 1; }
                    int lb, ub, rlb, rub;
                    snakeBounds(s, c, lb, ub);
                    snakeBoundsRef(s, c, rlb, rub);
                    if (lb != rlb || ub != rub) { printf("BOUNDS MISMATCH game %d ply %d: [%d,%d] vs ref [%d,%d]\n", g, (int)s.ply, lb, ub, rlb, rub); return 1; }
                }
                positions++;
                hist.push_back(s);
                // mix uniformly random and "grow own snake" moves so games get long snakes
                Move m{};
                if (!fast.empty()) {
                    std::vector<Move> grow;
                    for (Move x : fast) if (x.pl().nb & s.own[s.stm]) grow.push_back(x);
                    if (!grow.empty() && rng.below(3)) m = grow[rng.below((int)grow.size())];
                    else m = fast[rng.below((int)fast.size())];
                }
                s.play(m);
            }
            int fin[2] = {longest(s.own[0]), longest(s.own[1])};
            for (auto& h : hist)
                for (int c = 0; c < 2; c++) {
                    int lb, ub;
                    snakeBounds(h, c, lb, ub);
                    boundChecks++;
                    if (fin[c] < lb || fin[c] > ub) {
                        printf("BOUND VIOLATION game %d ply %d player %d: final %d, bounds [%d,%d]\n", g, (int)h.ply, c, fin[c], lb, ub);
                        return 1;
                    }
                }
        }
        printf("ok: %ld positions movegen identical, %ld bound checks hold\n", positions, boundChecks);
    } else if (mode == "gendata") {
        // snek gendata <games> <plyMin> <plyMax> <solveSecs> <threads> <seed>
        // Plays softmax-greedy games (1-ply eval, some random moves), solves the
        // positions in [plyMin, plyMax] exactly and prints features + result (CSV).
        int games = argc > 2 ? std::atoi(argv[2]) : 100;
        int plyMin = argc > 3 ? std::atoi(argv[3]) : 7;
        int plyMax = argc > 4 ? std::atoi(argv[4]) : 10;
        double secs = argc > 5 ? std::atof(argv[5]) : 5;
        int threads = argc > 6 ? std::atoi(argv[6]) : 8;
        uint64_t seed = argc > 7 ? std::atoll(argv[7]) : 7;
        bool goodPlay = argc > 8 && std::string(argv[8]) == "good";
        int featHead = argc > 9 ? std::atoi(argv[9]) : 0;  // Voronoi head start for the feature columns
        std::atomic<int> next{0};
        std::mutex mu;
        printDataHeader();
        std::vector<std::thread> th;
        for (int w = 0; w < threads; w++)
            th.emplace_back([&, w] {
                Solver solver(21);
                std::vector<Move> mv;
                for (;;) {
                    int g = next++;
                    if (g >= games) break;
                    Rng rng(seed * 1000003 + g);
                    Pos s = Pos::start();
                    double temp = 40 + rng.below(160);  // centi-cells
                    double randP = 0.05 + 0.2 * rng.uni();
                    bool growPolicy = g % 2 == 1;
                    if (goodPlay) {  // random first two plies, then (almost) greedy 1-ply play
                        temp = 10 + rng.below(30);
                        randP = 0;
                        growPolicy = false;
                    }
                    while (!s.over() && s.ply <= plyMax) {
                        if (s.ply >= plyMin) {
                            solver.clear();
                            int v = solver.solve(s, secs);
                            if (v != -9) {
                                std::string row = dataRow(g, s, v, featHead);
                                std::lock_guard<std::mutex> lk(mu);
                                fputs(row.c_str(), stdout);
                                fflush(stdout);
                            }
                        }
                        s.genMoves(mv);
                        if (mv.empty()) { s.play(Move{}); continue; }
                        if (rng.uni() < randP || s.ply == 0 || (goodPlay && s.ply == 1)) { s.play(mv[rng.below((int)mv.size())]); continue; }
                        if (growPolicy) {
                            // prefer growing an own snake (mostly the longest one), sometimes block
                            std::vector<Move> grow;
                            BB me = s.own[s.stm];
                            for (Move m : mv) if (m.pl().nb & me) grow.push_back(m);
                            if (!grow.empty() && rng.uni() < 0.85) { s.play(grow[rng.below((int)grow.size())]); continue; }
                            s.play(mv[rng.below((int)mv.size())]);
                            continue;
                        }
                        std::vector<double> sc(mv.size());
                        double mx = -1e18;
                        for (size_t i = 0; i < mv.size(); i++) {
                            Pos c = s; c.play(mv[i]);
                            sc[i] = c.over() ? -terminalScore(c) : -evaluate(c);
                            mx = std::max(mx, sc[i]);
                        }
                        double tot = 0;
                        for (auto& x : sc) { x = std::exp((x - mx) / temp); tot += x; }
                        double r = rng.uni() * tot;
                        size_t pick = 0;
                        for (; pick + 1 < sc.size(); pick++) { r -= sc[pick]; if (r <= 0) break; }
                        s.play(mv[pick]);
                    }
                }
            });
        for (auto& t : th) t.join();
    } else if (mode == "proofgame") {
        // snek proofgame <proof dir> [games] [seed] [blue seconds]: red plays the proof strategy (proofplay.hpp)
        // against random blue moves (or the alpha-beta engine with that time per move); every game must end in a red win. Reports how long red's solver moves take.
        ProofPlayer pp;
        pp.dir = argc > 2 ? argv[2] : "proof";
        int games = argc > 3 ? std::atoi(argv[3]) : 20;
        Rng rng(argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1);
        double blueSec = argc > 5 ? std::atof(argv[5]) : 0;  // > 0: blue is the alpha-beta engine
        std::string err;
        if (!pp.loadIndex(err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        int wins = 0;
        double maxSolve = 0, sumSolve = 0;
        int nSolve = 0, nProof = 0;
        for (int g = 0; g < games; g++) {
            Pos s = Pos::start();
            std::vector<Move> h, buf;
            while (!s.over()) {
                Move m;
                if (s.stm == 0) {
                    std::string src, need;
                    auto a = std::chrono::steady_clock::now();
                    if (!pp.redMove(h, m, src, need)) { std::fprintf(stderr, "missing %s\n", need.c_str()); return 2; }
                    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
                    if (src == "solver") { nSolve++; sumSolve += el; maxSolve = std::max(maxSolve, el); } else nProof++;
                } else {
                    s.genMoves(buf);
                    if (buf.empty()) m = Move{};
                    else if (blueSec > 0) m = abAnalyze(s, blueSec, 1, 1).moves[0].mv;  // a strong blue
                    else m = buf[rng.below((int)buf.size())];
                }
                if (!m.isPass() && !s.legalMove(m)) { std::printf("game %d: illegal red move %s\n", g, moveStr(m).c_str()); return 1; }
                s.play(m);
                h.push_back(m);
            }
            bool won = finalScore(s) > 0;
            wins += won;
            std::printf("game %d: %s (%d-%d):", g, won ? "red wins" : "RED DID NOT WIN", (int)s.L[0], (int)s.L[1]);
            for (Move m : h) std::printf(" %s", moveStr(m).c_str());
            std::printf("\n");
            std::fflush(stdout);
        }
        std::printf("red won %d of %d games; red moves from the proof: %d, by the solver: %d (avg %.3f s, max %.3f s)\n",
                    wins, games, nProof, nSolve, nSolve ? sumSolve / nSolve : 0.0, maxSolve);
        return wins == games ? 0 : 1;
    } else if (mode == "proofcheck") {
        // snek proofcheck <proof dir> [threads] [k|all] [seed] [records|leaves]: check the stored proof that red wins with 1:60
        // (proof.hpp): the structure in full, then every record (k omitted or "all") or k random ones, or leaves
        std::string dir = argc > 2 ? argv[2] : "proof";
        int threads = argc > 3 ? std::atoi(argv[3]) : (int)std::thread::hardware_concurrency();
        long long k = argc > 4 && std::string(argv[4]) != "all" ? std::atoll(argv[4]) : -1;
        uint64_t seed = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 1;
        bool leaves = argc > 6 && std::string(argv[6]) == "leaves";  // default: records
        const char* unit = leaves ? "leaves" : "records";
        std::vector<ProofFile> P;
        std::string err;
        if (!loadProof(dir, P, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        std::printf("proof: %zu blue replies to 1:60, %d threads, %s %s\n", P.size(), threads,
                    k < 0 ? "all" : (std::to_string(k) + " random").c_str(), unit);
        std::fflush(stdout);
        auto r = proofCheck(P, threads, k, seed, leaves);
        std::printf("checked %ld of %llu %s: %ld failed; structure problems: %ld\n", r.checked,
                    (unsigned long long)r.total, unit, r.failed, r.structural);
        if (r.checked)
            std::printf("time: %.2f ms per %s on average (one thread), %.2f CPU-s in all, wall %.1f s; all %s: ~%.1f thread-hours\n",
                        r.cpuSec * 1e3 / r.checked, leaves ? "leaf" : "record", r.cpuSec, r.wallSec, unit,
                        r.cpuSec / r.checked * r.total / 3600);
        std::printf("%s\n", r.failed || r.structural ? "RESULT: NOT VERIFIED" :
                    k >= 0 ? "RESULT: no counterexample in the sample" :
                    !r.complete ? "RESULT: this part is verified (all parts together verify the proof)" :
                    leaves ? "RESULT: every leaf below the stored moves is won for red"
                           : "RESULT: VERIFIED - red wins with 1:60");
        return r.failed || r.structural ? 1 : 0;
    } else if (mode == "solvepos") {
        // snek solvepos <seconds> <threads> <move> <move> ...   (parallel exact solve)
        double secs = std::atof(argv[2]);
        int threads = std::atoi(argv[3]);
        Pos s = Pos::start();
        for (int i = 4; i < argc; i++) {
            Move m;
            if (!parseMove(argv[i], m) || !s.legalMove(m)) { printf("illegal move %s\n", argv[i]); return 1; }
            s.play(m);
        }
        // env PROGRESS=<file>: append every solved root move ("<move> <win|draw|loss> <nodes> <sec>")
        // and, if the file exists, skip the moves already recorded there (resume).
        std::vector<std::pair<Move, int>> prev;
        FILE* prog = nullptr;
        if (const char* pf = getenv("PROGRESS")) {
            std::ifstream in(pf);
            std::string ms, vs;
            std::string line;
            while (std::getline(in, line)) {
                std::istringstream ls(line);
                Move m;
                if (ls >> ms >> vs && parseMove(ms, m)) prev.push_back({m, vs == "win" ? 1 : vs == "loss" ? -1 : 0});
            }
            if (!prev.empty()) printf("resuming: %zu root moves already solved\n", prev.size());
            prog = fopen(pf, "a");
        }
        printf("root moves: %zu  ply %d stm %d\n", uniqueRootMoves(s).size(), (int)s.ply, (int)s.stm);
        fflush(stdout);
        auto t0 = std::chrono::steady_clock::now();
        size_t done = prev.size();
        // the opponent's reply that decided a root move (the refutation of a lost move), 5th column
        Move lastRoot{}, lastReply{};
        bool haveReply = false;
        g_rootReply = [&](Move m, Move rep) { lastRoot = m; lastReply = rep; haveReply = true; };
        auto r = parallelSolve(s, secs, threads, 22, prev, [&](Move m, int v, uint64_t nodes, double sec) {
            done++;
            const char* vs = v > 0 ? "win" : v < 0 ? "loss" : getenv("SOLVE_WINONLY") ? "nowin" : getenv("SOLVE_DRAWTEST") ? "drawplus" : "draw";
            const std::string rs = haveReply && lastRoot == m ? moveStr(lastReply) : "-";
            haveReply = false;
            if (prog) { fprintf(prog, "%s %s %llu %.1f %s\n", moveStr(m).c_str(), vs, (unsigned long long)nodes, sec, rs.c_str()); fflush(prog); }
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("[%6.0fs] %4zu solved  %-22s %-4s %6.1fs %12llu nodes\n", el, done, moveStr(m).c_str(), vs, sec, (unsigned long long)nodes);
            fflush(stdout);
        });
        if (prog) fclose(prog);
        int known = 0;
        for (auto& pm : r.perMove) known += pm.second != -9;
        printf("ply %d stm %d: value %s  (%d/%zu root moves solved, %llu nodes, %.1fs)\n", (int)s.ply, (int)s.stm,
               r.value == -9 ? "unknown" : std::to_string(r.value).c_str(), known, r.perMove.size(), (unsigned long long)r.nodes, r.seconds);
        for (auto& pm : r.perMove)
            if (pm.second != -9) printf("  %-22s %s\n", moveStr(pm.first).c_str(), pm.second > 0 ? "win" : pm.second < 0 ? "loss" : "draw");
    } else if (mode == "worker") {
        // snek worker <host[:port]> [threads] [name]
        if (argc < 3) { printf("usage: snek worker <coordinator-ip[:port]> [threads] [name]\n"); return 1; }
        int threads = argc > 3 ? std::atoi(argv[3]) : (int)std::max(1u, std::thread::hardware_concurrency());
        char hn[128] = "worker";
#ifdef _WIN32
        { DWORD n = sizeof hn; GetComputerNameA(hn, &n); }
#else
        gethostname(hn, sizeof hn);
#endif
        return runWorker(argv[2], threads, argc > 4 ? argv[4] : hn);
    } else if (mode == "solvebatch") {
        // snek solvebatch <seconds per problem> <threads> <file>: parallelSolve every position of the
        // file (one move list per line) in turn, in one process. With SOLVE_PERSIST=1 the solvers
        // and TTs are kept between the problems.
        double secs = std::atof(argv[2]);
        int threads = std::atoi(argv[3]);
        std::ifstream in(argv[4]);
        std::string line;
        auto tAll = std::chrono::steady_clock::now();
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ls(line);
            std::string tok;
            Pos s = Pos::start();
            bool ok = true;
            while (ls >> tok) { Move m; if (!parseMove(tok, m) || !s.legalMove(m)) { ok = false; break; } s.play(m); }
            if (!ok || s.ply == 0) continue;
            auto r = parallelSolve(s, secs, threads, 22);
            int known = 0;
            for (auto& pm : r.perMove) known += pm.second != -9;
            Move bestM{};
            for (auto& pm : r.perMove) if (pm.second == r.value && r.value != -9) { bestM = pm.first; if (r.value == 1 || pm.second >= 0) break; }
            printf("%-60s value %2d  %4d/%zu moves  %12llu nodes  %7.1fs  %s\n", line.c_str(), r.value, known, r.perMove.size(),
                   (unsigned long long)r.nodes, r.seconds, r.value == -9 ? "" : moveStr(bestM).c_str());
            fflush(stdout);
        }
        printf("BATCH total %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tAll).count());
    } else if (mode == "nwcheck") {
        // snek nwcheck <count> <seed> <plies> [threads]: null-window answers ("value >= 0?",
        // "value >= 1?") against the exact value, on random positions after <plies> random moves.
        // The null-window solver keeps its TT across all queries (mixed windows), the exact one
        // is fresh per position.
        int cnt = std::atoi(argv[2]), seed = std::atoi(argv[3]), plies = std::atoi(argv[4]);
        int threads = argc > 5 ? std::atoi(argv[5]) : 8;
        std::atomic<int> next{0}, bad{0}, done{0};
        std::vector<std::thread> th;
        for (int w = 0; w < threads; w++)
            th.emplace_back([&, w] {
                Solver X(20), N(22);
                for (;;) {
                    int i = next++;
                    if (i >= cnt) break;
                    std::mt19937 rng(seed * 1000003 + i);
                    Pos s = Pos::start();
                    bool ok = true;
                    while (s.ply < plies) {
                        std::vector<Move> mv; s.genMoves(mv);
                        if (mv.empty()) { ok = false; break; }
                        s.play(mv[rng() % mv.size()]);
                    }
                    if (!ok || s.over()) continue;
                    X.clear();
                    int v = X.solve(s);
                    for (int t : {(int)(rng() & 1), 0, 1}) {
                        int r = N.solve(s, 0, UINT64_MAX, t - 1, t);
                        if ((r >= t) != (v >= t)) { bad++; printf("MISMATCH pos %d t %d exact %d null %d\n", i, t, v, r); fflush(stdout); }
                    }
                    done++;
                }
            });
        for (auto& t : th) t.join();
        printf("NWCHECK %d positions after %d plies: %d mismatches\n", done.load(), plies, bad.load());
    } else if (mode == "benchset") {
        // snek benchset <file> <threads>: exact solve of every position in the file (one move
        // list per line). Threads take positions in order and keep their solver between
        // positions, like parallelSolve does. Prints per-position and total cost.
        // A line may start with "@<t><y|n>" (t = 0 or 1): then only the question "value for the
        // side to move >= t?" is searched (null window (t-1, t)), as inside a proof of a win (t = 1)
        // or of a draw (t = 0); y/n is the known answer, and wrong answers are counted.
        std::ifstream in(argv[2]);
        int threads = argc > 3 ? std::atoi(argv[3]) : 8;
        std::vector<Pos> ps;
        std::vector<int> tgt;    // 9 = exact value
        std::vector<char> want;  // 'y' / 'n' / '?'
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string tok;
            Pos s = Pos::start();
            bool ok = true, any = false;
            int t = 9;
            char w = '?';
            while (ls >> tok) {
                if (!any && tok.size() == 3 && tok[0] == '@') { t = tok[1] - '0'; w = tok[2]; continue; }
                Move m; if (!parseMove(tok, m) || !s.legalMove(m)) { ok = false; break; } s.play(m); any = true;
            }
            if (ok && any) { ps.push_back(s); tgt.push_back(t); want.push_back(w); }
        }
        const double cap = envInt("BENCH_CAP", 0);
        std::vector<int> val(ps.size());
        std::vector<uint64_t> nod(ps.size());
        std::vector<double> sec(ps.size());
        std::atomic<size_t> next{0};
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> th;
        std::mutex profMu;
        uint64_t profV[16] = {}, profT[16] = {}, profB[16] = {}, profE[16] = {}, profH[16] = {}, profTm[16][8] = {};
        uint64_t profFt[16][8] = {}, profFc[16][8] = {}, profEo[16][6] = {};
        uint64_t profCs[16][2][8] = {};
        uint64_t profCd[16][2][3] = {};
        int sharedLog2 = envInt("SOLVER_SHARED", 0);
        std::shared_ptr<LTT> shared = sharedLog2 ? std::make_shared<LTT>(sharedLog2 - 1) : nullptr;
        FILE* dumpF = getenv("DUMP_PLY") ? fopen(getenv("DUMP_FILE") ? getenv("DUMP_FILE") : "dump.txt", "w") : nullptr;
        std::mutex dumpMu;
        for (int w = 0; w < threads; w++)
            th.emplace_back([&] {
                std::unique_ptr<Solver> SP(shared ? new Solver(shared) : new Solver(envInt("BENCH_TT", 22)));
                Solver& S = *SP;
                if (dumpF) { S.dumpOut = dumpF; S.dumpMu = &dumpMu; S.dumpPly = envInt("DUMP_PLY", -1); }
                for (;;) {
                    size_t i = next++;
                    if (i >= ps.size()) break;
                    auto ts = std::chrono::steady_clock::now();
                    if (getenv("BENCH_FRESH")) S.clear();  // deterministic: no TT carry-over between positions
                    // BENCH_CAP=<seconds>: give up a position after that long (value -9), for labelling new sets
                    if (tgt[i] == 9) val[i] = S.solve(ps[i], cap);
                    else {
                        int v = S.solve(ps[i], cap, UINT64_MAX, tgt[i] - 1, tgt[i]);
                        val[i] = v == -9 ? -9 : v >= tgt[i] ? 1 : 0;  // 1 = yes
                    }
                    nod[i] = S.nodes;
                    {
                        std::lock_guard<std::mutex> lk(profMu);
                        for (int p = 0; p < 16; p++) { profV[p] += S.pVisit[p]; profT[p] += S.pTT[p]; profB[p] += S.pBound[p]; profE[p] += S.pExp[p]; profH[p] += S.pHor[p]; }
                        for (int p = 0; p < 16; p++) for (int q = 0; q < 8; q++) { profTm[p][q] += S.tm[p][q]; S.tm[p][q] = 0; }
                        for (int p = 0; p < 16; p++) for (int q = 0; q < 8; q++) { profFt[p][q] += S.ft[p][q]; profFc[p][q] += S.fc[p][q]; S.ft[p][q] = S.fc[p][q] = 0; }
                        for (int p = 0; p < 16; p++) for (int q = 0; q < 6; q++) { profEo[p][q] += S.eo[p][q]; S.eo[p][q] = 0; }
                        for (int p = 0; p < 16; p++) for (int w = 0; w < 2; w++) for (int q = 0; q < 8; q++) { profCs[p][w][q] += S.cs[p][w][q]; S.cs[p][w][q] = 0; }
                        for (int p = 0; p < 16; p++) for (int w = 0; w < 2; w++) for (int q = 0; q < 3; q++) { profCd[p][w][q] += S.cutDir[p][w][q]; S.cutDir[p][w][q] = 0; }
                        std::fill(S.pVisit, S.pVisit + 16, 0); std::fill(S.pTT, S.pTT + 16, 0);
                        std::fill(S.pBound, S.pBound + 16, 0); std::fill(S.pExp, S.pExp + 16, 0);
                        std::fill(S.pHor, S.pHor + 16, 0);
                    }
                    sec[i] = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
                }
            });
        for (auto& t : th) t.join();
        if (dumpF) fclose(dumpF);
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        uint64_t tn = 0;
        double ts = 0;
        int wrong = 0;
        for (size_t i = 0; i < ps.size(); i++) {
            if (tgt[i] == 9) printf("%3zu  value %2d  %12llu nodes  %7.1fs\n", i, val[i], (unsigned long long)nod[i], sec[i]);
            else {
                char a = val[i] == -9 ? '?' : val[i] ? 'y' : 'n';
                bool bad = want[i] != '?' && a != '?' && want[i] != a;
                wrong += bad;
                printf("%3zu  >=%d? %c%s  %12llu nodes  %7.1fs\n", i, tgt[i], a, bad ? " WRONG" : "", (unsigned long long)nod[i], sec[i]);
            }
            tn += nod[i]; ts += sec[i];
        }
        printf("TOTAL %zu positions: %llu nodes, %.1f thread-seconds, %.1f s wall (%d threads)", ps.size(), (unsigned long long)tn, ts, wall, threads);
#ifdef _WIN32
        {
            FILETIME c, e, k, u;
            if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) {
                auto ft = [](FILETIME f) { return double((uint64_t)f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7; };
                printf(", CPU time %.1f s", ft(k) + ft(u));
            }
        }
#endif
        if (std::count(tgt.begin(), tgt.end(), 9) != (long)tgt.size()) printf(", wrong answers %d", wrong);
#ifdef __linux__
        {
            struct rusage ru;
            getrusage(RUSAGE_SELF, &ru);
            printf(", CPU time %.1f s", ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec / 1e6);
        }
#endif
        printf("\n");
#ifdef HSTATS
        printf("HSTATS calls(1,1) %llu: lo==hi %llu, lo==1 %llu, after free %llu, after blocking %llu; entries %llu, blocking candidates %llu\n",
               (unsigned long long)HST[6], (unsigned long long)HST[0], (unsigned long long)HST[1], (unsigned long long)HST[2],
               (unsigned long long)HST[3], (unsigned long long)HST[4], (unsigned long long)HST[5]);
#endif
        if (getenv("PROFILE")) {
            double allc = 0;
            bool anyF = false;
            for (int p = 0; p < 16; p++) for (int q = 0; q < 8; q++) { allc += profTm[p][q]; anyF |= profFt[p][q] > 0; }
            if (anyF && allc > 0) {
                printf("bound steps, %% of all cycles (cycles/visit): moveLB   oppUB   moveLB2   ownUB   miai   bslot\n");
                for (int p = 0; p < 16; p++) {
                    double rs = 0;
                    for (int q = 0; q < 8; q++) rs += profFt[p][q];
                    if (rs <= 0) continue;
                    printf("  ply %2d:", p);
                    for (int q = 0; q < 7; q++) if (q != 2) printf("  %5.1f%% (%5.0f)", 100.0 * profFt[p][q] / allc, profFc[p][q] ? (double)profFt[p][q] / profFc[p][q] : 0.0);
                    printf("\n");
                }
            }
            printf("eval-ordered nodes: ply  count  cut@1st  cut@2-3  cut@later  no-cut  moves/node\n");
            for (int p = 0; p < 16; p++)
                if (profEo[p][0]) {
                    double c = (double)profEo[p][0];
                    printf("                    %3d %8llu %7.1f%% %7.1f%% %8.1f%% %6.1f%% %8.1f\n", p, (unsigned long long)profEo[p][0],
                           100 * profEo[p][1] / c, 100 * profEo[p][2] / c, 100 * profEo[p][3] / c, 100 * profEo[p][4] / c, profEo[p][5] / c);
                }
            printf("eval-ordered children: ply win  nodes  moves/node  A(touch own) B(take front) A|B  | best in A  B  A|B\n");
            for (int p = 0; p < 16; p++)
                for (int w = 0; w < 2; w++)
                    if (profCs[p][w][0]) {
                        const uint64_t* q = profCs[p][w];
                        const double c = (double)q[0];
                        printf("                       %3d %d %8llu %8.1f %8.1f %8.1f %8.1f   | %6.1f%% %6.1f%% %6.1f%%\n", p, w, (unsigned long long)q[0],
                               q[1] / c, q[2] / c, q[3] / c, q[4] / c, 100 * q[5] / c, 100 * q[6] / c, 100 * q[7] / c);
                    }
            printf("bound decisions: ply win  lo-cut  hi-cut  none (of nodes reaching the bounds)\n");
            for (int p = 0; p < 16; p++)
                for (int w = 0; w < 2; w++) {
                    const uint64_t* q = profCd[p][w];
                    const double c = (double)(q[0] + q[1] + q[2]);
                    if (c > 0) printf("                 %3d %d %10llu  %5.1f%%  %5.1f%%  %5.1f%%\n", p, w, (unsigned long long)c, 100 * q[0] / c, 100 * q[1] / c, 100 * q[2] / c);
                }
            printf("ply     visited    tt-cut  bound-cut  expanded     horizon\n");
            for (int p = 0; p < 16; p++) {
                double v = (double)std::max<uint64_t>(1, profV[p]);
                if (profV[p] || profH[p]) printf("%3d %11llu %9.1f%% %9.1f%% %9.1f%% %11llu\n", p, (unsigned long long)profV[p], 100.0 * profT[p] / v, 100.0 * profB[p] / v, 100.0 * profE[p] / v, (unsigned long long)profH[p]);
            }
            double all = 0;
            for (int p = 0; p < 16; p++) for (int q = 0; q < 8; q++) all += profTm[p][q];
            if (all > 0) {
                printf("cycles %% of total  ply: horizon     tt  bounds    gen  order  sel+play  store\n");
                for (int p = 0; p < 16; p++) {
                    double rs = 0;
                    for (int q = 0; q < 8; q++) rs += profTm[p][q];
                    if (rs <= 0) continue;
                    printf("                  %3d:", p);
                    for (int q = 0; q < 7; q++) printf(" %6.1f", 100.0 * profTm[p][q] / all);
                    printf("   (row %.1f%%)\n", 100.0 * rs / all);
                }
                printf("total measured: %.2f Gcycles\n", all / 1e9);
            }
        }
    } else if (mode == "genorient") {
        // snek genorient > src/orient_gen.hpp: one move-generator function per orientation with
        // compile-time shifts (same result as orientLegal)
        printf("// Generated by `snek genorient` - do not edit. One legality function per piece orientation\n");
        printf("// with compile-time shift amounts; identical results to orientLegal() in core.hpp.\n#pragma once\n\n");
        auto sh = [](const char* b, int dq, int dr) {
            char buf[96];
            snprintf(buf, sizeof buf, "shc<%d, %d>(%s)", dq, dr, b);
            return std::string(buf);
        };
        for (int full = 1; full >= 0; full--)
            for (size_t o = 0; o < T.OR.size(); o++) {
                const Orient& od = T.OR[o];
                printf("inline BB %s_%zu(const GenCtx& g) {\n", full ? "olFull" : "olGrow", o);
                if (full) {
                    printf("    BB nw = ");
                    for (int k = 0; k < od.n; k++) printf("%s%s", k ? " & " : "", sh("g.iso", od.dq[k], od.dr[k]).c_str());
                    printf(";\n    if (!g.adjOwn) return nw & T.VALID;\n");
                }
                printf("    BB gr = ");
                for (int k = 0; k < od.n; k++) {
                    char b[32]; snprintf(b, sizeof b, "g.allow[%d]", od.ideg[k]);
                    printf("%s%s", k ? " & " : "", sh(b, od.dq[k], od.dr[k]).c_str());
                }
                printf(";\n    gr &= ");
                for (int k = 0; k < od.n; k++) printf("%s%s", k ? " | " : "(", sh("g.adjOwn", od.dq[k], od.dr[k]).c_str());
                printf(");\n");
                if (!od.tri.empty() || !od.two.empty()) {
                    printf("    if (gr) gr &= ~(");
                    bool first = true;
                    for (auto [q, r] : od.tri) { printf("%s%s", first ? "" : " | ", sh("g.me", q, r).c_str()); first = false; }
                    for (auto [q, r] : od.two) { printf("%s%s", first ? "" : " | ", sh("g.meDeg1", q, r).c_str()); first = false; }
                    printf(");\n");
                }
                printf("    return %s;\n}\n", full ? "(nw | gr) & T.VALID" : "gr & T.VALID");
            }
        for (int full = 1; full >= 0; full--) {
            printf("inline BB (*const %s[%zu])(const GenCtx&) = {", full ? "OL_FULL" : "OL_GROW", T.OR.size());
            for (size_t o = 0; o < T.OR.size(); o++) printf("%s%s_%zu", o ? ", " : "", full ? "olFull" : "olGrow", o);
            printf("};\n");
        }
        // anchors whose placement covers a cell of b
        for (size_t o = 0; o < T.OR.size(); o++) {
            const Orient& od = T.OR[o];
            printf("inline BB touch_%zu(BB b) { return (", o);
            for (int k = 0; k < od.n; k++) printf("%s%s", k ? " | " : "", sh("b", od.dq[k], od.dr[k]).c_str());
            printf(") & T.VALID; }\n");
        }
        printf("inline BB (*const TOUCH[%zu])(BB) = {", T.OR.size());
        for (size_t o = 0; o < T.OR.size(); o++) printf("%stouch_%zu", o ? ", " : "", o);
        printf("};\n");
    } else if (mode == "olcheck") {
        // snek olcheck <n>: generated move generator == orientLegal on n random positions
        int n = argc > 2 ? std::atoi(argv[2]) : 10000;
        Rng rng(5);
        long bad = 0, tot = 0;
        for (int i = 0; i < n; i++) {
            Pos s = randomPosition(rng.below(14), rng);
            GenCtx g;
            s.genCtx(g);
            for (size_t o = 0; o < T.OR.size(); o++) {
                BB ref = orientLegal(T.OR[o], g);
                BB gen = olFull((int)o, g);
                if (ref != gen) bad++;
                // growth-only part: the legal placements touching own cells
                BB grRef = 0;
                if (g.adjOwn) { GenCtx g2 = g; g2.iso = 0; grRef = orientLegal(T.OR[o], g2); }
                if (grRef != olGrow((int)o, g)) bad++;
                tot += 2;
            }
        }
        printf("OLCHECK %ld comparisons, %ld mismatches\n", tot, bad);
    } else if (mode == "hbench") {
        // snek hbench <benchfile>: cost breakdown of horizonValue on positions reached by random
        // play from the benchmark positions to the last two plies
        std::ifstream in(argv[2]);
        std::vector<Pos> roots, hs;
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string tok;
            Pos s = Pos::start();
            while (ls >> tok) { Move m; if (parseMove(tok, m)) s.play(m); }
            roots.push_back(s);
        }
        Rng rng(3);
        std::vector<Move> buf;
        while (hs.size() < 200000) {
            Pos s = roots[rng.below((int)roots.size())];
            while (!s.over() && pieceCount(s.avail[0]) + pieceCount(s.avail[1]) > 2) {
                s.genMoves(buf);
                if (buf.empty()) s.play(Move{}); else s.play(buf[rng.below((int)buf.size())]);
            }
            if (!s.over() && pieceCount(s.avail[s.stm]) == 1 && pieceCount(s.avail[s.stm ^ 1]) == 1) hs.push_back(s);
        }
        auto timeit = [&](const char* name, auto&& f) {
            auto t0 = std::chrono::steady_clock::now();
            long sink = 0;
            for (int rep = 0; rep < 5; rep++) for (auto& s : hs) sink += f(s);
            double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / (5.0 * hs.size());
            printf("%-28s %8.1f ns/pos  (sink %ld)\n", name, ns, sink);
        };
        timeit("horizonValue", [&](const Pos& s) { return horizonValue(s, 0); });
        timeit("genCtxFor x2", [&](const Pos& s) { GenCtx a, b; BB occ = s.occ(); genCtxFor(s.own[0], occ, a); genCtxFor(s.own[1], occ, b); return (long)(a.iso ^ b.allow[1]); });
        timeit("SnakeSet x2", [&](const Pos& s) { GenCtx a; genCtxFor(s.own[0], s.occ(), a); SnakeSet S, S2; S.init(s.comp[0], s.ncomp[0], a); S2.init(s.comp[1], s.ncomp[1], a); return (long)(S.n + S2.n); });
        timeit("bestValue (Y)", [&](const Pos& s) { int Y = s.stm ^ 1; return (long)bestValue(s.own[Y], s.comp[Y], s.ncomp[Y], s.L[Y], onlyPiece(s.avail[Y]), s.occ()); });
        timeit("olFull X all orients", [&](const Pos& s) { GenCtx g; genCtxFor(s.own[s.stm], s.occ(), g); int p = onlyPiece(s.avail[s.stm]); long c = 0; for (int o = T.orBegin[p]; o < T.orBegin[p + 1]; o++) c += popc(olFull(o, g)); return c; });
        timeit("snakeBounds x2", [&](const Pos& s) { int a, b, c, d; snakeBounds(s, 0, a, b); snakeBounds(s, 1, c, d); return (long)(a + b + c + d); });
        timeit("Pos copy+play first move", [&](const Pos& s) { std::vector<Move> mv; s.genMoves(mv); if (mv.empty()) return 0L; Pos c = s; c.play(mv[0]); return (long)c.L[0]; });
        timeit("genMoves", [&](const Pos& s) { std::vector<Move> mv; s.genMoves(mv); return (long)mv.size(); });
    } else if (mode == "mcheck") {
        // snek mcheck <n>: miaiLower(s, opponent) <= the opponent's best single placement after
        // every legal move of the side to move (what the bound promises)
        int n = argc > 2 ? std::atoi(argv[2]) : 20000;
        Rng rng(13);
        long bad = 0, nontriv = 0, checks = 0;
        std::vector<Move> mv;
        for (int i = 0; i < n; i++) {
            Pos s = randomPosition(1 + rng.below(13), rng);
            if (s.over()) continue;
            const int o = s.stm ^ 1;
            if (!s.avail[o]) continue;
            const int m = miaiLower(s, o);
            if (m <= s.L[o]) continue;
            nontriv++;
            s.genMoves(mv);
            if (mv.empty()) mv.push_back(Move{});
            for (Move x : mv) {
                Pos c = s; c.play(x);
                int best = bestValueAny(c.own[o], c.comp[o], c.ncomp[o], c.L[o], c.avail[o], c.own[0] | c.own[1]);
                checks++;
                if (best < m) { bad++; if (bad <= 5) printf("VIOLATION miai %d, after %s best %d  %s\n", m, moveStr(x).c_str(), best, posStr(s).c_str()); }
            }
        }
        printf("MCHECK %ld positions with a non-trivial bound, %ld replies checked: %ld violations\n", nontriv, checks, bad);
    } else if (mode == "bcheck") {
        // snek bcheck <n>: snakeBoundsCap == snakeBounds whenever ub <= cap, > cap otherwise
        int n = argc > 2 ? std::atoi(argv[2]) : 100000;
        Rng rng(9);
        long bad = 0, exact = 0;
        for (int i = 0; i < n; i++) {
            Pos s = randomPosition(rng.below(13), rng);
            for (int c = 0; c < 2; c++) {
                int lb, ub, lb2, ub2;
                snakeBounds(s, c, lb, ub);
                int cap = s.L[c] + rng.below(20) - 3;
                snakeBoundsCap(s, c, cap, lb2, ub2);
                if (lb != lb2 || (ub <= cap ? ub2 != ub : ub2 <= cap)) { bad++; if (bad <= 5) printf("MISMATCH ub %d capped %d cap %d  %s\n", ub, ub2, cap, posStr(s).c_str()); }
                if (ub <= cap) exact++;
            }
        }
        printf("BCHECK %d positions x 2 players (%ld with ub <= cap): %ld mismatches (expected 0 only with -DSB_TIGHT=0)\n", n, exact, bad);
        // soundness of the (tightened) capped bound: never below the final length of any random playout
        long viol = 0, checks = 0;
        std::vector<Move> buf;
        for (int i = 0; i < n / 4; i++) {
            Pos s = randomPosition(rng.below(13), rng);
            int ub[2], lb;
            for (int c = 0; c < 2; c++) snakeBoundsCap(s, c, 999, lb, ub[c]);
            for (int rep = 0; rep < 4; rep++) {
                Pos t = s;
                while (!t.over()) { t.genMoves(buf); t.play(buf.empty() ? Move{} : buf[rng.below((int)buf.size())]); }
                for (int c = 0; c < 2; c++) { checks++; if (t.L[c] > ub[c]) { viol++; if (viol <= 5) printf("VIOLATION player %d final %d > ub %d  %s\n", c, t.L[c], ub[c], posStr(s).c_str()); } }
            }
        }
        printf("BCHECK soundness: %ld playout checks, %ld violations\n", checks, viol);
    } else if (mode == "lcheck") {
        // snek lcheck <n> [seed]: horizonAfter (leaf nodes) == plain search of the child, for every
        // move of n random positions where the side to move has 2 pieces and the opponent 1
        int n = argc > 2 ? std::atoi(argv[2]) : 2000;
        uint64_t seed = argc > 3 ? std::atoll(argv[3]) : 1;
        Rng rng(seed);
        Solver S(20);
        S.useHorizon = false; S.useOnePiece = false; S.fastLeaf = false;
        long done = 0, moves = 0, decided = 0, bad = 0;
        std::vector<Move> mv;
        while (done < n) {
            Pos s = randomPosition(11 + rng.below(3), rng);
            if (s.over() || pieceCount(s.avail[s.stm]) != 2 || pieceCount(s.avail[s.stm ^ 1]) != 1) continue;
            LeafPre pre;
            leafPre(s, pre);
            s.genMoves(mv);
            for (Move m : mv) {
                int hv = horizonAfter(s, m, pre, RED_BONUS);
                moves++;
                if (hv == -9) continue;
                decided++;
                Pos c = s; c.play(m);
                S.clear();
                int v = S.solve(c);
                if (v != hv) { bad++; if (bad <= 10) printf("MISMATCH after %s: leaf %d search %d  %s\n", moveStr(m).c_str(), hv, v, posStr(s).c_str()); }
            }
            done++;
        }
        printf("LCHECK %ld positions, %ld children (%ld decided by the leaf formula): %ld mismatches\n", done, moves, decided, bad);
    } else if (mode == "hcheck") {
        // snek hcheck <n> [seed]: horizonValue (lastply.hpp) == plain search on n random
        // positions where it applies (random play to ply 11..17)
        int n = argc > 2 ? std::atoi(argv[2]) : 10000;
        uint64_t seed = argc > 3 ? std::atoll(argv[3]) : 1;
        Rng rng(seed);
        Solver S(20);
        S.useHorizon = false;
        S.useOnePiece = false;
        int done = 0, bad = 0, cnt[3] = {}, skipped = 0;
        while (done < n) {
            int plies = 12 + rng.below(3);  // later plies are (almost) always game over
            Pos s = randomPosition(plies, rng);
            int hv = horizonValue(s, RED_BONUS);
            if (hv == -9) { skipped++; continue; }
            S.clear();
            int v = S.solve(s);
            cnt[v + 1]++;
            if (v != hv) { bad++; if (bad <= 10) printf("MISMATCH horizon %d search %d  %s\n", hv, v, posStr(s).c_str()); }
            done++;
        }
        printf("HCHECK %d positions (L/D/W %d/%d/%d, %d not applicable): %d mismatches\n", done, cnt[0], cnt[1], cnt[2], skipped, bad);
    } else if (mode == "valset") {
        // snek valset <labels.csv> <n> <threads> [plyMin] [plyMax]: exact solve of n positions
        // (evenly spread over the rows with plyMin <= ply <= plyMax) with a fresh solver each,
        // compared with the stored value. Correctness check + benchmark on mixed results.
        std::ifstream in(argv[2]);
        int n = argc > 3 ? std::atoi(argv[3]) : 100;
        int threads = argc > 4 ? std::atoi(argv[4]) : 3;
        int plyMin = argc > 5 ? std::atoi(argv[5]) : 0, plyMax = argc > 6 ? std::atoi(argv[6]) : 99;
        std::string line;
        auto getl = [&](std::string& l) { if (!std::getline(in, l)) return false; if (!l.empty() && l.back() == '\r') l.pop_back(); return true; };
        getl(line);
        std::vector<std::string> cols;
        { std::stringstream ss(line); std::string c; while (std::getline(ss, c, ',')) cols.push_back(c); }
        int iv = -1, ip = -1;
        for (int i = 0; i < (int)cols.size(); i++) { if (cols[i] == "value") iv = i; if (cols[i] == "pos") ip = i; }
        if (iv < 0 || ip < 0) { fprintf(stderr, "no value/pos column\n"); return 1; }
        std::vector<Pos> all;
        std::vector<int> allV;
        while (getl(line)) {
            std::vector<std::string> f;
            std::stringstream ss(line); std::string c;
            while (std::getline(ss, c, ',')) f.push_back(c);
            if ((int)f.size() != (int)cols.size()) continue;
            Pos s;
            if (!parsePosStr(f[ip], s) || s.ply < plyMin || s.ply > plyMax) continue;
            all.push_back(s); allV.push_back(std::stoi(f[iv]));
        }
        std::vector<Pos> ps;
        std::vector<int> want;
        for (int k = 0; k < n && k < (int)all.size(); k++) {
            size_t i = (size_t)k * all.size() / std::min<size_t>(n, all.size());
            ps.push_back(all[i]); want.push_back(allV[i]);
        }
        std::vector<int> val(ps.size());
        std::vector<uint64_t> nod(ps.size());
        std::atomic<size_t> next{0};
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> th;
        for (int w = 0; w < threads; w++)
            th.emplace_back([&] {
                std::unique_ptr<Solver> SP(new Solver(22));
                for (;;) {
                    size_t i = next++;
                    if (i >= ps.size()) break;
                    SP->clear();
                    val[i] = SP->solve(ps[i]);
                    nod[i] = SP->nodes;
                }
            });
        for (auto& t : th) t.join();
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        uint64_t tn = 0;
        int bad = 0, cnt[3] = {};
        for (size_t i = 0; i < ps.size(); i++) {
            tn += nod[i];
            cnt[want[i] + 1]++;
            if (val[i] != want[i]) { bad++; printf("MISMATCH %zu: got %d want %d  %s\n", i, val[i], want[i], posStr(ps[i]).c_str()); }
        }
        printf("VALSET %zu positions (L/D/W for stm: %d/%d/%d): %d mismatches, %llu nodes, %.2f s wall (%d threads)\n",
               ps.size(), cnt[0], cnt[1], cnt[2], bad, (unsigned long long)tn, wall, threads);
    } else if (mode == "evalexplain") {
        // snek evalexplain <moves...>: feature values and logit contributions for every position
        // (env WEIGHTS=<file> to use a weights file instead of the compiled-in set)
        if (getenv("WEIGHTS")) {
            if (!loadWeights(getenv("WEIGHTS"), 1)) return 1;
            for (int i = 0; i < NFEAT; i++) EW[0][i] = EW[1][i];
            EHEAD[0] = EHEAD[1];
        }
        Pos s = Pos::start();
        std::vector<Move> moves;
        for (int i = 2; i < argc; i++) { Move m; if (!parseMove(argv[i], m)) return 1; moves.push_back(m); }
        printf("%-12s", "feature");
        for (size_t k = 1; k <= moves.size(); k++) printf("  ply%-2zu(%s)", k, k % 2 ? "B" : "R");
        printf("\n");
        std::vector<std::array<double, NFEAT>> xs;
        std::vector<int> stms;
        for (size_t k = 0; k < moves.size(); k++) {
            s.play(moves[k]);
            double x[NFEAT];
            featureVector(s, x);
            std::array<double, NFEAT> a;
            for (int i = 0; i < NFEAT; i++) a[i] = x[i];
            xs.push_back(a);
            stms.push_back(s.stm);
        }
        for (int i = 0; i < NFEAT; i++) {
            printf("%-12s", FEAT_NAMES[i]);
            for (auto& a : xs) printf("  %4.0f %+5.2f", a[i], a[i] * EW[0][i]);
            printf("\n");
        }
        printf("%-12s", "logit(stm)");
        for (auto& a : xs) { double z = 0; for (int i = 0; i < NFEAT; i++) z += a[i] * EW[0][i]; printf("  %10.2f", z); }
        printf("\n%-12s", "red win %");
        for (size_t k = 0; k < xs.size(); k++) { double z = 0; for (int i = 0; i < NFEAT; i++) z += xs[k][i] * EW[0][i]; double p = 1 / (1 + std::exp(-z)); printf("  %9.1f%%", 100 * (stms[k] == 0 ? p : 1 - p)); }
        printf("\n");
    } else if (mode == "evaltrace") {
        // snek evaltrace <maxDepth> <seconds> <threads> <moves...>
        // For every position of the game: static eval and fixed-depth search results,
        // all as red's (player 1's) win probability.
        int maxD = std::atoi(argv[2]);
        double secs = std::atof(argv[3]);
        int threads = std::atoi(argv[4]);
        int exactFrom = getenv("EXACT_FROM") ? std::atoi(getenv("EXACT_FROM")) : 99;
        double exactSecs = getenv("EXACT_SECS") ? std::atof(getenv("EXACT_SECS")) : 30;
        Pos s = Pos::start();
        std::vector<Move> moves;
        for (int i = 5; i < argc; i++) { Move m; if (!parseMove(argv[i], m)) { printf("bad move %s\n", argv[i]); return 1; } moves.push_back(m); }
        std::vector<double> uiSecs;
        if (getenv("UI_SECS")) { std::stringstream ss(getenv("UI_SECS")); std::string t; while (std::getline(ss, t, ',')) uiSecs.push_back(std::atof(t.c_str())); }
        printf("ply stm  static");
        for (double us : uiSecs) printf("  ui%.0fs  ", us);
        for (int d = 1; d <= maxD; d++) printf("   d%d  ", d);
        printf("  exact   move played\n");
        for (size_t k = 0; k <= moves.size() && !s.over(); k++) {
            auto red = [&](int score) { double p = scoreToWinProb(score); return s.stm == 0 ? p : 1 - p; };
            printf("%3d  %s  %5.1f%%", (int)s.ply, s.stm == 0 ? "R" : "B", 100 * red(evaluate(s)));
            if (s.ply > 0 && uiSecs.size())
                for (double us : uiSecs) {  // what the UI shows: time-limited search, smoothed score
                    AbAnalysis a = abAnalyze(s, us, threads, 1);
                    const RootResult& b = a.moves[0];
                    printf("  %5.1f%%", 100 * red(b.exact ? b.display : b.score));
                }
            if (s.ply > 0)
                for (int d = 1; d <= maxD; d++) {
                    AbAnalysis a = abAnalyze(s, secs, threads, 1, d);
                    printf("  %5.1f%%", 100 * red(a.moves[0].score));
                    if (a.depth < d) printf("(%d)", a.depth);
                }
            if (s.ply >= exactFrom) {
                auto r = parallelSolve(s, exactSecs, threads, 21);
                int v = r.value;
                if (v == -9) printf("     ?  ");
                else { int red = s.stm == 0 ? v : -v; printf("  %5s ", red > 0 ? "R win" : red < 0 ? "B win" : "draw"); }
            } else printf("        ");
            printf("   %s\n", k < moves.size() ? moveStr(moves[k]).c_str() : "");
            fflush(stdout);
            if (k < moves.size()) s.play(moves[k]);
        }
    } else if (mode == "speed") {
        // micro-benchmarks of the hot functions on positions from random games
        Rng rng(5);
        std::vector<Pos> ps;
        std::vector<Move> buf;
        for (int g = 0; g < 300; g++) {
            Pos s = Pos::start();
            while (!s.over()) { ps.push_back(s); s.genMoves(buf); s.play(buf.empty() ? Move{} : buf[rng.below((int)buf.size())]); }
        }
        auto bench = [&](const char* name, auto&& f) {
            auto t0 = std::chrono::steady_clock::now();
            long sink = 0;
            int reps = 20;
            for (int r = 0; r < reps; r++) for (auto& p : ps) sink += f(p);
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("%-14s %8.3f us/call   (sink %ld)\n", name, el / (reps * ps.size()) * 1e6, sink);
        };
        bench("genMoves", [&](const Pos& p) { p.genMoves(buf); return (long)buf.size(); });
        bench("hasMove", [&](const Pos& p) { return (long)p.hasMove(); });
        bench("snakeBounds", [&](const Pos& p) { int a, b; snakeBounds(p, 0, a, b); return (long)b; });
        bench("evalHand", [&](const Pos& p) { return (long)evaluateHand(p); });
        bench("evalLearned", [&](const Pos& p) { return (long)evaluate(p); });
    } else if (mode == "refeature") {
        // snek refeature <in.csv> <head>: recompute feature columns from the stored positions
        std::ifstream in(argv[2]);
        int head = argc > 3 ? std::atoi(argv[3]) : 0;
        std::string line;
        std::getline(in, line);
        std::vector<std::string> cols;
        { std::stringstream ss(line); std::string c; while (std::getline(ss, c, ',')) cols.push_back(c); }
        int ig = -1, iv = -1, ip = -1;
        for (int i = 0; i < (int)cols.size(); i++) { if (cols[i] == "game") ig = i; if (cols[i] == "value") iv = i; if (cols[i] == "pos") ip = i; }
        if (ip < 0) { fprintf(stderr, "no pos column\n"); return 1; }
        printDataHeader();
        while (std::getline(in, line)) {
            std::vector<std::string> f;
            std::stringstream ss(line); std::string c;
            while (std::getline(ss, c, ',')) f.push_back(c);
            if ((int)f.size() != (int)cols.size()) continue;
            Pos s;
            if (!parsePosStr(f[ip], s)) continue;
            fputs(dataRow(ig >= 0 ? std::stoi(f[ig]) : 0, s, std::stoi(f[iv]), head).c_str(), stdout);
        }
    } else if (mode == "arena") {
        // engine A vs engine B from random openings (random 1-hex + random reply),
        // every opening played twice with colours swapped.
        int pairs = argc > 2 ? std::atoi(argv[2]) : 10;
        double secs = argc > 3 ? std::atof(argv[3]) : 1;
        int threads = argc > 4 ? std::atoi(argv[4]) : 8;
        std::string A = argc > 5 ? argv[5] : "ab", B = argc > 6 ? argv[6] : "mcts";
        uint64_t seed = argc > 7 ? std::atoll(argv[7]) : 2024;
        int randomPlies = argc > 8 ? std::atoi(argv[8]) : 2;  // random moves at the start of each opening
        Solver solver(22);
        Rng rng(seed);
        int winA = 0, winB = 0, draws = 0;
        std::vector<Move> buf;
        for (int p = 0; p < pairs; p++) {
            Pos open = Pos::start();
            for (int k = 0; k < randomPlies; k++) { open.genMoves(buf); open.play(buf[rng.below((int)buf.size())]); }
            for (int sw = 0; sw < (A == B ? 1 : 2); sw++) {  // identical engines: the swapped game would be the same
                std::string e[2] = {sw ? B : A, sw ? A : B};
                Pos s = open;
                std::string rec = moveStr(Move{}) ;
                rec.clear();
                while (!s.over()) {
                    Analysis a = analyze(s, secs, threads, solver, e[s.stm]);
                    rec += moveStr(a.items[0].mv) + " ";
                    s.play(a.items[0].mv);
                }
                int l0 = longest(s.own[0]), l1 = longest(s.own[1]);
                const char* res = l0 > l1 ? "P1" : l0 < l1 ? "P2" : "draw";
                if (l0 == l1) draws++;
                else if ((l0 > l1) == (sw == 0)) winA++;
                else winB++;
                printf("pair %d  P1=%s P2=%s  %d-%d  %s   | A %d  B %d  draws %d\n", p, e[0].c_str(), e[1].c_str(), l0, l1, res, winA, winB, draws);
                printf("   %s\n", rec.c_str());
                fflush(stdout);
            }
        }
        printf("FINAL %s %d : %s %d  (draws %d)\n", A.c_str(), winA, B.c_str(), winB, draws);
    } else if (mode == "info") {
        for (int p = 0; p < NPIECE; p++) printf("%s: %zu placements\n", PNAME[p], T.PL[p].size());
        printf("total placements %d\n", T.NGID);
    } else if (mode == "solvebench") {
        int plies = argc > 2 ? std::atoi(argv[2]) : 10;
        int count = argc > 3 ? std::atoi(argv[3]) : 10;
        double secs = argc > 4 ? std::atof(argv[4]) : 60;
        int threads = argc > 5 ? std::atoi(argv[5]) : 1;
        std::atomic<int> next{0};
        std::mutex mu;
        std::vector<std::thread> th;
        for (int w = 0; w < threads; w++)
            th.emplace_back([&, w] {
                Solver solver(21);
                for (;;) {
                    int i = next++;
                    if (i >= count) break;
                    Rng rng(1000 + plies * 100000 + i);
                    Pos s = randomPosition(plies, rng);
                    solver.clear();
                    auto t0 = std::chrono::steady_clock::now();
                    int v = solver.solve(s, secs);
                    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    std::lock_guard<std::mutex> lk(mu);
                    printf("plies=%d pos=%d value(stm)=%s nodes=%llu time=%.3fs knps=%.0f\n", plies, i,
                           v == -9 ? "timeout" : std::to_string(v).c_str(), (unsigned long long)solver.nodes, el,
                           solver.nodes / el / 1000);
                    fflush(stdout);
                }
            });
        for (auto& t : th) t.join();
    } else if (mode == "selfplay") {
        int games = argc > 2 ? std::atoi(argv[2]) : 1;
        double secs = argc > 3 ? std::atof(argv[3]) : 1;
        int threads = argc > 4 ? std::atoi(argv[4]) : 8;
        Solver solver(22);
        for (int g = 0; g < games; g++) {
            Pos s = Pos::start();
            std::string rec;
            while (!s.over()) {
                Analysis a = analyze(s, secs, threads, solver);
                Move m = a.items[0].mv;
                rec += moveStr(m) + " ";
                printf("ply %d stm %d: %s  (%s %s)\n", (int)s.ply, (int)s.stm, moveStr(m).c_str(),
                       a.exact ? "exact" : "mcts",
                       a.exact ? std::to_string(a.value).c_str() : std::to_string(a.items[0].w).c_str());
                fflush(stdout);
                s.play(m);
            }
            printf("game %d: P1 %d  P2 %d\nmoves: %s\n", g, longest(s.own[0]), longest(s.own[1]), rec.c_str());
        }
    } else if (mode == "serve") {
        int threads = argc > 2 ? std::atoi(argv[2]) : std::max(1u, std::thread::hardware_concurrency() - 2);
        serve(threads);
    }
    return 0;
}
