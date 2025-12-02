/*
 * Santa 2025 Super-Powered Optimizer
 * Based on Tree Packer v21 & Ensemble strategies
 * Features:
 * - Multi-threaded Simulated Annealing (OpenMP)
 * - Swap, Translate, and Rotate moves
 * - Squeeze & Compaction local search
 * - Continuous improvement loop with backups
 * - Robust collision detection
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <map>
#include <string>
#include <sstream>
#include <random>
#include <tuple>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <cstring>

using namespace std;
namespace fs = std::filesystem;

// Timing for hourly checkpoints
auto last_checkpoint_time = chrono::steady_clock::now();

// ==========================================
// Constants & Geometry Definitions
// ==========================================

constexpr int MAX_N = 200;
constexpr int NV = 15; // Number of vertices in the tree polygon
constexpr double PI = 3.14159265358979323846;

// Tree Polygon Template (Relative Coordinates)
alignas(64) const double TX[NV] = {
    0.0, 0.125, 0.0625, 0.2, 0.1, 0.35, 0.075, 0.075,
    -0.075, -0.075, -0.35, -0.1, -0.2, -0.0625, -0.125
};

alignas(64) const double TY[NV] = {
    0.8, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0, -0.2,
    -0.2, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5
};

// Fast Random Number Generator
struct FastRNG {
    uint64_t s[2];
    FastRNG(uint64_t seed = 42) {
        s[0] = seed ^ 0x853c49e6748fea9bULL;
        s[1] = (seed * 0x9e3779b97f4a7c15ULL) ^ 0xc4ceb9fe1a85ec53ULL;
    }
    inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t s0 = s[0], s1 = s[1], r = s0 + s1;
        s1 ^= s0;
        s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        s[1] = rotl(s1, 37);
        return r;
    }
    inline double rf() { return (next() >> 11) * 0x1.0p-53; } // [0, 1)
    inline double rf2() { return rf() * 2.0 - 1.0; }         // [-1, 1)
    inline int ri(int n) { return next() % n; }
    inline double gaussian() {
        double u1 = rf() + 1e-10, u2 = rf();
        return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    }
};

// Polygon Structure
struct Poly {
    double px[NV], py[NV];
    double x0, y0, x1, y1; // Bounding box
};

// Transform Template to Actual Coordinates
inline void getPoly(double cx, double cy, double deg, Poly& q) {
    double rad = deg * (PI / 180.0);
    double s = sin(rad), c = cos(rad);
    double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
    
    for (int i = 0; i < NV; i++) {
        // Rotate and Translate
        double x = TX[i] * c - TY[i] * s + cx;
        double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x;
        q.py[i] = y;
        
        // Update BBox
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
    }
    q.x0 = minx; q.y0 = miny; q.x1 = maxx; q.y1 = maxy;
}

// Point in Polygon Test
inline bool pip(double px, double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        if ((q.py[i] > py) != (q.py[j] > py) &&
            px < (q.px[j] - q.px[i]) * (py - q.py[i]) / (q.py[j] - q.py[i]) + q.px[i])
            in = !in;
        j = i;
    }
    return in;
}

// Segment Intersection Test
inline bool segInt(double ax, double ay, double bx, double by,
                   double cx, double cy, double dx, double dy) {
    double d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
    double d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
    double d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    double d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
}

// Polygon Overlap Test
inline bool overlap(const Poly& a, const Poly& b) {
    // Bounding box check first for speed
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    
    // Check points of A in B
    for (int i = 0; i < NV; i++) {
        if (pip(a.px[i], a.py[i], b)) return true;
        if (pip(b.px[i], b.py[i], a)) return true;
    }
    
    // Check edge intersections
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (segInt(a.px[i], a.py[i], a.px[ni], a.py[ni],
                       b.px[j], b.py[j], b.px[nj], b.py[nj])) return true;
        }
    }
    return false;
}

// Configuration Structure
struct Cfg {
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    double gx0, gy0, gx1, gy1; // Global BBox

    // Update polygon for tree i
    inline void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }
    
    // Update all polygons and global bbox
    inline void updAll() { 
        for (int i = 0; i < n; i++) upd(i); 
        updGlobal(); 
    }

    inline void updGlobal() {
        gx0 = gy0 = 1e9;
        gx1 = gy1 = -1e9;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }

    inline bool hasOvl(int i) const {
        for (int j = 0; j < n; j++)
            if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }

    inline bool anyOvl() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return true;
        return false;
    }

    inline double side() const { return max(gx1 - gx0, gy1 - gy0); }
    inline double score() const { double s = side(); return s * s / n; }
    
    // Find trees touching the boundary (candidates for optimization)
    void getBoundary(vector<int>& b) const {
        b.clear();
        double eps = 0.01;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 - gx0 < eps || gx1 - pl[i].x1 < eps ||
                pl[i].y0 - gy0 < eps || gy1 - pl[i].y1 < eps)
                b.push_back(i);
        }
    }
};

// ==========================================
// Optimization Strategies
// ==========================================

// Swap two trees
bool swapTrees(Cfg& c, int i, int j) {
    if (i == j || i >= c.n || j >= c.n) return false;
    swap(c.x[i], c.x[j]);
    swap(c.y[i], c.y[j]);
    swap(c.a[i], c.a[j]);
    c.upd(i);
    c.upd(j);
    // Check overlaps for involved trees
    // Ideally check all, but local check is faster heuristic
    return !c.hasOvl(i) && !c.hasOvl(j);
}

// Squeeze: Linearly shrink the arrangement towards the center
Cfg squeeze(Cfg c) {
    double cx = (c.gx0 + c.gx1) / 2.0;
    double cy = (c.gy0 + c.gy1) / 2.0;
    // Try shrinking from 0.9995 down to 0.98
    for (double scale = 0.9995; scale >= 0.98; scale -= 0.0005) {
        Cfg trial = c;
        for (int i = 0; i < c.n; i++) {
            trial.x[i] = cx + (c.x[i] - cx) * scale;
            trial.y[i] = cy + (c.y[i] - cy) * scale;
        }
        trial.updAll();
        if (!trial.anyOvl()) c = trial; // Success, keep shrinking
        else break; // Collision, stop
    }
    return c;
}

// Compaction: Push individual trees towards the center
Cfg compaction(Cfg c, int iters) {
    double bs = c.side();
    for (int it = 0; it < iters; it++) {
        double cx = (c.gx0 + c.gx1) / 2.0;
        double cy = (c.gy0 + c.gy1) / 2.0;
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            double ox = c.x[i], oy = c.y[i];
            double dx = cx - c.x[i];
            double dy = cy - c.y[i];
            double d = sqrt(dx*dx + dy*dy);
            if (d < 1e-6) continue;
            
            // Try various step sizes
            for (double step : {0.02, 0.008, 0.003, 0.001, 0.0004}) {
                c.x[i] = ox + dx/d * step; 
                c.y[i] = oy + dy/d * step;
                c.upd(i);
                
                if (!c.hasOvl(i)) {
                    c.updGlobal();
                    if (c.side() < bs - 1e-12) {
                        bs = c.side();
                        improved = true;
                        ox = c.x[i]; 
                        oy = c.y[i];
                    } else {
                        // No improvement in score, revert to prev valid pos
                        // or keep if valid? Strategy: Keep valid compaction
                        // ox = c.x[i]; oy = c.y[i]; // keep
                        // Revert if no score improvement to prevent drift?
                        // Actually, packing tighter is usually good even if score doesn't drop immediately
                        // But here we stick to strict improvement for safety
                        c.x[i] = ox; c.y[i] = oy; c.upd(i);
                    }
                } else {
                    // Overlap, revert
                    c.x[i] = ox; c.y[i] = oy; c.upd(i);
                }
            }
        }
        c.updGlobal();
        if (!improved) break;
    }
    return c;
}

// Local Search: Randomized small moves
Cfg localSearch(Cfg c, int maxIter) {
    double bs = c.side();
    const double steps[] = {0.01, 0.004, 0.0015, 0.0006, 0.00025, 0.0001};
    const double rots[] = {5.0, 2.0, 0.8, 0.3, 0.1};
    const int dx[] = {1,-1,0,0,1,1,-1,-1};
    const int dy[] = {0,0,1,-1,1,-1,1,-1};

    for (int iter = 0; iter < maxIter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            // Move towards center logic
            double cx = (c.gx0 + c.gx1) / 2, cy = (c.gy0 + c.gy1) / 2;
            double ddx = cx - c.x[i], ddy = cy - c.y[i];
            double dist = sqrt(ddx*ddx + ddy*ddy);
            
            if (dist > 1e-6) {
                for (double st : steps) {
                    double ox = c.x[i], oy = c.y[i];
                    c.x[i] += ddx/dist * st; c.y[i] += ddy/dist * st; c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); }
                    } else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            
            // Random direction moves
            for (double st : steps) {
                for (int d = 0; d < 8; d++) {
                    double ox = c.x[i], oy = c.y[i];
                    c.x[i] += dx[d]*st; c.y[i] += dy[d]*st; c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); }
                    } else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            
            // Rotation moves
            for (double rt : rots) {
                for (double da : {rt, -rt}) {
                    double oa = c.a[i]; 
                    c.a[i] += da;
                    while (c.a[i] < 0) c.a[i] += 360;
                    while (c.a[i] >= 360) c.a[i] -= 360;
                    c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12) { bs = c.side(); improved = true; }
                        else { c.a[i]=oa; c.upd(i); c.updGlobal(); }
                    } else { c.a[i]=oa; c.upd(i); }
                }
            }
        }
        if (!improved) break;
    }
    return c;
}

// Simulated Annealing Optimization
Cfg sa_opt(Cfg c, int iter, double T0, double Tm, uint64_t seed) {
    FastRNG rng(seed);
    Cfg best = c, cur = c;
    double bs = best.side(), cs = bs, T = T0;
    double alpha = pow(Tm / T0, 1.0 / iter);
    int noImp = 0;

    for (int it = 0; it < iter; it++) {
        int mt = rng.ri(11); // Move Type
        double sc = T / T0;
        bool valid = true;

        if (mt == 0) { // Random Gaussian Move
            int i = rng.ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            cur.x[i] += rng.gaussian() * 0.5 * sc;
            cur.y[i] += rng.gaussian() * 0.5 * sc;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (mt == 1) { // Move to Center
            int i = rng.ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            double bcx = (cur.gx0+cur.gx1)/2, bcy = (cur.gy0+cur.gy1)/2;
            double dx = bcx - cur.x[i], dy = bcy - cur.y[i];
            double d = sqrt(dx*dx + dy*dy);
            if (d > 1e-6) { cur.x[i] += dx/d * rng.rf() * 0.6 * sc; cur.y[i] += dy/d * rng.rf() * 0.6 * sc; }
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (mt == 2) { // Large Rotation
            int i = rng.ri(c.n);
            double oa = cur.a[i];
            cur.a[i] += rng.gaussian() * 80 * sc;
            while (cur.a[i] < 0) cur.a[i] += 360;
            while (cur.a[i] >= 360) cur.a[i] -= 360;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.a[i]=oa; cur.upd(i); valid=false; }
        }
        else if (mt == 10 && c.n > 1) { // Swap Move
            int i = rng.ri(c.n), j = rng.ri(c.n);
            Cfg old = cur;
            if (!swapTrees(cur, i, j)) {
                cur = old;
                valid = false;
            }
        }
        else { // Small Perturbations
            int i = rng.ri(c.n);
            double ox=cur.x[i], oy=cur.y[i];
            cur.x[i] += rng.rf2() * 0.002; 
            cur.y[i] += rng.rf2() * 0.002; 
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }

        if (!valid) { noImp++; T *= alpha; if (T < Tm) T = Tm; continue; }

        cur.updGlobal();
        double ns = cur.side();
        double delta = ns - cs;

        if (delta < 0 || rng.rf() < exp(-delta / T)) {
            cs = ns;
            if (ns < bs) { bs = ns; best = cur; noImp = 0; }
            else noImp++;
        } else {
            cur = best; cs = bs; noImp++;
        }
        
        // Reheat if stuck
        if (noImp > 200) { T = min(T * 5.0, T0); noImp = 0; }
        T *= alpha;
        if (T < Tm) T = Tm;
    }
    return best;
}

// Perturbation for restarting
Cfg perturb(Cfg c, double str, FastRNG& rng) {
    Cfg original = c;
    int np = max(1, (int)(c.n * 0.08 + str * 3));
    for (int k = 0; k < np; k++) {
        int i = rng.ri(c.n);
        c.x[i] += rng.gaussian() * str * 0.5;
        c.y[i] += rng.gaussian() * str * 0.5;
        c.a[i] += rng.gaussian() * 30;
        while (c.a[i] < 0) c.a[i] += 360;
        while (c.a[i] >= 360) c.a[i] -= 360;
    }
    c.updAll();
    if (c.anyOvl()) return original; // If invalid, return original (or try to fix)
    return c;
}

// ==========================================
// File I/O
// ==========================================

map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) { cerr << "Cannot open " << fn << endl; return cfg; }
    string ln; getline(f, ln); // Skip header
    map<int, vector<tuple<int, double, double, double>>> data;
    while (getline(f, ln)) {
        if (ln.empty()) continue;
        auto p1 = ln.find(','), p2 = ln.find(',', p1 + 1), p3 = ln.find(',', p2 + 1);
        if (p1 == string::npos || p2 == string::npos || p3 == string::npos) continue;
        
        string id = ln.substr(0, p1);
        string xs = ln.substr(p1 + 1, p2 - p1 - 1);
        string ys = ln.substr(p2 + 1, p3 - p2 - 1);
        string ds = ln.substr(p3 + 1);
        
        // Handle 's' prefix
        if (!xs.empty() && xs[0] == 's') xs = xs.substr(1);
        if (!ys.empty() && ys[0] == 's') ys = ys.substr(1);
        if (!ds.empty() && ds[0] == 's') ds = ds.substr(1);
        
        int n = stoi(id.substr(0, 3));
        int idx = stoi(id.substr(4));
        data[n].push_back({idx, stod(xs), stod(ys), stod(ds)});
    }
    
    for (auto& [n, v] : data) {
        Cfg c; c.n = n;
        for (auto& [i, x, y, d] : v) {
            if (i < n) { c.x[i] = x; c.y[i] = y; c.a[i] = d; }
        }
        c.updAll();
        cfg[n] = c;
    }
    return cfg;
}

void saveCSV(const string& fn, const map<int, Cfg>& cfg) {
    ofstream f(fn);
    f << fixed << setprecision(15) << "id,x,y,deg\n";
    for (int n = 1; n <= 200; n++) {
        if (cfg.count(n)) {
            const Cfg& c = cfg.at(n);
            for (int i = 0; i < n; i++) {
                f << setfill('0') << setw(3) << n << "_" << i 
                  << ",s" << c.x[i] << ",s" << c.y[i] << ",s" << c.a[i] << "\n";
            }
        }
    }
}

void ensure_dir(const string& path) {
    if (!fs::exists(path)) fs::create_directory(path);
}

// ==========================================
// Main Loop
// ==========================================

int main(int argc, char** argv) {
    string in = "submission.csv";
    string outDir = "out";
    string solutionsDir = "solutions";
    int si = 20000, nr = 32; // Iterations and Restarts

    ensure_dir(outDir);
    ensure_dir(solutionsDir);

    // Load existing solution
    auto cfg = loadCSV(in);
    if (cfg.empty()) { cerr << "No data loaded!" << endl; return 1; }

    map<int, Cfg> best_so_far = cfg;
    double global_best_score = 0;
    for (const auto& [n, c] : best_so_far) global_best_score += c.score();

    cout << "Tree Packer Super-Powered Optimizer" << endl;
    cout << "Threads: " << omp_get_max_threads() << endl;
    cout << "Starting Score: " << global_best_score << endl;

    int generation = 0;
    
    // Infinite loop for continuous improvement
    while (true) {
        generation++;
        cout << "\n=== Generation " << generation << " ===" << endl;
        
        map<int, Cfg> current_gen = best_so_far;
        bool improved_any = false;

        // Parallelize over N (1 to 200)
        // Note: For heavy loads, dynamic scheduling is good.
        #pragma omp parallel for schedule(dynamic, 1)
        for (int n = 1; n <= 200; n++) {
            if (!current_gen.count(n)) continue;

            Cfg c = current_gen[n];
            double original_score = c.score();
            
            // Thread-local RNG
            int tid = omp_get_thread_num();
            FastRNG rng(42 + generation * 123 + tid * 456 + n);

            // Try multiple restarts/perturbations locally
            Cfg local_best = c;
            
            for (int r = 0; r < nr; r++) {
                Cfg cand = c;
                if (r > 0) {
                    // Perturb slightly to escape local minima
                    cand = perturb(cand, 0.02 + (r%5)*0.01, rng);
                }
                
                // SA Optimization
                cand = sa_opt(cand, si, 2.5, 0.000001, rng.next());
                
                // Fine tuning
                cand = squeeze(cand);
                cand = compaction(cand, 50);
                cand = localSearch(cand, 100);
                
                if (!cand.anyOvl() && cand.score() < local_best.score()) {
                    local_best = cand;
                }
            }

            // Update shared structure if improved
            #pragma omp critical
            {
                if (local_best.score() < best_so_far[n].score() - 1e-9) {
                    double old_s = best_so_far[n].score();
                    double new_s = local_best.score();
                    best_so_far[n] = local_best;
                    improved_any = true;
                    cout << "Improved n=" << n << ": " << old_s << " -> " << new_s << endl;
                }
            }
        }

        if (improved_any) {
            double new_global_score = 0;
            for (const auto& [n, c] : best_so_far) new_global_score += c.score();
            
            double improvement = global_best_score - new_global_score;
            cout << "New Global Score: " << new_global_score << " (Improved by " << improvement << ")" << endl;
            global_best_score = new_global_score;
            
            // Save Main Submission
            saveCSV("submission.csv", best_so_far);
            
            // Get current timestamp for verbose naming
            auto now = chrono::system_clock::now();
            auto now_time_t = chrono::system_clock::to_time_t(now);
            auto now_tm = *localtime(&now_time_t);
            
            // Save every improvement to out/ folder with verbose naming
            stringstream out_ss;
            out_ss << outDir << "/gen" << setfill('0') << setw(6) << generation 
                   << "_score" << fixed << setprecision(6) << global_best_score
                   << "_improved" << setprecision(6) << improvement
                   << "_" << put_time(&now_tm, "%Y%m%d_%H%M%S") << ".csv";
            saveCSV(out_ss.str(), best_so_far);
            cout << "Saved improvement: " << out_ss.str() << endl;
            
            // Hourly checkpoint to solutions/ folder
            auto current_time = chrono::steady_clock::now();
            auto elapsed = chrono::duration_cast<chrono::minutes>(current_time - last_checkpoint_time).count();
            if (elapsed >= 60) {
                stringstream sol_ss;
                sol_ss << solutionsDir << "/checkpoint_" 
                       << put_time(&now_tm, "%Y%m%d_%H%M%S")
                       << "_gen" << setfill('0') << setw(6) << generation
                       << "_score" << fixed << setprecision(6) << global_best_score << ".csv";
                saveCSV(sol_ss.str(), best_so_far);
                cout << "\n*** HOURLY CHECKPOINT SAVED: " << sol_ss.str() << " ***\n" << endl;
                last_checkpoint_time = current_time;
            }
        } else {
            cout << "No improvement this generation. Retrying..." << endl;
        }
    }

    return 0;
}