#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_PUZZLES 200
#define MAX_TREES_PER_PUZZLE 200
#define MAX_ENTRIES 20100
#define POLY_POINTS 15
#define COORD_LIMIT 100.0
#define EPS 1e-9

typedef struct {
    double x, y, deg;
    double px[POLY_POINTS];
    double py[POLY_POINTS];
    double minx, maxx, miny, maxy;
} Tree;

typedef struct {
    int n;
    Tree trees[MAX_TREES_PER_PUZZLE];
    double score;
    double bx0, by0, bx1, by1; /* bounding box of puzzle */
} Puzzle;

typedef struct {
    char id[16];
    int puzzle_idx;
    int local_idx;
} Entry;

static Puzzle puzzles[MAX_PUZZLES];
static Entry entries[MAX_ENTRIES];
static int entry_count = 0;
static int puzzle_count = 0;

static const double base_x[POLY_POINTS] = {
    0.0, 0.125, 0.0625, 0.20, 0.10,
    0.35, 0.075, 0.075, -0.075, -0.075,
    -0.35, -0.10, -0.20, -0.0625, -0.125};

static const double base_y[POLY_POINTS] = {
    0.8, 0.5, 0.5, 0.25, 0.25,
    0.0, 0.0, -0.2, -0.2, 0.0,
    0.0, 0.25, 0.25, 0.5, 0.5};

static uint64_t rng_state = 1;

static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static double rand_double(void) {
    return (xorshift64() >> 11) * (1.0 / 9007199254740992.0); /* /2^53 */
}

static double clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static void compute_polygon(Tree *t) {
    const double rad = t->deg * M_PI / 180.0;
    const double c = cos(rad);
    const double s = sin(rad);
    double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
    for (int i = 0; i < POLY_POINTS; ++i) {
        double gx = c * base_x[i] - s * base_y[i] + t->x;
        double gy = s * base_x[i] + c * base_y[i] + t->y;
        t->px[i] = gx;
        t->py[i] = gy;
        if (gx < minx) minx = gx;
        if (gx > maxx) maxx = gx;
        if (gy < miny) miny = gy;
        if (gy > maxy) maxy = gy;
    }
    t->minx = minx;
    t->maxx = maxx;
    t->miny = miny;
    t->maxy = maxy;
}

static double cross(double ax, double ay, double bx, double by) {
    return ax * by - ay * bx;
}

static int point_on_segment(double px, double py, double ax, double ay, double bx, double by) {
    double crossp = cross(bx - ax, by - ay, px - ax, py - ay);
    if (fabs(crossp) > EPS) return 0;
    double dotp = (px - ax) * (px - bx) + (py - ay) * (py - by);
    return dotp <= EPS;
}

static int segments_intersect(double ax1, double ay1, double ax2, double ay2,
                              double bx1, double by1, double bx2, double by2) {
    double v1x = ax2 - ax1, v1y = ay2 - ay1;
    double v2x = bx2 - bx1, v2y = by2 - by1;
    double c1 = cross(v1x, v1y, bx1 - ax1, by1 - ay1);
    double c2 = cross(v1x, v1y, bx2 - ax1, by2 - ay1);
    double c3 = cross(v2x, v2y, ax1 - bx1, ay1 - by1);
    double c4 = cross(v2x, v2y, ax2 - bx1, ay2 - by1);

    if (((c1 > EPS && c2 < -EPS) || (c1 < -EPS && c2 > EPS)) &&
        ((c3 > EPS && c4 < -EPS) || (c3 < -EPS && c4 > EPS))) {
        return 1; /* proper crossing */
    }

    /* inclusive checks: any touching counts as overlap */
    if (fabs(c1) <= EPS && point_on_segment(bx1, by1, ax1, ay1, ax2, ay2)) return 1;
    if (fabs(c2) <= EPS && point_on_segment(bx2, by2, ax1, ay1, ax2, ay2)) return 1;
    if (fabs(c3) <= EPS && point_on_segment(ax1, ay1, bx1, by1, bx2, by2)) return 1;
    if (fabs(c4) <= EPS && point_on_segment(ax2, ay2, bx1, by1, bx2, by2)) return 1;
    return 0;
}

static int point_in_poly_inclusive(const Tree *poly, double px, double py) {
    int inside = 0;
    for (int i = 0, j = POLY_POINTS - 1; i < POLY_POINTS; j = i++) {
        double xi = poly->px[i], yi = poly->py[i];
        double xj = poly->px[j], yj = poly->py[j];
        if (point_on_segment(px, py, xi, yi, xj, yj)) {
            return 1; /* touching counts as overlap */
        }
        int intersect = ((yi > py) != (yj > py)) &&
                        (px < (xj - xi) * (py - yi) / (yj - yi + 1e-20) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

static int polygons_overlap(const Tree *a, const Tree *b) {
    if (a->maxx < b->minx - EPS || b->maxx < a->minx - EPS) return 0;
    if (a->maxy < b->miny - EPS || b->maxy < a->miny - EPS) return 0;

    /* edge intersections (touching counts as overlap) */
    for (int i = 0; i < POLY_POINTS; ++i) {
        int i2 = (i + 1) % POLY_POINTS;
        double ax1 = a->px[i], ay1 = a->py[i];
        double ax2 = a->px[i2], ay2 = a->py[i2];
        for (int j = 0; j < POLY_POINTS; ++j) {
            int j2 = (j + 1) % POLY_POINTS;
            double bx1 = b->px[j], by1 = b->py[j];
            double bx2 = b->px[j2], by2 = b->py[j2];
            if (segments_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)) {
                return 1;
            }
        }
    }

    /* containment (inclusive) */
    if (point_in_poly_inclusive(a, b->px[0], b->py[0])) return 1;
    if (point_in_poly_inclusive(b, a->px[0], a->py[0])) return 1;
    return 0;
}

static double recompute_puzzle(Puzzle *p) {
    double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
    for (int i = 0; i < p->n; ++i) {
        compute_polygon(&p->trees[i]);
        if (p->trees[i].minx < minx) minx = p->trees[i].minx;
        if (p->trees[i].maxx > maxx) maxx = p->trees[i].maxx;
        if (p->trees[i].miny < miny) miny = p->trees[i].miny;
        if (p->trees[i].maxy > maxy) maxy = p->trees[i].maxy;
    }
    p->bx0 = minx;
    p->by0 = miny;
    p->bx1 = maxx;
    p->by1 = maxy;
    double side = (maxx - minx > maxy - miny) ? (maxx - minx) : (maxy - miny);
    p->score = side * side / (double)p->n;
    return p->score;
}

static int puzzle_first_overlap(const Puzzle *p, int *a_out, int *b_out) {
    for (int i = 0; i < p->n; ++i) {
        for (int j = i + 1; j < p->n; ++j) {
            if (polygons_overlap(&p->trees[i], &p->trees[j])) {
                if (a_out) *a_out = i;
                if (b_out) *b_out = j;
                return 1;
            }
        }
    }
    return 0;
}

static int puzzle_has_overlap(const Puzzle *p) {
    return puzzle_first_overlap(p, NULL, NULL);
}

/* Resolve overlaps by nudging overlapping pairs apart slightly.
   Returns number of nudges performed, or -1 if max_iter exceeded without resolving. */
static int resolve_puzzle_overlaps(Puzzle *p, int max_iter, double step) {
    int moves = 0;
    for (int it = 0; it < max_iter; ++it) {
        int a = -1, b = -1;
        if (!puzzle_first_overlap(p, &a, &b)) {
            if (moves) recompute_puzzle(p);
            return moves;
        }
        Tree *ta = &p->trees[a];
        Tree *tb = &p->trees[b];
        double dx = ta->x - tb->x;
        double dy = ta->y - tb->y;
        double len = sqrt(dx * dx + dy * dy);
        if (len < 1e-9) {
            dx = rand_double() - 0.5;
            dy = rand_double() - 0.5;
            len = sqrt(dx * dx + dy * dy);
        }
        if (len < 1e-9) {
            dx = 1.0;
            dy = 0.0;
            len = 1.0;
        }
        double push = step * (1.0 + rand_double()); /* slight randomness to avoid cycles */
        double mx = dx / len * push;
        double my = dy / len * push;
        ta->x = clamp(ta->x + mx, -COORD_LIMIT, COORD_LIMIT);
        ta->y = clamp(ta->y + my, -COORD_LIMIT, COORD_LIMIT);
        tb->x = clamp(tb->x - mx, -COORD_LIMIT, COORD_LIMIT);
        tb->y = clamp(tb->y - my, -COORD_LIMIT, COORD_LIMIT);
        compute_polygon(ta);
        compute_polygon(tb);
        moves++;
    }
    recompute_puzzle(p);
    return -1;
}

/* Try to clean all puzzles before search; returns total moves or -1 on failure. */
static int resolve_all_overlaps(int max_iter_per_puzzle, double step) {
    int total_moves = 0;
    for (int i = 0; i < puzzle_count; ++i) {
        int r = resolve_puzzle_overlaps(&puzzles[i], max_iter_per_puzzle, step);
        if (r < 0) return -1;
        total_moves += r;
    }
    return total_moves;
}

static int find_any_overlap(int *puzzle_idx, int *a_idx, int *b_idx) {
    for (int i = 0; i < puzzle_count; ++i) {
        int a = -1, b = -1;
        if (puzzle_first_overlap(&puzzles[i], &a, &b)) {
            if (puzzle_idx) *puzzle_idx = i;
            if (a_idx) *a_idx = a;
            if (b_idx) *b_idx = b;
            return 1;
        }
    }
    return 0;
}

static double compute_total_score(void) {
    double total = 0.0;
    for (int i = 0; i < puzzle_count; ++i) {
        total += recompute_puzzle(&puzzles[i]);
    }
    return total;
}

static int read_csv(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        return 0;
    }
    char line[256];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return 0;
    }
    while (fgets(line, sizeof(line), f)) {
        if (entry_count >= MAX_ENTRIES) break;
        char id[32], xs[64], ys[64], ds[64];
        if (sscanf(line, "%31[^,],%63[^,],%63[^,],%63s", id, xs, ys, ds) != 4) {
            continue;
        }
        int puzzle_idx = (id[0] - '0') * 100 + (id[1] - '0') * 10 + (id[2] - '0');
        if (puzzle_idx < 1 || puzzle_idx > MAX_PUZZLES) continue;
        puzzle_idx -= 1;
        if (puzzle_idx + 1 > puzzle_count) puzzle_count = puzzle_idx + 1;
        Puzzle *p = &puzzles[puzzle_idx];
        int local_idx = p->n++;
        if (local_idx >= MAX_TREES_PER_PUZZLE) {
            fprintf(stderr, "Too many trees in puzzle %d\n", puzzle_idx + 1);
            fclose(f);
            return 0;
        }
        entries[entry_count].puzzle_idx = puzzle_idx;
        entries[entry_count].local_idx = local_idx;
        strncpy(entries[entry_count].id, id, sizeof(entries[entry_count].id) - 1);
        entries[entry_count].id[sizeof(entries[entry_count].id) - 1] = '\0';
        entry_count++;

        Tree *t = &p->trees[local_idx];
        t->x = strtod(xs + 1, NULL);
        t->y = strtod(ys + 1, NULL);
        t->deg = strtod(ds + 1, NULL);
    }
    fclose(f);
    for (int i = 0; i < puzzle_count; ++i) {
        recompute_puzzle(&puzzles[i]);
    }
    return 1;
}

static void write_submission(const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "Failed to write %s\n", path);
        return;
    }
    fprintf(f, "id,x,y,deg\n");
    for (int i = 0; i < entry_count; ++i) {
        Entry *e = &entries[i];
        Tree *t = &puzzles[e->puzzle_idx].trees[e->local_idx];
        fprintf(f, "%s,s%.9f,s%.9f,s%.9f\n", e->id, t->x, t->y, t->deg);
    }
    fclose(f);
}

#define KEEP_BEST 10

static void save_best(int counter, double score) {
    static char kept[KEEP_BEST][128];
    static int kept_count = 0;
    int bad_puzzle = -1, a_idx = -1, b_idx = -1;

    if (find_any_overlap(&bad_puzzle, &a_idx, &b_idx)) {
        fprintf(stderr, "Skipped saving best #%04d due to overlap in puzzle %03d between %d and %d\n",
                counter, bad_puzzle + 1, a_idx, b_idx);
        return;
    }

    mkdir("out", 0755);
    char path[128];
    snprintf(path, sizeof(path), "out/best_submission_%04d.csv", counter);
    write_submission(path);
    printf("Saved %s with score %.6f\n", path, score);

    if (kept_count == KEEP_BEST) {
        /* delete oldest */
        remove(kept[0]);
        for (int i = 1; i < KEEP_BEST; ++i) {
            strcpy(kept[i - 1], kept[i]);
        }
        kept_count = KEEP_BEST - 1;
    }
    strcpy(kept[kept_count++], path);
    fflush(stdout);
}

static void timestamp_now(char *buf, size_t len) {
    time_t t = time(NULL);
    struct tm tmv;
    localtime_r(&t, &tmv);
    strftime(buf, len, "%Y%m%d_%H%M%S", &tmv);
}

/* Quick scoring helper for an external CSV (e.g., test.csv baseline). */
static double score_from_csv(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1.0;
    char line[256];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1.0; }

    typedef struct { int n; Tree t[MAX_TREES_PER_PUZZLE]; } P;
    P *tmp = (P *)calloc(MAX_PUZZLES, sizeof(P));
    if (!tmp) { fclose(f); return -1.0; }
    int pcnt = 0;
    while (fgets(line, sizeof(line), f)) {
        char id[32], xs[64], ys[64], ds[64];
        if (sscanf(line, "%31[^,],%63[^,],%63[^,],%63s", id, xs, ys, ds) != 4) continue;
        int pid = (id[0] - '0') * 100 + (id[1] - '0') * 10 + (id[2] - '0');
        if (pid < 1 || pid > MAX_PUZZLES) continue;
        pid -= 1;
        if (pid + 1 > pcnt) pcnt = pid + 1;
        P *p = &tmp[pid];
        int idx = p->n++;
        if (idx >= MAX_TREES_PER_PUZZLE) { free(tmp); fclose(f); return -1.0; }
        p->t[idx].x = strtod(xs + 1, NULL);
        p->t[idx].y = strtod(ys + 1, NULL);
        p->t[idx].deg = strtod(ds + 1, NULL);
        compute_polygon(&p->t[idx]);
    }
    fclose(f);

    double total = 0.0;
    for (int i = 0; i < pcnt; ++i) {
        P *p = &tmp[i];
        if (p->n == 0) continue;
        double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
        for (int j = 0; j < p->n; ++j) {
            if (p->t[j].minx < minx) minx = p->t[j].minx;
            if (p->t[j].maxx > maxx) maxx = p->t[j].maxx;
            if (p->t[j].miny < miny) miny = p->t[j].miny;
            if (p->t[j].maxy > maxy) maxy = p->t[j].maxy;
        }
        double side = (maxx - minx > maxy - miny) ? (maxx - minx) : (maxy - miny);
        total += side * side / (double)p->n;
    }
    free(tmp);
    return total;
}

/* ------------ Extra heuristics inspired by shrink/slide packers ------------ */

/* Try shrinking all trees in a puzzle toward its center by a factor.
   Returns 1 if accepted (delta set), otherwise 0 and puzzle is restored. */
static int try_shrink(Puzzle *p, double factor, double *delta_out) {
    double cx = 0.5 * (p->bx0 + p->bx1);
    double cy = 0.5 * (p->by0 + p->by1);
    Tree backup[MAX_TREES_PER_PUZZLE];
    memcpy(backup, p->trees, sizeof(Tree) * p->n);
    double old_score = p->score;

    for (int i = 0; i < p->n; ++i) {
        p->trees[i].x = cx + (p->trees[i].x - cx) * factor;
        p->trees[i].y = cy + (p->trees[i].y - cy) * factor;
        compute_polygon(&p->trees[i]);
    }
    if (puzzle_has_overlap(p)) {
        memcpy(p->trees, backup, sizeof(Tree) * p->n);
        p->score = old_score;
        return 0;
    }
    double new_score = recompute_puzzle(p);
    if (new_score < old_score - 1e-12) {
        if (delta_out) *delta_out = new_score - old_score;
        return 1;
    }
    memcpy(p->trees, backup, sizeof(Tree) * p->n);
    p->score = old_score;
    return 0;
}

/* Slide a single tree toward the bbox center with binary search. */
static int try_edge_slide(Puzzle *p, int idx, double *delta_out) {
    double cx = 0.5 * (p->bx0 + p->bx1);
    double cy = 0.5 * (p->by0 + p->by1);
    double dx = cx - p->trees[idx].x;
    double dy = cy - p->trees[idx].y;
    double len = sqrt(dx * dx + dy * dy);
    if (len < 1e-9) return 0;
    dx /= len;
    dy /= len;

    Tree backup = p->trees[idx];
    double old_score = p->score;
    double lo = 0.0, hi = 0.25, best = 0.0;
    for (int it = 0; it < 18; ++it) {
        double mid = 0.5 * (lo + hi);
        p->trees[idx].x = backup.x + dx * mid;
        p->trees[idx].y = backup.y + dy * mid;
        compute_polygon(&p->trees[idx]);
        int bad = 0;
        for (int j = 0; j < p->n; ++j) {
            if (j == idx) continue;
            if (polygons_overlap(&p->trees[idx], &p->trees[j])) { bad = 1; break; }
        }
        if (!bad) { best = mid; lo = mid; }
        else hi = mid;
    }
    if (best < 1e-6) {
        p->trees[idx] = backup;
        p->score = old_score;
        return 0;
    }
    p->trees[idx].x = backup.x + dx * best;
    p->trees[idx].y = backup.y + dy * best;
    compute_polygon(&p->trees[idx]);
    double new_score = recompute_puzzle(p);
    if (new_score < old_score - 1e-12) {
        if (delta_out) *delta_out = new_score - old_score;
        return 1;
    }
    p->trees[idx] = backup;
    p->score = old_score;
    return 0;
}

static void run_sa(long long max_iters, double step_xy, double step_deg, double init_temp, double cooling, long long save_every, double freeze_temp, double min_save_delta) {
    double total_score = compute_total_score();
    double best_score = total_score;
    int best_counter = 0;
    printf("Initial score: %.6f\n", total_score);
    save_best(best_counter++, best_score);

    for (long long it = 1;; ++it) {
        if (max_iters > 0 && it > max_iters) break;
        int pid = (int)(rand_double() * puzzle_count);
        if (pid >= puzzle_count) pid = puzzle_count - 1;
        Puzzle *p = &puzzles[pid];
        if (p->n == 0) continue;
        int tid = (int)(rand_double() * p->n);
        if (tid >= p->n) tid = p->n - 1;
        Tree backup = p->trees[tid];
        double old_score = p->score;

        double temp = init_temp / (1.0 + cooling * it);

        /* move types: 0=translate,1=rotate,2=both,3=kick,4=shrink puzzle,5=edge slide */
        int mv = (int)(rand_double() * 6.0);
        double dx = 0.0, dy = 0.0, dd = 0.0;
        if (mv == 0) {
            dx = (rand_double() * 2.0 - 1.0) * step_xy * temp;
            dy = (rand_double() * 2.0 - 1.0) * step_xy * temp;
        } else if (mv == 1) {
            dd = (rand_double() * 2.0 - 1.0) * step_deg * temp;
        } else if (mv == 2) {
            dx = (rand_double() * 2.0 - 1.0) * step_xy * temp;
            dy = (rand_double() * 2.0 - 1.0) * step_xy * temp;
            dd = (rand_double() * 2.0 - 1.0) * step_deg * temp;
        } else if (mv == 3) { /* kick */
            dx = (rand_double() * 2.0 - 1.0) * step_xy * 2.0;
            dy = (rand_double() * 2.0 - 1.0) * step_xy * 2.0;
            dd = (rand_double() * 2.0 - 1.0) * step_deg * 2.0;
        } else if (mv == 4) { /* shrink whole puzzle, greedy */
            double f = 0.985 + rand_double() * 0.012; /* [0.985,0.997] */
            double delta = 0.0;
            if (try_shrink(p, f, &delta)) {
                total_score += delta;
                if (total_score + 1e-9 < best_score) {
                    best_score = total_score;
                    save_best(best_counter++, best_score);
                }
            }
            continue;
        } else { /* mv == 5 edge slide */
            double delta = 0.0;
            if (try_edge_slide(p, tid, &delta)) {
                total_score += delta;
                if (total_score + 1e-9 < best_score) {
                    best_score = total_score;
                    save_best(best_counter++, best_score);
                }
            }
            continue;
        }

        p->trees[tid].x = clamp(p->trees[tid].x + dx, -COORD_LIMIT, COORD_LIMIT);
        p->trees[tid].y = clamp(p->trees[tid].y + dy, -COORD_LIMIT, COORD_LIMIT);
        p->trees[tid].deg += dd;
        if (p->trees[tid].deg >= 360.0 || p->trees[tid].deg <= -360.0) {
            p->trees[tid].deg = fmod(p->trees[tid].deg, 360.0);
        }
        compute_polygon(&p->trees[tid]);

        int bad = 0;
        for (int j = 0; j < p->n; ++j) {
            if (j == tid) continue;
            if (polygons_overlap(&p->trees[tid], &p->trees[j])) {
                bad = 1;
                break;
            }
        }
        if (!bad) {
            double new_score = recompute_puzzle(p);
            double delta = new_score - old_score;
            int accept = 0;
            if (delta <= 0) {
                accept = 1;
            } else if (temp > freeze_temp) {
                double prob = exp(-delta / (temp + 1e-12));
                if (rand_double() < prob) accept = 1;
            }
            if (accept) {
                total_score += delta;
            } else {
                p->trees[tid] = backup;
                recompute_puzzle(p);
            }
        } else {
            p->trees[tid] = backup;
        }

        if (it % 10000 == 0) {
            printf("Iter %lld temp %.6f current %.6f best %.6f\n", it, temp, total_score, best_score);
            fflush(stdout);
        }
        if (total_score + min_save_delta < best_score) {
            best_score = total_score;
            save_best(best_counter++, best_score);
        } else if (save_every > 0 && (it % save_every == 0)) {
            mkdir("out", 0755);
            char path[128];
            char ts[32];
            timestamp_now(ts, sizeof(ts));
            snprintf(path, sizeof(path), "out/checkpoint_%04lld_%s.csv", it / save_every, ts);
            int bad_puzzle = -1, a_idx = -1, b_idx = -1;
            if (find_any_overlap(&bad_puzzle, &a_idx, &b_idx)) {
                fprintf(stderr, "Skipped checkpoint %s due to overlap in puzzle %03d between %d and %d\n",
                        path, bad_puzzle + 1, a_idx, b_idx);
            } else {
                write_submission(path);
                printf("Saved %s (checkpoint) score %.6f\n", path, total_score);
            }
        }
    }
    mkdir("out", 0755);
    int bad_puzzle = -1, a_idx = -1, b_idx = -1;
    if (find_any_overlap(&bad_puzzle, &a_idx, &b_idx)) {
        fprintf(stderr, "Not writing final submission: overlap in puzzle %03d between %d and %d\n",
                bad_puzzle + 1, a_idx, b_idx);
    } else {
        write_submission("out/submission.csv");
    }
}

int main(int argc, char **argv) {
    const char *input = "test.csv";
    const char *baseline = "test.csv";
    long long iters = 200000;
    double step_xy = 0.10;
    double step_deg = 10.0;
    double init_temp = 0.05;
    double cooling = 5e-6;
    long long save_every = 0; /* disabled unless set */
    double freeze_temp = 0.01;
    uint64_t seed = (uint64_t)time(NULL);
    int threads = 1;
    double min_save_delta = 1e-5; /* minimum improvement to write a new best file */

    if (argc > 1) input = argv[1];
    if (argc > 2) iters = atoll(argv[2]);
    if (argc > 3) seed = (uint64_t)atoll(argv[3]);
    if (argc > 4) step_xy = atof(argv[4]);
    if (argc > 5) step_deg = atof(argv[5]);
    if (argc > 6) init_temp = atof(argv[6]);
    if (argc > 7) cooling = atof(argv[7]);
    if (argc > 8) save_every = atoll(argv[8]);
    if (argc > 9) freeze_temp = atof(argv[9]);
    if (argc > 10) threads = atoi(argv[10]);
    if (argc > 11) baseline = argv[11];
    if (argc > 12) min_save_delta = atof(argv[12]);

    rng_state = seed ? seed : 1;

    if (!read_csv(input)) {
        fprintf(stderr, "Failed to load input CSV.\n");
        return 1;
    }
    for (int i = 0; i < puzzle_count; ++i) {
        int a_idx = -1, b_idx = -1;
        if (puzzle_first_overlap(&puzzles[i], &a_idx, &b_idx)) {
            fprintf(stderr, "Warning: puzzle %d has overlapping trees in input (trees %d and %d).\n",
                    i + 1, a_idx, b_idx);
        }
    }
    int resolved = resolve_all_overlaps(500, 0.02);
    if (resolved < 0) {
        fprintf(stderr, "Failed to resolve initial overlaps; results may be invalid.\n");
    } else if (resolved > 0) {
        printf("Resolved %d overlap nudges before SA.\n", resolved);
    }

    if (threads > 1) {
        fprintf(stderr, "Note: current build is single-threaded; threads=%d ignored. Run multiple seeds in parallel processes to use multiple cores.\n", threads);
    }

    double baseline_score = score_from_csv(baseline);
    if (baseline_score > 0) {
        printf("baseline (%s): %.6f\n", baseline, baseline_score);
    } else {
        printf("baseline (%s): unavailable\n", baseline);
    }

    printf("cfg input=%s iters=%lld seed=%llu step_xy=%.4f step_deg=%.4f init_temp=%.4f cooling=%.2e save_every=%lld freeze_temp=%.4f min_save_delta=%.6g threads=%d\n",
           input, iters, (unsigned long long)seed, step_xy, step_deg, init_temp, cooling, save_every, freeze_temp, min_save_delta, threads);

    run_sa(iters, step_xy, step_deg, init_temp, cooling, save_every, freeze_temp, min_save_delta);
    return 0;
}
