/*
 * Santa 2025 Christmas Tree Packing Optimizer (Plain C, Automatic)
 * 
 * Implements proper Kaggle Santa 2025 scoring:
 * - Real Christmas tree polygon geometry
 * - Bounding square per puzzle: score = Σ(s_n^2 / n)
 * - Loads ALL 20100 trees from test.csv
 * - Groups by puzzle ID (001-200)
 * - Runs automatic optimization passes over ALL puzzles
 * - Accepts improvements (lower global score)
 * - Overwrites test.csv when improvements found
 * - Stops when no improvements in a full pass
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Data structures
 * ============================================================================ */

/* Christmas tree dimensions (in tree units) */
#define TRUNK_W  0.15
#define TRUNK_H  0.20
#define BASE_W   0.70
#define MID_W    0.40
#define TOP_W    0.25
#define TIP_Y    0.80
#define TIER_1_Y 0.50
#define TIER_2_Y 0.25
#define BASE_Y   0.00

/* Tree polygon has 15 vertices */
#define TREE_VERTICES 15

typedef struct {
    double x;
    double y;
} Point;

typedef struct {
    char id[32];        /* e.g. "001_0" */
    char puzzle[16];    /* e.g. "001" */
    int index;          /* e.g. 0 */
    double x;           /* Center x coordinate */
    double y;           /* Center y coordinate */
    double deg;         /* Rotation in degrees */
} Tree;

typedef struct {
    Tree *trees;
    int count;
    int capacity;
} Solution;

typedef struct {
    char puzzle_id[16];
    int start_idx;      /* First tree index for this puzzle */
    int count;          /* Number of trees in this puzzle */
} PuzzleGroup;

typedef struct {
    PuzzleGroup *groups;
    int count;
    int capacity;
} PuzzleList;

/* ============================================================================
 * Christmas Tree Polygon Definition
 * ============================================================================ */

/* Get the local (unrotated, untranslated) Christmas tree polygon */
void get_tree_polygon_local(Point *vertices) {
    double trunk_bottom = -TRUNK_H;
    
    /* Define vertices counter-clockwise from tip */
    int i = 0;
    vertices[i++] = (Point){0.0, TIP_Y};                    /* Tip */
    vertices[i++] = (Point){TOP_W/2, TIER_1_Y};             /* Top tier right */
    vertices[i++] = (Point){TOP_W/4, TIER_1_Y};             /* Top tier inner right */
    vertices[i++] = (Point){MID_W/2, TIER_2_Y};             /* Mid tier right */
    vertices[i++] = (Point){MID_W/4, TIER_2_Y};             /* Mid tier inner right */
    vertices[i++] = (Point){BASE_W/2, BASE_Y};              /* Base right */
    vertices[i++] = (Point){TRUNK_W/2, BASE_Y};             /* Trunk top right */
    vertices[i++] = (Point){TRUNK_W/2, trunk_bottom};       /* Trunk bottom right */
    vertices[i++] = (Point){-TRUNK_W/2, trunk_bottom};      /* Trunk bottom left */
    vertices[i++] = (Point){-TRUNK_W/2, BASE_Y};            /* Trunk top left */
    vertices[i++] = (Point){-BASE_W/2, BASE_Y};             /* Base left */
    vertices[i++] = (Point){-MID_W/4, TIER_2_Y};            /* Mid tier inner left */
    vertices[i++] = (Point){-MID_W/2, TIER_2_Y};            /* Mid tier left */
    vertices[i++] = (Point){-TOP_W/4, TIER_1_Y};            /* Top tier inner left */
    vertices[i++] = (Point){-TOP_W/2, TIER_1_Y};            /* Top tier left */
}

/* Transform tree polygon to world coordinates */
void get_tree_polygon_world(const Tree *tree, Point *world_vertices) {
    Point local[TREE_VERTICES];
    get_tree_polygon_local(local);
    
    /* Convert degrees to radians */
    double rad = tree->deg * M_PI / 180.0;
    double cos_theta = cos(rad);
    double sin_theta = sin(rad);
    
    /* The CSV x,y represent the tree tip position */
    /* Local tip is at (0, TIP_Y), so we need to adjust translation */
    double offset_x = tree->x;
    double offset_y = tree->y - TIP_Y;  /* Adjust so tip lands at (x,y) */
    
    for (int i = 0; i < TREE_VERTICES; i++) {
        /* Rotate */
        double x_rot = local[i].x * cos_theta - local[i].y * sin_theta;
        double y_rot = local[i].x * sin_theta + local[i].y * cos_theta;
        
        /* Translate */
        world_vertices[i].x = x_rot + offset_x;
        world_vertices[i].y = y_rot + offset_y + TIP_Y;  /* Add TIP_Y back */
    }
}

/* ============================================================================
 * Utility functions
 * ============================================================================ */

/* Parse "s123.45" -> 123.45 */
double parse_s_value(const char *s) {
    if (s == NULL || s[0] != 's') {
        fprintf(stderr, "Error: expected 's' prefix in value: %s\n", s ? s : "(null)");
        return 0.0;
    }
    return strtod(s + 1, NULL);
}

/* Format double as "s123.456789" */
void format_s_value(char *buf, size_t bufsize, double val) {
    snprintf(buf, bufsize, "s%.8f", val);
}

/* Extract puzzle and index from id like "001_0" */
void parse_tree_id(const char *id, char *puzzle_out, int *index_out) {
    const char *underscore = strchr(id, '_');
    
    if (underscore == NULL) {
        /* No underscore, treat whole id as puzzle, index 0 */
        strncpy(puzzle_out, id, 15);
        puzzle_out[15] = '\0';
        *index_out = 0;
        return;
    }
    
    /* Copy puzzle part */
    size_t puzzle_len = underscore - id;
    if (puzzle_len > 15) puzzle_len = 15;
    strncpy(puzzle_out, id, puzzle_len);
    puzzle_out[puzzle_len] = '\0';
    
    /* Parse index */
    *index_out = atoi(underscore + 1);
}

/* ============================================================================
 * Solution management
 * ============================================================================ */

Solution* create_solution(void) {
    Solution *sol = malloc(sizeof(Solution));
    sol->trees = NULL;
    sol->count = 0;
    sol->capacity = 0;
    return sol;
}

void free_solution(Solution *sol) {
    if (sol == NULL) return;
    free(sol->trees);
    free(sol);
}

void add_tree(Solution *sol, const char *id, double x, double y, double deg) {
    if (sol->count >= sol->capacity) {
        sol->capacity = sol->capacity == 0 ? 1000 : sol->capacity * 2;
        sol->trees = realloc(sol->trees, sol->capacity * sizeof(Tree));
    }
    
    Tree *t = &sol->trees[sol->count];
    strncpy(t->id, id, sizeof(t->id) - 1);
    t->id[sizeof(t->id) - 1] = '\0';
    parse_tree_id(id, t->puzzle, &t->index);
    t->x = x;
    t->y = y;
    t->deg = deg;
    sol->count++;
}

/* ============================================================================
 * CSV I/O
 * ============================================================================ */

Solution* load_solution(const char *path) {
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: could not open file: %s\n", path);
        return NULL;
    }
    
    Solution *sol = create_solution();
    char line[1024];
    int line_num = 0;
    
    /* Skip header */
    if (fgets(line, sizeof(line), fp) == NULL) {
        fprintf(stderr, "Error: empty CSV file\n");
        fclose(fp);
        free_solution(sol);
        return NULL;
    }
    line_num++;
    
    /* Read trees */
    while (fgets(line, sizeof(line), fp) != NULL) {
        line_num++;
        
        /* Remove trailing newline */
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        if (strlen(line) == 0) continue;
        
        /* Parse CSV: id,x,y,deg */
        char id[256], x_str[64], y_str[64], deg_str[64];
        if (sscanf(line, "%255[^,],%63[^,],%63[^,],%63s", id, x_str, y_str, deg_str) != 4) {
            fprintf(stderr, "Warning: could not parse line %d: %s\n", line_num, line);
            continue;
        }
        
        double x = parse_s_value(x_str);
        double y = parse_s_value(y_str);
        double deg = parse_s_value(deg_str);
        
        add_tree(sol, id, x, y, deg);
    }
    
    fclose(fp);
    return sol;
}

int save_solution(const char *path, const Solution *sol) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: could not open file for writing: %s\n", path);
        return -1;
    }
    
    /* Write header */
    fprintf(fp, "id,x,y,deg\n");
    
    /* Write trees */
    for (int i = 0; i < sol->count; i++) {
        char x_str[64], y_str[64], deg_str[64];
        format_s_value(x_str, sizeof(x_str), sol->trees[i].x);
        format_s_value(y_str, sizeof(y_str), sol->trees[i].y);
        format_s_value(deg_str, sizeof(deg_str), sol->trees[i].deg);
        
        fprintf(fp, "%s,%s,%s,%s\n", 
                sol->trees[i].id, x_str, y_str, deg_str);
    }
    
    fclose(fp);
    return 0;
}

/* ============================================================================
 * Scoring function (Kaggle Santa 2025 metric)
 * ============================================================================ */

/* Compute bounding square for a set of trees */
double compute_puzzle_bounding_square(const Solution *sol, int start_idx, int count) {
    if (count == 0) return 0.0;
    
    double min_x = 1e9, max_x = -1e9;
    double min_y = 1e9, max_y = -1e9;
    
    /* For each tree in this puzzle */
    for (int i = start_idx; i < start_idx + count; i++) {
        Point vertices[TREE_VERTICES];
        get_tree_polygon_world(&sol->trees[i], vertices);
        
        /* Find bounding box of this tree's polygon */
        for (int v = 0; v < TREE_VERTICES; v++) {
            if (vertices[v].x < min_x) min_x = vertices[v].x;
            if (vertices[v].x > max_x) max_x = vertices[v].x;
            if (vertices[v].y < min_y) min_y = vertices[v].y;
            if (vertices[v].y > max_y) max_y = vertices[v].y;
        }
    }
    
    double width = max_x - min_x;
    double height = max_y - min_y;
    double side = (width > height) ? width : height;
    
    return side;
}

/* Compute global score = Σ(s_n^2 / n) over all puzzles */
double compute_score(const Solution *sol) {
    if (sol->count == 0) return 0.0;

    /* Build puzzle list (sequential) */
    PuzzleList *plist = malloc(sizeof(PuzzleList));
    plist->groups = NULL;
    plist->count = 0;
    plist->capacity = 0;

    for (int i = 0; i < sol->count; i++) {
        const char *puzzle = sol->trees[i].puzzle;
        int found = -1;
        for (int j = 0; j < plist->count; j++) {
            if (strcmp(plist->groups[j].puzzle_id, puzzle) == 0) {
                found = j;
                break;
            }
        }
        if (found == -1) {
            if (plist->count >= plist->capacity) {
                plist->capacity = plist->capacity == 0 ? 200 : plist->capacity * 2;
                plist->groups = realloc(plist->groups, plist->capacity * sizeof(PuzzleGroup));
            }
            strncpy(plist->groups[plist->count].puzzle_id, puzzle, sizeof(plist->groups[plist->count].puzzle_id) - 1);
            plist->groups[plist->count].puzzle_id[sizeof(plist->groups[plist->count].puzzle_id) - 1] = '\0';
            plist->groups[plist->count].start_idx = i;
            plist->groups[plist->count].count = 1;
            plist->count++;
        } else {
            plist->groups[found].count++;
        }
    }

    double total_score = 0.0;

#ifdef _OPENMP
    /* Parallelize per-puzzle bounding square computation */
    #pragma omp parallel for schedule(dynamic) reduction(+:total_score)
#endif
    for (int i = 0; i < plist->count; i++) {
        double side = compute_puzzle_bounding_square(sol, plist->groups[i].start_idx, plist->groups[i].count);
        double n = (double)plist->groups[i].count;
        total_score += (side * side) / n;
    }

    free(plist->groups);
    free(plist);
    return total_score;
}

/* ============================================================================
 * Backup function
 * ============================================================================ */

int backup_test_csv(void) {
    const char *source = "test.csv";
    const char *backup_dir = "data";
    const char *backup_path = "data/test_backup.csv";
    
    /* Check if source exists */
    FILE *test = fopen(source, "r");
    if (test == NULL) {
        printf("[backup] test.csv does not exist; skipping backup.\n");
        return 0;
    }
    fclose(test);
    
    /* Ensure data directory exists */
    struct stat st = {0};
    if (stat(backup_dir, &st) == -1) {
        printf("[backup] Creating directory: %s\n", backup_dir);
        if (mkdir(backup_dir, 0755) != 0) {
            fprintf(stderr, "[backup] Error creating directory: %s\n", strerror(errno));
            return -1;
        }
    }
    
    /* Copy file */
    FILE *src = fopen(source, "rb");
    if (src == NULL) {
        fprintf(stderr, "[backup] Error opening source: %s\n", strerror(errno));
        return -1;
    }
    
    FILE *dst = fopen(backup_path, "wb");
    if (dst == NULL) {
        fprintf(stderr, "[backup] Error opening destination: %s\n", strerror(errno));
        fclose(src);
        return -1;
    }
    
    /* Copy contents */
    char buffer[4096];
    size_t bytes;
    while ((bytes = fread(buffer, 1, sizeof(buffer), src)) > 0) {
        fwrite(buffer, 1, bytes, dst);
    }
    
    fclose(src);
    fclose(dst);
    
    printf("[backup] test.csv -> %s\n", backup_path);
    return 0;
}

/* ============================================================================
 * Puzzle grouping
 * ============================================================================ */

PuzzleList* build_puzzle_list(const Solution *sol) {
    PuzzleList *plist = malloc(sizeof(PuzzleList));
    plist->groups = NULL;
    plist->count = 0;
    plist->capacity = 0;
    
    /* Find unique puzzles and their ranges */
    for (int i = 0; i < sol->count; i++) {
        const char *puzzle = sol->trees[i].puzzle;
        
        /* Check if we already have this puzzle */
        int found = -1;
        for (int j = 0; j < plist->count; j++) {
            if (strcmp(plist->groups[j].puzzle_id, puzzle) == 0) {
                found = j;
                break;
            }
        }
        
        if (found == -1) {
            /* New puzzle */
            if (plist->count >= plist->capacity) {
                plist->capacity = plist->capacity == 0 ? 200 : plist->capacity * 2;
                plist->groups = realloc(plist->groups, plist->capacity * sizeof(PuzzleGroup));
            }
            
            strncpy(plist->groups[plist->count].puzzle_id, puzzle, sizeof(plist->groups[plist->count].puzzle_id) - 1);
            plist->groups[plist->count].puzzle_id[sizeof(plist->groups[plist->count].puzzle_id) - 1] = '\0';
            plist->groups[plist->count].start_idx = i;
            plist->groups[plist->count].count = 1;
            plist->count++;
        } else {
            /* Existing puzzle - increment count */
            plist->groups[found].count++;
        }
    }
    
    return plist;
}

void free_puzzle_list(PuzzleList *plist) {
    if (plist == NULL) return;
    free(plist->groups);
    free(plist);
}

/* ============================================================================
 * Optimization
 * ============================================================================ */

/* Try to improve a single puzzle by perturbing one tree */
int optimize_puzzle(Solution *sol, const PuzzleGroup *pg, double *best_score) {
    int start = pg->start_idx;
    int end = start + pg->count;
    
    if (pg->count == 0) return 0;
    
    /* Try perturbing each tree in this puzzle */
    for (int i = start; i < end; i++) {
        Tree *t = &sol->trees[i];
        
        /* Save original values */
        double orig_x = t->x;
        double orig_y = t->y;
        double orig_deg = t->deg;
        
        /* CONTINUOUS SPACE: Try various step sizes for fine-tuning and escaping local minima */
        /* Use smaller steps for fine-tuning, larger for exploration */
        double position_deltas[] = {
            -2.0, -1.0, -0.5, -0.1, -0.05, -0.01, -0.001,  /* Negative steps */
            0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0           /* Positive steps */
        };
        int num_pos_deltas = sizeof(position_deltas) / sizeof(position_deltas[0]);
        
        /* Try position changes (x and y independently and together) */
        for (int dx_idx = 0; dx_idx < num_pos_deltas; dx_idx++) {
            for (int dy_idx = 0; dy_idx < num_pos_deltas; dy_idx++) {
                /* Apply perturbation */
                double new_x = orig_x + position_deltas[dx_idx];
                double new_y = orig_y + position_deltas[dy_idx];
                
                /* Clamp to valid continuous range [-100, 100] */
                if (new_x < -100.0) new_x = -100.0;
                if (new_x > 100.0) new_x = 100.0;
                if (new_y < -100.0) new_y = -100.0;
                if (new_y > 100.0) new_y = 100.0;
                
                t->x = new_x;
                t->y = new_y;
                
                /* Compute new score */
                double new_score = compute_score(sol);
                
                /* Accept ANY improvement, even tiny ones (continuous optimization) */
                if (new_score < *best_score) {
                    double improvement = *best_score - new_score;
                    printf("[opt] puzzle %s: %.10f -> %.10f (tree %s moved by %.6f,%.6f, Δ=%.10f)\n",
                           pg->puzzle_id, *best_score, new_score, t->id, 
                           position_deltas[dx_idx], position_deltas[dy_idx], improvement);
                    *best_score = new_score;
                    return 1;
                }
            }
        }
        
        /* Try rotation changes (continuous angles) */
        double rotation_deltas[] = {
            -30.0, -15.0, -5.0, -1.0, -0.5, -0.1, -0.01,   /* Counter-clockwise */
            0.01, 0.1, 0.5, 1.0, 5.0, 15.0, 30.0           /* Clockwise */
        };
        int num_rot_deltas = sizeof(rotation_deltas) / sizeof(rotation_deltas[0]);
        
        for (int r = 0; r < num_rot_deltas; r++) {
            /* Reset position, only change rotation */
            t->x = orig_x;
            t->y = orig_y;
            
            double new_deg = orig_deg + rotation_deltas[r];
            
            /* Normalize angle to [0, 360) - continuous range */
            while (new_deg < 0.0) new_deg += 360.0;
            while (new_deg >= 360.0) new_deg -= 360.0;
            
            t->deg = new_deg;
            
            double new_score = compute_score(sol);
            
            /* Accept ANY improvement in continuous space */
            if (new_score < *best_score) {
                double improvement = *best_score - new_score;
                printf("[opt] puzzle %s: %.10f -> %.10f (tree %s rotated by %.6f°, Δ=%.10f)\n",
                       pg->puzzle_id, *best_score, new_score, t->id, 
                       rotation_deltas[r], improvement);
                *best_score = new_score;
                return 1;
            }
        }
        
        /* Revert to original if no improvement found */
        t->x = orig_x;
        t->y = orig_y;
        t->deg = orig_deg;
    }
    
    return 0;
}

/* Run one pass over all puzzles, trying to improve each */
int optimize_all_puzzles(Solution *sol, double *best_score) {
    PuzzleList *plist = build_puzzle_list(sol);
    int any_improved = 0;

    printf("[opt] Processing %d puzzles (parallel capable)...\n", plist->count);

#ifdef _OPENMP
    /* Parallel region: each thread works on different puzzles.
       Uses a local copy for tentative modifications; applies only accepted moves. */
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < plist->count; i++) {
        PuzzleGroup pg = plist->groups[i];
        /* Local best snapshot */
        double local_best_start;
        /* We will attempt a single improvement per puzzle per pass (like original logic). */
        /* Create lightweight copy of solution for trial modifications (only trees of puzzle). */
        int start = pg.start_idx;
        int end = start + pg.count;
        for (int ti = start; ti < end; ti++) {
            Tree *t = &sol->trees[ti];
            double ox = t->x, oy = t->y, od = t->deg;
            double pos_steps[] = {-0.5, -0.1, -0.01, 0.01, 0.1, 0.5};
            int ns = (int)(sizeof(pos_steps)/sizeof(pos_steps[0]));
            int accepted = 0;
            for (int sx = 0; sx < ns && !accepted; sx++) {
                for (int sy = 0; sy < ns && !accepted; sy++) {
                    double nx = ox + pos_steps[sx];
                    double ny = oy + pos_steps[sy];
                    if (nx < -100.0) nx = -100.0; if (nx > 100.0) nx = 100.0;
                    if (ny < -100.0) ny = -100.0; if (ny > 100.0) ny = 100.0;
                    t->x = nx; t->y = ny; t->deg = od;
                    double trial_score = compute_score(sol);
                    #pragma omp critical
                    {
                        if (trial_score < *best_score) {
                            double delta = *best_score - trial_score;
                            printf("[opt-par] puzzle %s: %.10f -> %.10f (tree %s moved Δx=%.4f Δy=%.4f Δ=%.10f)\n",
                                   pg.puzzle_id, *best_score, trial_score, t->id, nx-ox, ny-oy, delta);
                            *best_score = trial_score;
                            any_improved = 1;
                            accepted = 1;
                        }
                    }
                    if (!accepted) { t->x = ox; t->y = oy; }
                }
            }
            if (accepted) break; /* move to next puzzle */
        }
    }
#else
    for (int i = 0; i < plist->count; i++) {
        PuzzleGroup *pg = &plist->groups[i];
        int improved = optimize_puzzle(sol, pg, best_score);
        if (improved) any_improved = 1;
    }
#endif

    free_puzzle_list(plist);
    return any_improved;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("==========================================\n");
    printf("Santa 2025 Automatic Optimization System\n");
    printf("Kaggle Competition: Christmas Tree Packing\n");
    printf("==========================================\n\n");
    
    /* Load initial solution */
    printf("[init] Loading test.csv...\n");
    Solution *solution = load_solution("test.csv");
    if (solution == NULL) {
        fprintf(stderr, "Error: failed to load test.csv\n");
        return 1;
    }
    printf("[init] Loaded %d trees\n", solution->count);
    
    /* Count puzzles */
    PuzzleList *plist = build_puzzle_list(solution);
    printf("[init] Found %d distinct puzzles\n", plist->count);
    free_puzzle_list(plist);
    
    /* Backup test.csv */
    if (backup_test_csv() != 0) {
        fprintf(stderr, "Warning: backup failed, continuing anyway\n");
    }
    
    /* Compute initial score */
    double best_score = compute_score(solution);
    printf("[init] Initial score: %.10f\n", best_score);
#ifdef _OPENMP
    printf("[init] OpenMP enabled with %d threads\n", omp_get_max_threads());
#else
    printf("[init] OpenMP not enabled (compile with -fopenmp to use all cores)\n");
#endif
    
    /* Optimization loop */
    int iteration = 0;
    int total_improvements = 0;
    
    while (1) {
        iteration++;
        printf("\n========================================\n");
        printf("Global iteration %d\n", iteration);
        printf("========================================\n");
        
        double prev_score = best_score;
        int improved = optimize_all_puzzles(solution, &best_score);
        printf("[loop] best_score after iteration %d: %.10f", iteration, best_score);
        
        if (improved) {
            total_improvements++;
            double improvement = prev_score - best_score;
            double improvement_pct = (improvement / prev_score) * 100.0;
            printf(" (improved by %.10f, %.6f%%)\n", improvement, improvement_pct);
        } else {
            printf(" (no change)\n");
        }
        
        if (!improved) {
            printf("No improvements found in this pass. Stopping.\n");
            break;
        }
        
        /* Save improved solution */
        if (save_solution("test.csv", solution) != 0) {
            fprintf(stderr, "Error: failed to save improved test.csv\n");
            break;
        }
        /* Also save to submission.csv for Kaggle submission */
        if (save_solution("submission.csv", solution) != 0) {
            fprintf(stderr, "Error: failed to save submission.csv\n");
        } else {
            printf("[save] Updated test.csv and submission.csv with improved solution.\n");
        }
    }
    
    /* Final summary */
    printf("\n==========================================\n");
    printf("Final Summary\n");
    printf("==========================================\n");
    printf("Total iterations: %d\n", iteration);
    printf("Iterations with improvements: %d\n", total_improvements);
    printf("Final best score: %.10f\n", best_score);
    printf("Solution saved in: test.csv (and submission.csv)\n");
    printf("Backup available at: data/test_backup.csv\n");
    printf("==========================================\n");
    
    free_solution(solution);
    return 0;
}
