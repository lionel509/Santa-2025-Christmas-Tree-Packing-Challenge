/*
 * Santa 2025 Christmas Tree Packing Optimizer
 * 
 * Single C++ program that:
 * - Loads/saves solutions from test.csv
 * - Backs up to /data before each iteration
 * - Runs optimization iterations
 * - Accepts better solutions (lower score)
 * - Interactive loop with user confirmation
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// Data structures
// ============================================================================

struct Tree {
    std::string id;
    double x;
    double y;
    double deg;
};

struct Solution {
    std::vector<Tree> trees;
};

// ============================================================================
// Global PRNG
// ============================================================================

std::mt19937 g_rng;

// ============================================================================
// Utility: Parse "s123.45" -> 123.45
// ============================================================================

double parse_s_value(const std::string& s) {
    if (s.empty() || s[0] != 's') {
        throw std::runtime_error("Expected 's' prefix in value: " + s);
    }
    return std::stod(s.substr(1));
}

std::string format_s_value(double val) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << val;
    return "s" + oss.str();
}

// ============================================================================
// CSV I/O
// ============================================================================

Solution load_solution_from_csv(const std::string& path) {
    Solution sol;
    std::ifstream infile(path);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::string line;
    // Skip header
    if (!std::getline(infile, line)) {
        throw std::runtime_error("Empty CSV file: " + path);
    }

    while (std::getline(infile, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string id, x_str, y_str, deg_str;

        if (!std::getline(ss, id, ',')) continue;
        if (!std::getline(ss, x_str, ',')) continue;
        if (!std::getline(ss, y_str, ',')) continue;
        if (!std::getline(ss, deg_str, ',')) continue;

        Tree tree;
        tree.id = id;
        tree.x = parse_s_value(x_str);
        tree.y = parse_s_value(y_str);
        tree.deg = parse_s_value(deg_str);

        sol.trees.push_back(tree);
    }

    infile.close();
    return sol;
}

void save_solution_to_csv(const std::string& path, const Solution& sol) {
    std::ofstream outfile(path);
    if (!outfile.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + path);
    }

    // Write header
    outfile << "id,x,y,deg\n";

    // Write trees
    for (const auto& tree : sol.trees) {
        outfile << tree.id << ","
                << format_s_value(tree.x) << ","
                << format_s_value(tree.y) << ","
                << format_s_value(tree.deg) << "\n";
    }

    outfile.close();
}

// ============================================================================
// Scoring function (lower is better)
// ============================================================================

double compute_score(const Solution& sol) {
    // Simple score: sum of squared distances from origin
    // This gives preference to solutions that are more compact around (0,0)
    double score = 0.0;
    for (const auto& tree : sol.trees) {
        score += (tree.x * tree.x + tree.y * tree.y);
    }
    return score;
}

// ============================================================================
// Backup function
// ============================================================================

std::string get_timestamp_string() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
    
    #ifdef _WIN32
        localtime_s(&tm_now, &time_t_now);
    #else
        localtime_r(&time_t_now, &tm_now);
    #endif

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y%m%d-%H%M%S");
    return oss.str();
}

void backup_test_csv_to_data() {
    const std::string source = "test.csv";
    
    if (!fs::exists(source)) {
        std::cout << "[backup] test.csv does not exist; skipping backup.\n";
        return;
    }

    // Ensure data directory exists (relative to current directory)
    const std::string data_dir = "data";
    if (!fs::exists(data_dir)) {
        std::cout << "[backup] Creating directory: " << data_dir << "\n";
        fs::create_directories(data_dir);
    }

    // Create backup filename with timestamp
    std::string timestamp = get_timestamp_string();
    std::string backup_path = data_dir + "/test_backup_" + timestamp + ".csv";

    try {
        fs::copy_file(source, backup_path, fs::copy_options::overwrite_existing);
        std::cout << "[backup] Backed up to: " << backup_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[backup] Error: " << e.what() << "\n";
    }
}

// ============================================================================
// Optimization function
// ============================================================================

Solution optimize_once(const Solution& current) {
    Solution candidate = current;

    if (candidate.trees.empty()) {
        // If no trees, create a random one
        Tree tree;
        tree.id = "tree_0";
        tree.x = 0.0;
        tree.y = 0.0;
        tree.deg = 0.0;
        candidate.trees.push_back(tree);
        return candidate;
    }

    // Strategy: randomly pick a tree and perturb its position or rotation
    std::uniform_int_distribution<size_t> tree_dist(0, candidate.trees.size() - 1);
    std::uniform_int_distribution<int> action_dist(0, 2);
    std::uniform_real_distribution<double> delta_dist(-5.0, 5.0);
    std::uniform_real_distribution<double> rotation_dist(-15.0, 15.0);

    size_t idx = tree_dist(g_rng);
    int action = action_dist(g_rng);

    switch (action) {
        case 0:
            // Perturb x
            candidate.trees[idx].x += delta_dist(g_rng);
            break;
        case 1:
            // Perturb y
            candidate.trees[idx].y += delta_dist(g_rng);
            break;
        case 2:
            // Perturb rotation
            candidate.trees[idx].deg += rotation_dist(g_rng);
            // Normalize to [0, 360)
            while (candidate.trees[idx].deg < 0) candidate.trees[idx].deg += 360.0;
            while (candidate.trees[idx].deg >= 360.0) candidate.trees[idx].deg -= 360.0;
            break;
    }

    return candidate;
}

// ============================================================================
// Main loop
// ============================================================================

int main() {
    std::cout << "==========================================\n";
    std::cout << "Santa 2025 Optimization System\n";
    std::cout << "==========================================\n\n";

    // Initialize PRNG
    std::random_device rd;
    g_rng.seed(rd());

    // Load or create initial solution
    Solution current_best;
    double best_score;

    const std::string test_csv_path = "test.csv";

    if (fs::exists(test_csv_path)) {
        std::cout << "[init] Loading existing test.csv...\n";
        try {
            current_best = load_solution_from_csv(test_csv_path);
            best_score = compute_score(current_best);
            std::cout << "[init] Loaded " << current_best.trees.size() 
                      << " trees with score: " << best_score << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[init] Error loading test.csv: " << e.what() << "\n";
            std::cerr << "[init] Creating new solution...\n";
            // Create a simple initial solution
            Tree tree;
            tree.id = "initial_0";
            tree.x = 0.0;
            tree.y = 0.0;
            tree.deg = 0.0;
            current_best.trees.push_back(tree);
            best_score = compute_score(current_best);
            save_solution_to_csv(test_csv_path, current_best);
        }
    } else {
        std::cout << "[init] test.csv not found. Creating initial solution...\n";
        // Create a simple initial solution
        Tree tree;
        tree.id = "initial_0";
        tree.x = 0.0;
        tree.y = 0.0;
        tree.deg = 0.0;
        current_best.trees.push_back(tree);
        best_score = compute_score(current_best);
        save_solution_to_csv(test_csv_path, current_best);
        std::cout << "[init] Created initial solution with score: " << best_score << "\n";
    }

    // Main optimization loop
    int iteration = 0;
    int accepted_count = 0;

    while (true) {
        iteration++;
        std::cout << "\n==========================================\n";
        std::cout << "Iteration " << iteration << "\n";
        std::cout << "==========================================\n";

        // Step 1: Backup test.csv
        backup_test_csv_to_data();

        // Step 2: Generate candidate solution
        Solution candidate = optimize_once(current_best);
        double candidate_score = compute_score(candidate);

        std::cout << "[candidate] score = " << std::fixed << std::setprecision(6) 
                  << candidate_score << ", [best] = " << best_score << "\n";

        // Step 3: Accept if better
        if (candidate_score < best_score) {
            best_score = candidate_score;
            current_best = candidate;
            
            try {
                save_solution_to_csv(test_csv_path, current_best);
                accepted_count++;
                std::cout << "[ACCEPT] New best score: " << best_score 
                          << " (test.csv updated)\n";
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Could not save solution: " << e.what() << "\n";
            }
        } else {
            std::cout << "[reject] Candidate not better.\n";
        }

        std::cout << "\nAcceptance rate: " << accepted_count << "/" << iteration 
                  << " (" << (100.0 * accepted_count / iteration) << "%)\n";

        // Step 4: Ask user to continue
        std::cout << "\nPress ENTER to run another iteration, or type 'q' to quit: ";
        std::string line;
        std::getline(std::cin, line);
        
        if (!line.empty()) {
            char c = std::tolower(line[0]);
            if (c == 'q') {
                std::cout << "\n==========================================\n";
                std::cout << "Stopping optimization loop.\n";
                std::cout << "==========================================\n";
                break;
            }
        }
    }

    // Final summary
    std::cout << "\n==========================================\n";
    std::cout << "Final Summary\n";
    std::cout << "==========================================\n";
    std::cout << "Total iterations: " << iteration << "\n";
    std::cout << "Accepted improvements: " << accepted_count << "\n";
    std::cout << "Final best score: " << std::fixed << std::setprecision(6) 
              << best_score << "\n";
    std::cout << "Solution saved in: " << test_csv_path << "\n";
    std::cout << "==========================================\n";

    return 0;
}
