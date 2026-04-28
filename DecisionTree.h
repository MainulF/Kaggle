#pragma once

#include <vector>
#include <map>
#include <memory>
#include <ranges>
#include <algorithm>
#include <cassert>

class DecisionTree {
private:
    struct Node {
        int feature_index = -1;
        double threshold = 0.0;
        double prediction = 0.0;
        bool is_leaf = false;
        std::unique_ptr<Node> left_child;
        std::unique_ptr<Node> right_child;
    };
    struct SplitResult {
        int feature_index = -1;
        double threshold = 0.0;
        double score = 1.0;
        bool valid = false;
    };
    int max_depth_;
    int min_samples_;
    std::unique_ptr<Node> root_;

    static double traverse(const Node* node, const std::vector<double>& sample) {
        if (node->is_leaf) return node->prediction;
        if (sample[node->feature_index] <= node->threshold) {
            return traverse(node->left_child.get(), sample);
        }
        return traverse(node->right_child.get(), sample);
    }
public:
    static double gini(const std::vector<double>& labels) {
        if (labels.empty()) return 0.0;
        std::map<double,int> unique;
        for (auto label : labels) unique[label]++;
        const auto size = static_cast<double>(labels.size());
        double sum = 0.0;
        for (const auto val: unique | std::views::values) {
            const auto p = val / size;
            sum += p * p;
        }
        return 1.0 - sum;
    }
    static double weightedGini(const std::vector<double>& left_labels, const std::vector<double>& right_labels) {
        const auto n_total = static_cast<double>(left_labels.size() + right_labels.size());
        if (n_total == 0) return 0.0;
        const auto w_left = static_cast<double>(left_labels.size()) / n_total;
        const auto w_right = static_cast<double>(right_labels.size()) / n_total;
        return w_left * gini(left_labels) + w_right * gini(right_labels);
    }
    static std::pair<std::vector<std::vector<double>>, std::vector<double>>
        splitLeft(
            const std::vector<std::vector<double>>& X,
            const std::vector<double>& y,
            const int feature_index,
            const double threshold) {
        std::vector<std::vector<double>> X_out;
        std::vector<double> y_out;
        for (int i = 0; i < X.size(); i++) {
            if (X[i][feature_index] <= threshold) {
                X_out.push_back(X[i]);
                y_out.push_back(y[i]);
            }
        }
        return std::make_pair(X_out, y_out);
    }
    static std::pair<std::vector<std::vector<double>>, std::vector<double>>
    splitRight(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const int feature_index,
        const double threshold) {
        std::vector<std::vector<double>> X_out;
        std::vector<double> y_out;
        for (int i = 0; i < X.size(); i++) {
            if (X[i][feature_index] > threshold) {
                X_out.push_back(X[i]);
                y_out.push_back(y[i]);
            }
        }
        return std::make_pair(X_out, y_out);
    }
    static SplitResult bestSplit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        SplitResult best;
        if (X.empty() || X[0].empty()) return best;
        const int n_features = static_cast<int>(X[0].size());
        for (int j = 0; j < n_features; j++) {
            std::vector<double> vals;
            vals.reserve(X.size());
            for (const auto& row : X) vals.push_back(row[j]);
            std::ranges::sort(vals);
            vals.erase(std::ranges::unique(vals).begin(), vals.end());
            for (size_t k = 0; k + 1 < vals.size(); k++) {
                const double threshold = (vals[k] + vals[k + 1]) / 2.0;
                auto [X_left,  y_left]  = splitLeft(X, y, j, threshold);
                auto [X_right, y_right] = splitRight(X, y, j, threshold);
                if (y_left.empty() || y_right.empty()) continue;
                if (const double score = weightedGini(y_left, y_right); score < best.score) {
                    best.feature_index = j;
                    best.threshold = threshold;
                    best.score = score;
                    best.valid = true;
                }
            }
        }
        return best;
    }
    static double leafPrediction(const std::vector<double>& labels) {
        std::map<double, int> counts;
        for (const auto label : labels) counts[label]++;
        const auto winner = std::ranges::max_element(counts,
        [](const auto& a, const auto& b) { return a.second < b.second; });
        return winner->first;
    }
    static std::unique_ptr<Node> buildTree(
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const int depth,
        const int max_depth,
        const int min_samples) {
        if (depth >= max_depth
            || y.size() < static_cast<size_t>(min_samples)
            || gini(y) == 0.0) {
            auto leaf = std::make_unique<Node>();
            leaf->is_leaf = true;
            leaf->prediction = leafPrediction(y);
            return leaf;
        }
        const SplitResult best = bestSplit(X, y);
        if (!best.valid) {
            auto leaf = std::make_unique<Node>();
            leaf->is_leaf = true;
            leaf->prediction = leafPrediction(y);
            return leaf;
        }
        auto [X_left,  y_left]  = splitLeft (X, y, best.feature_index, best.threshold);
        auto [X_right, y_right] = splitRight(X, y, best.feature_index, best.threshold);
        auto node = std::make_unique<Node>();
        node->feature_index = best.feature_index;
        node->threshold     = best.threshold;
        node->left_child  = buildTree(X_left,  y_left,  depth + 1, max_depth, min_samples);
        node->right_child = buildTree(X_right, y_right, depth + 1, max_depth, min_samples);
        return node;
    }
    explicit DecisionTree(const int max_depth = 5, const int min_samples = 2)
        : max_depth_(max_depth), min_samples_(min_samples) {}
    void fit(const std::vector<std::vector<double>>& X,const std::vector<double>& y) {
        root_ = buildTree(X, y, 0, max_depth_, min_samples_);
    }
    [[nodiscard]] double predict(const std::vector<double>& sample) const {
        assert(root_ != nullptr && "DecisionTree is not fitted yet.");
        return traverse(root_.get(), sample);
    }



};