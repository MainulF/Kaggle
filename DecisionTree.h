#pragma once

#include <vector>
#include <map>
#include <memory>
#include <ranges>

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

};


