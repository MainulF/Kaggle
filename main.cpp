#include <print>
#include "DecisionTree.h"

int main() {
    std::vector<double> vec = {1.0, 1.0};
    std::print("{}",DecisionTree::gini(vec));
}