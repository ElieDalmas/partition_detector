#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ostream>

// assert check, on fail => print and exit 1
#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "CHECK failed: " << #cond << "  (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            std::exit(1); \
        } \
    } while (0)

inline bool equal(double a, double b, double eps = 1e-6) {
    return std::abs(a - b) < eps;
}
