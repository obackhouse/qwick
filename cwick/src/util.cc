/*
 *  C++ specific utilities
 */

#include "util.h"

std::string format_float(const double val, const bool trailing_zero) {
    // Formats a floating point-number, optionally with a trailing
    // zero for integer values

    std::stringstream stream;
    stream << std::fixed << std::setprecision(TRAILING_ZERO_PRECISION) << val;
    std::string out = stream.str();

    unsigned int i = out.find_last_not_of("0") + 1;
    if (out.find_last_of(".") == (i-1)) {
        if (!(trailing_zero)) {
            i -= 1;
        } else {
            i += 1;
        }
    }

    out.erase(i, std::string::npos);

    return out;
}
