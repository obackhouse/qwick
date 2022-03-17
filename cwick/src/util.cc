/*
 *  C++ specific utilities
 */

#include "util.h"

#include <string>
#include <iomanip>
#include <sstream>

std::string format_float(double val, bool trailing_zero) {
    // Formats a floating point-number, optionally with a trailing
    // zero for integer values

    std::stringstream stream;
    stream << std::fixed << std::setprecision(10) << val;
    std::string out = stream.str();

    unsigned int i = out.find_last_not_of("0") + 1;
    if (out.find_last_of(".") == (i-1)) {
        if (!(trailing_zero)) {
            i -= 1;
        }
        else {
            i += 1;
        }
    }
    
    out.erase(i, std::string::npos);

    return out;
}
