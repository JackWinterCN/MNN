#include "log_utilities.hpp"
#include <cstring>

const char* const_basename(const char* filepath) {
  const char* base = strrchr(filepath, '/');
#ifdef GLOG_OS_WINDOWS  // Look for either path separator in Windows
  if (!base) base = strrchr(filepath, '\\');
#endif
  return base ? (base + 1) : filepath;
}