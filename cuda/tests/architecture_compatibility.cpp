#include "meep_cuda/detail/compiled_architectures.hpp"

#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

bool reference_compatibility(int capability) {
  for (std::size_t i = 1; i < meep_cuda::detail::compiled_real_architecture_count; ++i)
    if (meep_cuda::detail::compiled_real_architectures[i] == capability) return true;

  for (std::size_t i = 1; i < meep_cuda::detail::compiled_virtual_architecture_count; ++i)
    if (meep_cuda::detail::compiled_virtual_architectures[i] <= capability) return true;

  return false;
}

} // namespace

int main() {
  require(!meep_cuda::detail::compute_capability_is_compiled(-1, 0),
          "negative major must be incompatible");
  require(!meep_cuda::detail::compute_capability_is_compiled(8, -1),
          "negative minor must be incompatible");
  require(!meep_cuda::detail::compute_capability_is_compiled(8, 10),
          "two-digit minor must be rejected");

  for (int capability = 50; capability <= 120; ++capability) {
    const bool actual = meep_cuda::detail::compute_capability_is_compiled(
        capability / 10, capability % 10);
    require(actual == reference_compatibility(capability),
            "compiled architecture predicate differs from generated metadata");
  }

  std::cout << "PASS: CUDA compatibility predicate matches compiled fatbin metadata\n";
  return 0;
}
