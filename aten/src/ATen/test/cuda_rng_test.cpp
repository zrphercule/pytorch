#define CATCH_CONFIG_MAIN
#include "catch_utils.hpp"

#include "ATen/ATen.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include <thread>

void makeRandomNumber() {
  cudaSetDevice(std::rand() % 2);
  auto x = at::randn({1000});
}

void testCudaRNGMultithread() {
  auto threads = std::vector<std::thread>();
  for (auto i = 0; i < 1000; i++) {
    threads.emplace_back(makeRandomNumber);
  }
  for (auto& t : threads) {
    t.join();
  }
};

CATCH_TEST_CASE( "CUDA RNG test", "[cuda]" ) {
  CATCH_SECTION( "multithread" )
    testCudaRNGMultithread();
}
