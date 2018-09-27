#include "gtest/gtest.h"

#include "ATen/ATen.h"
#include "ATen/DLConvertor.h"

#include <iostream>
#include <string.h>
#include <sstream>
#include "test_seed.h"

using namespace at;

TEST(TestParallel, TestParallel) {
  manual_seed(123, at::kCPU);
  set_num_threads(1);

  Tensor a = rand({1, 3});
  a[0][0] = 1;
  a[0][1] = 0;
  a[0][2] = 0;
  Tensor as = rand({3});
  as[0] = 1;
  as[1] = 0;
  as[2] = 0;
  ASSERT_TRUE(a.sum(0).equal(as));
}
