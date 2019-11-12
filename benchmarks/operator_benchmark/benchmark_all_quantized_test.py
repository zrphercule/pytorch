from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import operator_benchmark as op_bench
from pt import ( # noqa
    # Quantized tests
    qactivation_test, qarithmetic_test, qconv_test, qlinear_test,  qpool_test  # noqa
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
