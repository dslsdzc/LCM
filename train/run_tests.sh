#!/bin/sh
# LCM 全量测试：C 推理引擎 + 全部 Python 回归套件。
# 用法: train/run_tests.sh    (从仓库根调用; python 位于 be/bin/python)
set -e
cd "$(dirname "$0")/.."

echo "=== [1/2] C 推理引擎 (make test) ==="
make -C infer test

echo
echo "=== [2/2] Python 回归套件 ==="
export JAX_PLATFORMS=cpu
for t in test_causal_mask test_fixes_core test_fixes_model test_fixes_aux \
         test_fixes_cython test_fixes_sweep2 test_fixes_sweep3 test_core_math; do
    echo "--- train.$t"
    be/bin/python -m train.$t
done

echo
echo "ALL SUITES PASSED"
