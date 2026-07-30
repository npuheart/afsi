"""
AFSI 全量测试运行器

由于 pytest 在多测试函数间与 MPI/PETSc 存在兼容性问题（段错误），
此脚本顺序运行所有测试文件，汇总结果。

运行方式:
    conda activate afsi-dolfinx
    python tests/run_all.py

或跳过耗时的集成测试:
    python tests/run_all.py --skip-integration
"""

import subprocess
import sys
import os
import time
from pathlib import Path

TESTS_DIR = Path(__file__).parent

TEST_SCRIPTS = [
    ("单元测试: 伴随关系", "test_duality.py"),
    ("单元测试: 独立算子", "test_operators.py"),
    ("集成测试: 方腔驱动圆盘", "test_integration.py"),
]

def run_test(name, script):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(TESTS_DIR / script)],
        capture_output=False,
        cwd=str(TESTS_DIR.parent),
        timeout=600,
    )
    elapsed = time.time() - start
    
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"\n  [{status}] {name} ({elapsed:.1f}s)")
    return result.returncode == 0


def main():
    skip_integration = "--skip-integration" in sys.argv
    
    scripts = TEST_SCRIPTS[:]
    if skip_integration:
        scripts = [s for s in scripts if "integration" not in s[1].lower()]
        print("  (跳过集成测试)")
    
    results = {}
    for name, script in scripts:
        ok = run_test(name, script)
        results[name] = ok
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"  测试汇总")
    print(f"{'='*60}")
    passed = sum(results.values())
    total = len(results)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    
    print(f"\n  {passed}/{total} 通过")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
