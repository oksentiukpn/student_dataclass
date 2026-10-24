"""
Benchmarking script for Homework 1.

Tasks:
1. 10-second throughput comparison for Task 1 approaches (Naive, Indexed, Optimized)
   with ratio 10:10:1 on the full dataset (100,000 students).
2. Scaling analysis for the best approach across 10 dataset sizes (100 to 100,000)
   with >= 5 runs per data point:
   - Memory usage (via tracemalloc)
   - Latency per operation (Op1, Op2, Op3) with mean and std dev.
3. Task 2 sorting performance comparison across 10 dataset sizes (Builtin vs Radix).
"""

import json
import os
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Tuple

import numpy as np

from src.generator import generate_operations, get_dataset_sizes, get_student_sample
from src.models import Student, read_data
from src.task1 import StudentDBIndexed, StudentDBNaive, StudentDBOptimized
from src.task2 import save_sorted_to_csv, sort_students_builtin, sort_students_radix


def run_10s_throughput_benchmark(
    students: List[Student], duration_sec: float = 10.0
) -> Dict[str, Any]:
    """Run 10-second throughput test for the 3 approaches on the dataset."""
    print("=" * 60)
    print(f"Running 10-second throughput benchmark (N = {len(students)})...")
    print("=" * 60)

    # Pre-generate a large pool of operations with ratio 10:10:1
    ops_pool = generate_operations(
        students, total_count=500_000, ratio=(10, 10, 1), seed=42
    )
    pool_len = len(ops_pool)

    approaches = [
        ("Naive (List)", StudentDBNaive),
        ("Indexed (Hash + Heap)", StudentDBIndexed),
        ("Optimized (Buckets + Cache)", StudentDBOptimized),
    ]

    results: Dict[str, Any] = {}

    for name, cls in approaches:
        print(f"Initializing {name}...")
        db_instance = cls(students)

        op_counts = {1: 0, 2: 0, 3: 0}
        total_ops = 0

        print(f"Executing operations for {duration_sec} seconds...")
        start_time = time.perf_counter()
        idx = 0

        while True:
            now = time.perf_counter()
            if now - start_time >= duration_sec:
                break

            op = ops_pool[idx]
            op_type = op[0]
            if op_type == 1:
                db_instance.get_top_100_students()
            elif op_type == 2:
                db_instance.set_student_rating(op[1], op[2])
            elif op_type == 3:
                db_instance.get_groups_with_highest_average_rating()

            op_counts[op_type] += 1
            total_ops += 1
            idx = (idx + 1) % pool_len

        elapsed = time.perf_counter() - start_time
        ops_per_sec = total_ops / elapsed
        print(f"Result for {name}: {total_ops} total ops ({ops_per_sec:.1f} ops/sec)")
        print(
            f"   Breakdown: Op1={op_counts[1]}, Op2={op_counts[2]}, Op3={op_counts[3]}"
        )

        results[name] = {
            "total_ops": total_ops,
            "elapsed_sec": elapsed,
            "ops_per_sec": ops_per_sec,
            "breakdown": op_counts,
        }

    return results


def run_scaling_benchmark_optimized(
    all_students: List[Student], sizes: List[int], runs_per_size: int = 5
) -> Dict[str, Any]:
    """Measure memory usage and latency per operation across 10 dataset sizes for StudentDBOptimized."""
    print("=" * 60)
    print(
        f"Running scaling benchmark for StudentDBOptimized ({runs_per_size} runs per size)..."
    )
    print("=" * 60)

    scaling_results: Dict[str, Any] = {
        "sizes": sizes,
        "memory_mb": [],
        "op1_mean_us": [],
        "op1_std_us": [],
        "op2_mean_us": [],
        "op2_std_us": [],
        "op3_mean_us": [],
        "op3_std_us": [],
    }

    for size in sizes:
        sub_students = get_student_sample(all_students, size)

        # 1. Measure memory usage
        tracemalloc.start()
        _mem_db = StudentDBOptimized(sub_students)
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_mb = current_mem / (1024 * 1024)
        scaling_results["memory_mb"].append(mem_mb)

        # 2. Measure per-operation latency across runs
        # Op 1: 50 calls per run
        # Op 2: 50 calls per run
        # Op 3: 20 calls per run
        op1_times_us: List[float] = []
        op2_times_us: List[float] = []
        op3_times_us: List[float] = []

        for r in range(runs_per_size):
            db_inst = StudentDBOptimized(sub_students)
            test_ops = generate_operations(
                sub_students, total_count=1000, ratio=(10, 10, 1), seed=100 + r
            )

            # Op 1 measurements
            for _ in range(30):
                t0 = time.perf_counter()
                db_inst.get_top_100_students()
                t1 = time.perf_counter()
                op1_times_us.append((t1 - t0) * 1e6)

            # Op 2 measurements
            op2_items = [op for op in test_ops if op[0] == 2][:30]
            for op in op2_items:
                t0 = time.perf_counter()
                db_inst.set_student_rating(op[1], op[2])
                t1 = time.perf_counter()
                op2_times_us.append((t1 - t0) * 1e6)

            # Op 3 measurements
            for _ in range(20):
                t0 = time.perf_counter()
                db_inst.get_groups_with_highest_average_rating()
                t1 = time.perf_counter()
                op3_times_us.append((t1 - t0) * 1e6)

        scaling_results["op1_mean_us"].append(float(np.mean(op1_times_us)))
        scaling_results["op1_std_us"].append(float(np.std(op1_times_us)))
        scaling_results["op2_mean_us"].append(float(np.mean(op2_times_us)))
        scaling_results["op2_std_us"].append(float(np.std(op2_times_us)))
        scaling_results["op3_mean_us"].append(float(np.mean(op3_times_us)))
        scaling_results["op3_std_us"].append(float(np.std(op3_times_us)))

        print(
            f"Size {size:6d} | Mem: {mem_mb:6.2f} MB | "
            f"Op1: {scaling_results['op1_mean_us'][-1]:6.2f}±{scaling_results['op1_std_us'][-1]:5.2f} µs | "
            f"Op2: {scaling_results['op2_mean_us'][-1]:6.2f}±{scaling_results['op2_std_us'][-1]:5.2f} µs | "
            f"Op3: {scaling_results['op3_mean_us'][-1]:6.2f}±{scaling_results['op3_std_us'][-1]:5.2f} µs"
        )

    return scaling_results


def run_task2_benchmark(
    all_students: List[Student], sizes: List[int], runs_per_size: int = 5
) -> Dict[str, Any]:
    """Compare Builtin Sort vs Radix Sort across dataset sizes."""
    print("=" * 60)
    print("Running Task 2 Sorting benchmark (Builtin vs Radix)...")
    print("=" * 60)

    task2_results: Dict[str, Any] = {
        "sizes": sizes,
        "builtin_mean_ms": [],
        "builtin_std_ms": [],
        "radix_mean_ms": [],
        "radix_std_ms": [],
    }

    for size in sizes:
        sub_students = get_student_sample(all_students, size)

        b_times: List[float] = []
        r_times: List[float] = []

        for _ in range(runs_per_size):
            # Builtin
            t0 = time.perf_counter()
            sort_students_builtin(sub_students)
            t1 = time.perf_counter()
            b_times.append((t1 - t0) * 1000)

            # Radix
            t0 = time.perf_counter()
            sort_students_radix(sub_students)
            t1 = time.perf_counter()
            r_times.append((t1 - t0) * 1000)

        task2_results["builtin_mean_ms"].append(float(np.mean(b_times)))
        task2_results["builtin_std_ms"].append(float(np.std(b_times)))
        task2_results["radix_mean_ms"].append(float(np.mean(r_times)))
        task2_results["radix_std_ms"].append(float(np.std(r_times)))

        print(
            f"Size {size:6d} | "
            f"Builtin: {task2_results['builtin_mean_ms'][-1]:7.3f}±{task2_results['builtin_std_ms'][-1]:6.3f} ms | "
            f"Radix: {task2_results['radix_mean_ms'][-1]:7.3f}±{task2_results['radix_std_ms'][-1]:6.3f} ms"
        )

    # Save sorted full dataset to data/students_sorted.csv
    out_csv = os.path.join(
        os.path.dirname(__file__), "..", "data", "students_sorted.csv"
    )
    print(f"Saving sorted full dataset to {out_csv}...")
    sorted_all = sort_students_radix(all_students)
    save_sorted_to_csv(sorted_all, out_csv)
    print(f"Saved {len(sorted_all)} sorted students to CSV successfully.")

    return task2_results


def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "students.csv")
    print(f"Loading data from {data_path}...")
    students = read_data(data_path)
    print(f"Loaded {len(students)} students.")

    # 1. 10-second throughput test
    t1_results = run_10s_throughput_benchmark(students, duration_sec=10.0)

    # 2. Scaling benchmark for best approach
    sizes = get_dataset_sizes()
    scaling_results = run_scaling_benchmark_optimized(students, sizes, runs_per_size=5)

    # 3. Task 2 sorting benchmark
    task2_results = run_task2_benchmark(students, sizes, runs_per_size=5)

    all_data = {
        "throughput_10s": t1_results,
        "scaling_optimized": scaling_results,
        "task2_sorting": task2_results,
    }

    out_file = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nAll benchmark results successfully saved to {out_file}!")


if __name__ == "__main__":
    main()
