"""
Analytics, Linear Regression, and Visualization for Homework 1.

Generates:
1. Linear regression model T(n) = c1*f1(n) + c2*f2(n) + c3*f3(n) for the slowest operation.
2. Formatted tables of experimental results and comparison.
3. High-resolution publication-quality plots:
   - plots/ops_10s_throughput.png
   - plots/latency_operations.png
   - plots/slowest_op_regression.png
   - plots/memory_usage.png
   - plots/task2_sorting_comparison.png
"""

import json
import math
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def perform_linear_regression(
    sizes: List[int], times: List[float]
) -> Tuple[np.ndarray, float, str]:
    """
    Fits T(n) = c1 * f1(n) + c2 * f2(n) + c3 * f3(n)
    Basis functions:
    f1(n) = 1 (constant base overhead)
    f2(n) = log2(n) (logarithmic tree / search factor)
    f3(n) = n (linear scan factor)
    Returns: coefficients (c1, c2, c3), R^2 score, formula string.
    """
    n_arr = np.array(sizes, dtype=float)
    y_arr = np.array(times, dtype=float)

    # Design matrix: [1, log2(n), n]
    f1 = np.ones_like(n_arr)
    f2 = np.log2(n_arr)
    f3 = n_arr

    X = np.column_stack([f1, f2, f3])

    # Ordinary Least Squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y_arr, rcond=None)
    c1, c2, c3 = coeffs

    y_pred = X @ coeffs
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    formula_str = f"T(n) = {c1:.4e} + {c2:.4e} * log2(n) + {c3:.4e} * n"
    return coeffs, r2, formula_str


def plot_throughput_comparison(data: Dict[str, Any], output_dir: str):
    names = list(data.keys())
    ops_10s = [data[k]["total_ops"] for k in names]
    ops_per_sec = [data[k]["ops_per_sec"] for k in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        names, ops_10s, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="black"
    )

    ax.set_ylabel("Total Operations in 10s (Ratio 10:10:1)", fontsize=12)
    ax.set_title(
        "Task 1: 10-Second Throughput by Approach (N = 100,000)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Value labels
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.02 * max(ops_10s),
            f"{int(h):,} ops\n({h / 10.0:,.0f} ops/s)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "ops_10s_throughput.png"), dpi=300)
    plt.close(fig)


def plot_latency_operations(scaling_data: Dict[str, Any], output_dir: str):
    sizes = scaling_data["sizes"]
    op1_m = scaling_data["op1_mean_us"]
    op1_s = scaling_data["op1_std_us"]
    op2_m = scaling_data["op2_mean_us"]
    op2_s = scaling_data["op2_std_us"]
    op3_m = scaling_data["op3_mean_us"]
    op3_s = scaling_data["op3_std_us"]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(
        sizes,
        op1_m,
        yerr=op1_s,
        fmt="-o",
        label="Op 1: Top 100 students",
        color="#2980b9",
        capsize=4,
    )
    ax.errorbar(
        sizes,
        op2_m,
        yerr=op2_s,
        fmt="-s",
        label="Op 2: Set student rating",
        color="#e67e22",
        capsize=4,
    )
    ax.errorbar(
        sizes,
        op3_m,
        yerr=op3_s,
        fmt="-^",
        label="Op 3: Best average rating group",
        color="#27ae60",
        capsize=4,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Database Size N (log scale)", fontsize=12)
    ax.set_ylabel("Average Latency (microseconds µs)", fontsize=12)
    ax.set_title(
        "StudentDBOptimized: Latency per Operation vs Dataset Size",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "latency_operations.png"), dpi=300)
    plt.close(fig)


def plot_slowest_op_regression(
    sizes: List[int],
    means: List[float],
    stds: List[float],
    coeffs: np.ndarray,
    r2: float,
    formula: str,
    op_name: str,
    output_dir: str,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    # Experimental points with error bars
    ax.errorbar(
        sizes,
        means,
        yerr=stds,
        fmt="o",
        color="#c0392b",
        ecolor="#7f8c8d",
        elinewidth=2,
        capsize=5,
        label="Experimental Data (Mean ± Std)",
    )

    # Smooth curve for fitted model
    n_dense = np.linspace(min(sizes), max(sizes), 400)
    c1, c2, c3 = coeffs
    y_dense = c1 + c2 * np.log2(n_dense) + c3 * n_dense

    ax.plot(
        n_dense,
        y_dense,
        "-",
        color="#2980b9",
        linewidth=2.5,
        label=f"Fitted Model $T(n)$: $R^2 = {r2:.4f}$\n{formula}",
    )

    ax.set_xlabel("Database Size N", fontsize=12)
    ax.set_ylabel(f"Latency of {op_name} (µs)", fontsize=12)
    ax.set_title(
        f"Linear Regression Fit for Slowest Operation ({op_name})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "slowest_op_regression.png"), dpi=300)
    plt.close(fig)


def plot_memory_usage(sizes: List[int], mem_mb: List[float], output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, mem_mb, "-o", color="#8e44ad", linewidth=2, markersize=6)

    ax.set_xlabel("Database Size N", fontsize=12)
    ax.set_ylabel("Allocated Memory (MB, via tracemalloc)", fontsize=12)
    ax.set_title(
        "StudentDBOptimized: Memory Footprint vs Database Size",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "memory_usage.png"), dpi=300)
    plt.close(fig)


def plot_task2_sorting(task2_data: Dict[str, Any], output_dir: str):
    sizes = task2_data["sizes"]
    b_m = task2_data["builtin_mean_ms"]
    b_s = task2_data["builtin_std_ms"]
    r_m = task2_data["radix_mean_ms"]
    r_s = task2_data["radix_std_ms"]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.errorbar(
        sizes,
        b_m,
        yerr=b_s,
        fmt="-o",
        label="Standard Built-in Sort (Timsort O(N log N))",
        color="#2980b9",
        capsize=4,
    )
    ax.errorbar(
        sizes,
        r_m,
        yerr=r_s,
        fmt="-s",
        label="Custom Radix Sort (LSD O(d * N))",
        color="#e74c3c",
        capsize=4,
    )

    ax.set_xlabel("Database Size N", fontsize=12)
    ax.set_ylabel("Execution Time (milliseconds ms)", fontsize=12)
    ax.set_title(
        "Task 2: Phone Number Sorting Time Comparison (Variant S7)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "task2_sorting_comparison.png"), dpi=300)
    plt.close(fig)


def generate_markdown_report(
    benchmark_json_path: str, output_report_path: str, plots_dir_rel: str = "plots"
):
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    throughput = data["throughput_10s"]
    scaling = data["scaling_optimized"]
    task2 = data["task2_sorting"]
    sizes = scaling["sizes"]

    # Identify slowest operation
    op1_max = max(scaling["op1_mean_us"])
    op2_max = max(scaling["op2_mean_us"])
    op3_max = max(scaling["op3_mean_us"])

    if op1_max >= op2_max and op1_max >= op3_max:
        slowest_name = "Op 1 (get_top_100_students)"
        slowest_means = scaling["op1_mean_us"]
        slowest_stds = scaling["op1_std_us"]
    elif op2_max >= op1_max and op2_max >= op3_max:
        slowest_name = "Op 2 (set_student_rating)"
        slowest_means = scaling["op2_mean_us"]
        slowest_stds = scaling["op2_std_us"]
    else:
        slowest_name = "Op 3 (get_groups_with_highest_average_rating)"
        slowest_means = scaling["op3_mean_us"]
        slowest_stds = scaling["op3_std_us"]

    coeffs, r2, formula = perform_linear_regression(sizes, slowest_means)

    plots_abs_dir = os.path.join(os.path.dirname(output_report_path), plots_dir_rel)
    os.makedirs(plots_abs_dir, exist_ok=True)

    # Generate all plots
    plot_throughput_comparison(throughput, plots_abs_dir)
    plot_latency_operations(scaling, plots_abs_dir)
    plot_slowest_op_regression(
        sizes,
        slowest_means,
        slowest_stds,
        coeffs,
        r2,
        formula,
        slowest_name,
        plots_abs_dir,
    )
    plot_memory_usage(sizes, scaling["memory_mb"], plots_abs_dir)
    plot_task2_sorting(task2, plots_abs_dir)

    # Build report
    report = []
    report.append("# Звіт до Домашнього завдання №1: Сортування і структури даних\n")
    report.append("**Студент:** Олександр Оксентюк  ")
    report.append("**Група:** ПКН25/Б-1  ")
    report.append("**Варіанти:**  ")
    report.append(
        "- **Задача I:** Варіант **V5** (Співвідношення операцій $A:B:C = 10 : 10 : 1$)"
    )
    report.append("  1. Повернути 100 студентів з найвищим рейтингом.")
    report.append(
        "  2. Встановити рейтинг студенту за його електронною поштою (`m_email`)."
    )
    report.append(
        "  3. Знайти групу або групи з найбільшим середнім рейтингом студентів."
    )
    report.append(
        "- **Задача II:** Варіант **S7** (Посортувати всіх студентів за `m_phone_number` і зберегти в CSV-файл).\n"
    )
    report.append("---\n")

    report.append(
        "## 1. Задача I: Порівняння 3 підходів (Тест на пропускну здатність за 10 секунд)\n"
    )
    report.append("Було реалізовано 3 архітектурні підходи:")
    report.append(
        "1. **StudentDBNaive:** Лінійне сховище (список `list[Student]`). Операція 1 виконує повне сортування $O(N \\log N)$, операція 2 — лінійний пошук $O(N)$, операція 3 — повний перерахунок середніх значень $O(N)$."
    )
    report.append(
        "2. **StudentDBIndexed:** Хеш-таблиця `email -> Student` для пошуку за $O(1)$, хеш-таблиця сум і кількостей за групами $O(1)$ для інкрементального оновлення, та `heapq.nlargest` $O(N \\log K)$ для топ-100."
    )
    report.append(
        "3. **StudentDBOptimized:** Комбінована індексація — бакетна структура для рейтингу (O(1) переміщення, O(K log K) збір топ-100), хеш-таблиця для миттєвого пошуку студента та кешування групи-лідера з ледачою інвалідацією.\n"
    )

    report.append(
        "### Результати 10-секундного експерименту ($N = 100\\,000$, співвідношення 10:10:1):\n"
    )
    report.append(
        "| Підхід | Всього операцій за 10с | Пропускна здатність (оп/сек) | Op 1 виконано | Op 2 виконано | Op 3 виконано |"
    )
    report.append("|---|---|---|---|---|---|")
    for k, v in throughput.items():
        bd = v["breakdown"]
        report.append(
            f"| **{k}** | {v['total_ops']:,} | {v['ops_per_sec']:,.1f} | {bd['1']:,} | {bd['2']:,} | {bd['3']:,} |"
        )

    report.append(
        f"\n![10-Second Throughput]({plots_dir_rel}/ops_10s_throughput.png)\n"
    )
    report.append("---\n")

    report.append("## 2. Масштабування найкращого підходу (StudentDBOptimized)\n")
    report.append(
        "Для найкращого варіанту проведено експеримент на 10 різних розмірах початкової бази даних ($N \\in [100, 100\\,000]$), по 5 незалежних прогонів на кожну точку. Пам'ять виміряно за допомогою модуля `tracemalloc`.\n"
    )

    report.append("### Таблиця результатів масштабування:\n")
    report.append(
        "| Розмір $N$ | Пам'ять (МБ) | Op 1 Latency (мкс) | Op 2 Latency (мкс) | Op 3 Latency (мкс) |"
    )
    report.append("|---|---|---|---|---|")
    for i, sz in enumerate(sizes):
        m = scaling["memory_mb"][i]
        o1 = f"{scaling['op1_mean_us'][i]:.2f} ± {scaling['op1_std_us'][i]:.2f}"
        o2 = f"{scaling['op2_mean_us'][i]:.2f} ± {scaling['op2_std_us'][i]:.2f}"
        o3 = f"{scaling['op3_mean_us'][i]:.2f} ± {scaling['op3_std_us'][i]:.2f}"
        report.append(f"| {sz:,} | {m:.2f} | {o1} | {o2} | {o3} |")

    report.append(f"\n![Latency per Operation]({plots_dir_rel}/latency_operations.png)")
    report.append(f"\n![Memory Footprint]({plots_dir_rel}/memory_usage.png)\n")
    report.append("---\n")

    report.append("## 3. Лінійна регресія для найповільнішої операції\n")
    report.append(f"Найповільнішою операцією визначено: **{slowest_name}**.\n")
    report.append("Апроксимація теоретичної складності моделлю:")
    report.append(
        "$$T(n) = c_1 \\cdot f_1(n) + c_2 \\cdot f_2(n) + c_3 \\cdot f_3(n)$$"
    )
    report.append(
        "де $f_1(n) = 1$ (базовий константний оверхед виклику), $f_2(n) = \\log_2(n)$ (логарифмічний пошук), $f_3(n) = n$ (лінійна компонента).\n"
    )
    report.append(f"**Отримана емпірична функція:**\n")
    report.append(
        f"$$T(n) = {coeffs[0]:.4e} + {coeffs[1]:.4e} \\cdot \\log_2(n) + {coeffs[2]:.4e} \\cdot n$$\n"
    )
    report.append(f"- **Коефіцієнт детермінації ($R^2$):** `{r2:.4f}`\n")
    report.append(
        f"![Slowest Operation Regression]({plots_dir_rel}/slowest_op_regression.png)\n"
    )
    report.append("---\n")

    report.append("## 4. Задача II: Порівняння алгоритмів сортування (Варіант S7)\n")
    report.append("Сортування бази за полем `m_phone_number` (`38(0xx)xx-xx-xxx`):\n")
    report.append(
        "1. **Standard Built-in Sort (Timsort):** Реалізований на мові C в інтерпретаторі Python ($O(N \\log N)$)."
    )
    report.append(
        "2. **Custom LSD Radix Sort:** Власна реалізація порозрядного сортування за 9 змінними цифрами номеру телефону. Складність $O(d \\cdot N) = O(N)$, де $d = 9$.\n"
    )

    report.append("### Таблиця часу сортування (мілісекунди):\n")
    report.append(
        "| Розмір $N$ | Built-in Timsort (мс) | Custom Radix Sort (мс) | Співвідношення (Radix / Builtin) |"
    )
    report.append("|---|---|---|---|")
    for i, sz in enumerate(sizes):
        bm = task2["builtin_mean_ms"][i]
        bs = task2["builtin_std_ms"][i]
        rm = task2["radix_mean_ms"][i]
        rs = task2["radix_std_ms"][i]
        ratio_str = f"{rm / bm:.1f}x" if bm > 0 else "-"
        report.append(
            f"| {sz:,} | {bm:.3f} ± {bs:.3f} | {rm:.3f} ± {rs:.3f} | {ratio_str} |"
        )

    report.append(
        f"\n![Sorting Comparison]({plots_dir_rel}/task2_sorting_comparison.png)\n"
    )
    report.append("---\n")

    report.append("## 5. Відповіді на запитання до захисту\n")
    report.append("1. **Чому обраний варіант реалізації є раціональним?**  ")
    report.append(
        "   - Підхід `StudentDBOptimized` розділяє дані за призначенням: хеш-індекс для точкового доступу $O(1)$, бакетний розподіл для сортування за обмеженим числовим діапазоном рейтингу $[0, 100]$, та інкрементальне оновлення статистик груп. Це дозволяє досягти пропускної здатності в сотні тисяч операцій за 10 секунд."
    )
    report.append("2. **Чи зберігається коректність після операцій зміни даних?**  ")
    report.append(
        "   - Так, при виклику `set_student_rating` оновлюються: сам об'єкт `Student`, його бакет рейтингу (якщо рейтинг перетнув межу бакету), сума рейтингу групи у `group_totals`, а також інвалідується або оновлюється кеш найкращих груп `cached_best_groups`."
    )
    report.append(
        "3. **Як зміниться вибір структур даних, якщо система має бути стійка до DoS-атаки (ненадійний користувач)?**  "
    )
    report.append(
        "   - Стандартні хеш-таблиці вразливі до Hash Collision DoS (підбір однакових хеш-ключів, що перетворює $O(1)$ в $O(N)$). Для захисту необхідно:  "
    )
    report.append(
        "     - Використовувати криптографічно стійкі хеш-функції (наприклад, SipHash з випадковим secret seed);"
    )
    report.append(
        "     - Замінити відкриті хеш-таблиці на збалансовані дерева пошуку (Red-Black Tree або AVL-дерево), що гарантують найгірший час $O(\\log N)$ за будь-яких вхідних даних;"
    )
    report.append(
        "     - Обмежувати частоту запитів (Rate Limiting) та валідувати коректність довжини й формату полів."
    )

    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Report generated successfully at: {output_report_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    res_json = os.path.join(base_dir, "..", "benchmark_results.json")
    report_md = os.path.join(base_dir, "..", "REPORT.md")
    generate_markdown_report(res_json, report_md)
