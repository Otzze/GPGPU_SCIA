
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
try:
    benchmark_df = pd.read_csv('benchmark_results.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the CSV files are in the same directory as the script.")
    exit()

# --- Data Cleaning and Preparation ---

# Fill missing avg_fps where possible
benchmark_df['avg_fps'] = pd.to_numeric(benchmark_df['avg_fps'], errors='coerce')
has_duration = benchmark_df['Duration_ms'].notna() & benchmark_df['nb_frames'].notna()
is_fps_missing = benchmark_df['avg_fps'].isna()
benchmark_df.loc[is_fps_missing & has_duration, 'avg_fps'] = \
    benchmark_df['nb_frames'] / (benchmark_df['Duration_ms'] / 1000)

# Drop rows that still don't have avg_fps or Duration_ms as they are not useful for plotting
benchmark_df.dropna(subset=['avg_fps', 'Duration_ms'], how='all', inplace=True)

# Aggregate results for configurations with multiple runs
agg_functions = {
    'avg_fps': 'mean',
    'Duration_ms': 'mean',
    'nb_frames': 'first' # Assuming nb_frames is constant for same Tag/Mode
}
benchmark_agg_df = benchmark_df.groupby(['Tag', 'Mode']).agg(agg_functions).reset_index()

# Combine Tag and Mode for unique labels
benchmark_agg_df['Tag_Mode'] = benchmark_agg_df['Tag'] + '_' + benchmark_agg_df['Mode']


# --- Plot 1: Performance (AVG FPS) of all configurations ---
fps_sorted = benchmark_agg_df.sort_values(by='avg_fps', ascending=True)
plt.figure(figsize=(14, 10))
plt.barh(fps_sorted['Tag_Mode'], fps_sorted['avg_fps'], color='skyblue')
plt.xlabel('Average FPS (Higher is Better)')
plt.ylabel('Configuration')
plt.title('Performance Comparison: Average FPS')
plt.tight_layout()
plt.savefig('all_configs_fps.png')
plt.close()
print("Generated all_configs_fps.png")


# --- Plot 2: Performance (Duration) of all configurations ---
duration_sorted = benchmark_agg_df.dropna(subset=['Duration_ms']).sort_values(by='Duration_ms', ascending=False)
plt.figure(figsize=(14, 10))
plt.barh(duration_sorted['Tag_Mode'], duration_sorted['Duration_ms'], color='salmon')
plt.xlabel('Execution Duration in ms (Lower is Better)')
plt.ylabel('Configuration')
plt.title('Performance Comparison: Execution Duration')
plt.tight_layout()
plt.savefig('all_configs_duration.png')
plt.close()
print("Generated all_configs_duration.png")

# --- Plot 3: GPU Optimization Progression (FPS) ---
gpu_df = benchmark_agg_df[benchmark_agg_df['Mode'] == 'gpu'].copy()
# Define a logical order for optimizations
optimization_order = [
    'cuda_first',
    'opti_hyst_memcpy',
    'opti_hyst_memcpy_v1',
    'opti_alert_memcpy',
    'opti_background',
    'opti_shared_morph',
    'opti_states_philox',
    'opti_states_no_philox',
    'opti_weights_uint8',
    'opti_shared_background',
    'opti_shared_hyst',
    'opti_unroll_hyst',
    'unroll_morph',
    'final',
    'final_128',
    'final_128_k_15'
]
# Some tags in the data might not be in our defined order, and vice-versa
# We will plot the ones that are in the data and in our order.
gpu_df['Tag'] = pd.Categorical(gpu_df['Tag'], categories=optimization_order, ordered=True)
gpu_opti_df = gpu_df.dropna(subset=['Tag']).sort_values('Tag')

plt.figure(figsize=(15, 8))
plt.plot(gpu_opti_df['Tag'], gpu_opti_df['avg_fps'], marker='o', linestyle='-', color='g')
plt.xlabel('Optimization Step')
plt.ylabel('Average FPS')
plt.title('GPU Performance Improvement Through Optimizations')
plt.xticks(rotation=45, ha='right')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('gpu_optimization_progression_fps.png')
plt.close()
print("Generated gpu_optimization_progression_fps.png")

# --- Plot 4: CPU vs GPU Comparison ---
cpu_tags = ['base_bep', 'cpu_opti_weights_uint8']
gpu_tags = ['cuda_first', 'opti_weights_uint8'] # Corresponding GPU versions

cpu_comp_df = benchmark_agg_df[benchmark_agg_df['Tag'].isin(cpu_tags) & (benchmark_agg_df['Mode'] == 'cpu')]
gpu_comp_df = benchmark_agg_df[benchmark_agg_df['Tag'].isin(gpu_tags) & (benchmark_agg_df['Mode'] == 'gpu')]

# For plotting, we need to align them. Let's create pairs.
# Pair 1: 'base_bep' (cpu) vs 'cuda_first' (gpu)
# Pair 2: 'cpu_opti_weights_uint8' (cpu) vs 'opti_weights_uint8' (gpu)
comparison_data = []
try:
    p1_cpu = cpu_comp_df[cpu_comp_df['Tag'] == 'base_bep']['avg_fps'].values[0]
    p1_gpu = gpu_comp_df[gpu_comp_df['Tag'] == 'cuda_first']['avg_fps'].values[0]
    comparison_data.append({'group': 'Initial Version', 'cpu': p1_cpu, 'gpu': p1_gpu})
except IndexError:
    print("Warning: Could not find data for 'base_bep' or 'cuda_first' for CPU vs GPU plot.")

try:
    p2_cpu = cpu_comp_df[cpu_comp_df['Tag'] == 'cpu_opti_weights_uint8']['avg_fps'].values[0]
    p2_gpu = gpu_comp_df[gpu_comp_df['Tag'] == 'opti_weights_uint8']['avg_fps'].values[0]
    comparison_data.append({'group': 'weights_uint8', 'cpu': p2_cpu, 'gpu': p2_gpu})
except IndexError:
    print("Warning: Could not find data for 'cpu_opti_weights_uint8' or 'opti_weights_uint8' for CPU vs GPU plot.")


if comparison_data:
    comp_df = pd.DataFrame(comparison_data)

    x = np.arange(len(comp_df['group']))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, comp_df['cpu'], width, label='CPU')
    rects2 = ax.bar(x + width/2, comp_df['gpu'], width, label='GPU')

    ax.set_ylabel('Average FPS')
    ax.set_title('CPU vs GPU Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df['group'])
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.savefig('cpu_vs_gpu_comparison.png')
    plt.close()
    print("Generated cpu_vs_gpu_comparison.png")
else:
    print("Skipping CPU vs GPU comparison plot as no data was found.")

# The CUDA operation time distribution plot from the original script is still useful.
try:
    cuda_df = pd.read_csv('cuda_first_cuda_api_gpu_sum.csv')

    # --- Plot 5: CUDA Operation Time Distribution ---
    # Get top 10 operations
    cuda_df_sorted = cuda_df.sort_values(by='Time (%)', ascending=False)
    top_10 = cuda_df_sorted.head(10)

    # Check if there are more than 10 operations
    if len(cuda_df_sorted) > 10:
        others_sum = cuda_df_sorted.iloc[10:]['Time (%)'].sum()
        # Create a new dataframe for plotting
        if others_sum > 0:
            # The column name could be 'Operation' or 'Name'
            op_col_name = 'Operation' if 'Operation' in cuda_df.columns else 'Name'
            others_row = pd.DataFrame([ {op_col_name: 'Others', 'Time (%)': others_sum}])
            plot_df = pd.concat([top_10, others_row], ignore_index=True)
        else:
            plot_df = top_10
    else:
        plot_df = cuda_df_sorted

    # Check the column name for operations
    if 'Operation' in plot_df.columns:
        operation_col = 'Operation'
    elif 'Name' in plot_df.columns:
        operation_col = 'Name'
    else:
        raise KeyError("Could not find operation/name column in cuda_first_cuda_api_gpu_sum.csv")


    plt.figure(figsize=(12, 12))
    wedges, texts, autotexts = plt.pie(
        plot_df['Time (%)'],
        labels=plot_df[operation_col],
        autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
        startangle=140,
        pctdistance=0.85
    )
    plt.title('CUDA Operation Time Distribution (Top 10)', pad=20)
    plt.axis('equal')

    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(8)

    plt.tight_layout()
    plt.savefig('cuda_operation_distribution.png')
    plt.close()

    print("Generated cuda_operation_distribution.png")

except FileNotFoundError:
    print("cuda_first_cuda_api_gpu_sum.csv not found, skipping CUDA operation distribution plot.")
except KeyError as e:
    print(f"Error processing CUDA data: {e}, skipping CUDA operation distribution plot.")
