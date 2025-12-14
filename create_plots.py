
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
try:
    benchmark_df = pd.read_csv('benchmark_results.csv')
    cuda_df = pd.read_csv('cuda_first_cuda_api_gpu_sum.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the CSV files are in the same directory as the script.")
    exit()

# Clean up benchmark data
benchmark_df['Tag_Mode'] = benchmark_df['Tag'] + '_' + benchmark_df['Mode']
# Convert avg_fps to numeric, coercing errors to NaN, then fill NaN with 0
benchmark_df['avg_fps'] = pd.to_numeric(benchmark_df['avg_fps'], errors='coerce').fillna(0)


# --- Plot 1: Execution Duration Comparison ---
plt.figure(figsize=(12, 7))
duration_plot = benchmark_df.loc[benchmark_df['Duration_ms'].notna()]
plt.bar(duration_plot['Tag_Mode'], duration_plot['Duration_ms'], color=['skyblue', 'lightgreen', 'salmon', 'gold', 'purple', 'orange', 'cyan'])
plt.xlabel('Configuration (Tag_Mode)')
plt.ylabel('Duration (ms)')
plt.title('Execution Duration Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('duration_comparison.png')
plt.close()

print("Generated duration_comparison.png")

# --- Plot 2: Average FPS Comparison ---
plt.figure(figsize=(10, 6))
fps_plot_df = benchmark_df[benchmark_df['avg_fps'] > 0]
plt.bar(fps_plot_df['Tag_Mode'], fps_plot_df['avg_fps'], color=['lightcoral', 'lightseagreen', 'mediumorchid'])
plt.xlabel('Configuration (Tag_Mode)')
plt.ylabel('Average FPS')
plt.title('Average FPS Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fps_comparison.png')
plt.close()

print("Generated fps_comparison.png")

# --- Plot 3: CUDA Operation Time Distribution ---
# Get top 10 operations
cuda_df_sorted = cuda_df.sort_values(by='Time (%)', ascending=False)
top_10 = cuda_df_sorted.head(10)
others_sum = cuda_df_sorted.iloc[10:]['Time (%)'].sum()

# Create a new dataframe for plotting
if others_sum > 0:
    others_row = pd.DataFrame([{'Operation': 'Others', 'Time (%)': others_sum}])
    plot_df = pd.concat([top_10, others_row], ignore_index=True)
else:
    plot_df = top_10

# Plotting the pie chart
plt.figure(figsize=(12, 12))
# Using a lambda function to prevent printing the percentage on small slices
# It will only print the percentage if it's larger than 2%
wedges, texts, autotexts = plt.pie(
    plot_df['Time (%)'], 
    labels=plot_df['Operation'], 
    autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
    startangle=140, 
    pctdistance=0.85
)
plt.title('CUDA Operation Time Distribution (Top 10)', pad=20)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Improve label readability
for text in texts:
    text.set_fontsize(9)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(8)

plt.tight_layout()
plt.savefig('cuda_operation_distribution.png')
plt.close()

print("Generated cuda_operation_distribution.png")
