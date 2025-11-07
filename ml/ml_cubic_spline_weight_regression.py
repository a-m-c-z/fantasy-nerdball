import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

for pos in ['GK', 'DEF', 'MID', 'FWD']:
    input_csv = f"ml_output/raw_output/weights_{pos}.csv"
    output_dir = "ml_output/cubic_spline_regressions"
    output_csv = os.path.join(output_dir, f"smoothed_weights_{pos}.csv")
    plot_file = os.path.join(
        output_dir, f"smoothed_weights_stacked_bar_{pos}.png")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    x = df['gameweek'].values

    smoothed_data = {'gameweek': df['gameweek']}

    smoothing_factors = {
        'form_dominance': 40,
        'historical_weight': 40,
        'fixture_sensitivity': 40  
    }

    for col in ['form_dominance', 'historical_weight', 'fixture_sensitivity']:
        y = df[col].values
        spline = UnivariateSpline(x, y, s=smoothing_factors[col], k=3)
        y_smooth = spline(x)
        smoothed_data[col + "_smoothed"] = y_smooth

    smoothed_df = pd.DataFrame(smoothed_data)

    smoothed_df.to_csv(output_csv, index=False)
    print(f"Smoothed spline CSV saved to: {output_csv}")

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(smoothed_df))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, col in enumerate(['form_dominance_smoothed',
                             'historical_weight_smoothed',
                             'fixture_sensitivity_smoothed']
                             ):
        plt.bar(
            smoothed_df['gameweek'],
            smoothed_df[col],
            bottom=bottom,
            color=colors[i],
            label=col.replace("_smoothed", "")
            )
        bottom += smoothed_df[col].values

    plt.xlabel("Gameweek")
    plt.ylabel("Smoothed Weight")
    plt.title("Stacked Bar Chart of Smoothed Weights by Gameweek "
              "(Cubic Spline)")
    plt.legend()
    plt.xticks(smoothed_df['gameweek'])
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.show()
    print(f"Stacked bar chart saved to: {plot_file}")
