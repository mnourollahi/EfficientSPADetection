import pandas as pd
import sys

if __name__ == "__main__":
    # Paths to baseline and antipattern folders
    blob1 = sys.argv[1]
    blob2 = sys.argv[2]
    # est1 = sys.argv[3]
    # est2 = sys.argv[4]
    # est3 = sys.argv[5]

    blob1_df = pd.read_csv(blob1)
    blob2_df = pd.read_csv(blob2)
    # est1_df = pd.read_csv(est1)
    # est2_df = pd.read_csv(est2)
    # est3_df = pd.read_csv(est3)


    # Merge the DataFrames (you can choose different merge strategies depending on your needs)
    merged_df_blob = pd.concat([blob1_df, blob2_df], ignore_index=True)
    # # Filter rows where the 'load' column is 5 or more
    filtered_df_blob = merged_df_blob[merged_df_blob['load'] >= 5]
    filtered_df_blob.to_csv("blob_not_agg.txt", index=False)
    # merged_df_est = pd.concat([est1_df, est2_df, est3_df], ignore_index=True)
    # merged_df_est = pd.concat([est2_df, est3_df], ignore_index=True)
    # filtered_df_est = merged_df_est[merged_df_est['load'] >= 5]
    # filtered_df_est.to_csv("est_not_agg.txt", index=False)
