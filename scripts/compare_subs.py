import pandas as pd

# Load submissions
v4 = pd.read_csv('submissions/submission_godmode_v4.csv')
best18 = pd.read_csv('submissions/submission_18.csv')

# Find differences
merged = v4.merge(best18, on='id', suffixes=('_v4', '_18'))
merged['diff'] = merged['Tm_v4'] - merged['Tm_18']
merged['abs_diff'] = merged['diff'].abs()

# Stats
print('V4 vs submission_18 (your previous best):')
print(f"Total: {len(merged)}")
print(f"Identical (diff<0.01): {(merged['abs_diff'] < 0.01).sum()}")
print(f"Different: {(merged['abs_diff'] >= 0.01).sum()}")
print(f"Mean abs diff: {merged['abs_diff'].mean():.4f}K")
print()

# Show all different predictions
diff_preds = merged[merged['abs_diff'] >= 0.01].sort_values('abs_diff', ascending=False)
print('All different predictions:')
print(diff_preds[['id', 'Tm_18', 'Tm_v4', 'diff']].to_string())
