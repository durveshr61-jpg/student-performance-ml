import pandas as pd
import os
import json
import glob
from sklearn.metrics import accuracy_score, f1_score

# 1. Get GitHub username
group_name = os.environ.get("GROUP_NAME", "Unknown Student")

# 2. Load answer key
truth_file = "grader/test_labels.csv"
if not os.path.exists(truth_file):
    print("❌ Error: Secret labels not found. Contact Instructor.")
    exit(1)

truth_df = pd.read_csv(truth_file)
# If no 'id' column, create one from row index
if 'id' not in truth_df.columns:
    truth_df['id'] = range(1, len(truth_df) + 1)
truth_df['id'] = truth_df['id'].astype(int)

print(f"✅ Loaded answer key: {len(truth_df)} rows, columns: {truth_df.columns.tolist()}")

# 3. Find student submission
sub_files = glob.glob("submission/*.csv")
# If no 'id' column, create one from row index
if 'id' not in sub_df.columns:
    sub_df['id'] = range(1, len(sub_df) + 1)
sub_df['id'] = sub_df['id'].astype(int)

print(f"✅ Loaded submission: {len(sub_df)} rows, columns: {sub_df.columns.tolist()}")

# 4. Grade
try:
    truth_df = truth_df.sort_values("id").reset_index(drop=True)
    sub_df   = sub_df.sort_values("id").reset_index(drop=True)

    if len(truth_df) != len(sub_df):
        print(f"❌ Error: Row count mismatch. Expected {len(truth_df)}, got {len(sub_df)}")
        exit(1)

    acc = accuracy_score(truth_df['result'], sub_df['result'])
    f1  = f1_score(truth_df['result'], sub_df['result'], zero_division=0)

    print(f"✅ Grading Successful!")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    os.makedirs("leaderboard_data", exist_ok=True)
    result_data = {
        "group":    group_name,
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4)
    }
    with open("leaderboard_data/result.json", "w") as f:
        json.dump(result_data, f)

except Exception as e:
    print(f"❌ Error during grading: {e}")
    exit(1)
