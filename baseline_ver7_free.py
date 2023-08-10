import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import re
from scipy.stats import mode, skew, kurtosis

def extract_duration_in_minutes(duration_str):
    if pd.isnull(duration_str):
        return None
    else:
        # Regex pattern to find numbers and hr or min
        pattern = r'([0-9]+)\s*(hr|min)\.'
        matches = re.findall(pattern, duration_str)

        # Initialize total minutes
        total_minutes = 0

        for match in matches:
            # If hr, convert to minutes
            if match[1] == 'hr':
                total_minutes += int(match[0]) * 60
            # If min, just add to total
            elif match[1] == 'min':
                total_minutes += int(match[0])
        
        return total_minutes

# Function to convert episodes into integers
def convert_episodes_to_int(episodes_str):
    if pd.isnull(episodes_str):
        return None
    else:
        # If episodes is "Unknown", return None
        if episodes_str == "Unknown":
            return None
        else:
            # Else, convert to integer
            return int(episodes_str)

# Load the data
train_df = pd.read_csv('train.csv')
anime_df = pd.read_csv('anime.csv')
series_df = pd.read_csv("series.csv")

test_df = pd.read_csv('test.csv')

# Convert the scores to discrete values for stratification
train_df['score_discrete'] = train_df['score']

# Initialize the StratifiedKFold object
skf = StratifiedKFold(n_splits=5)

# Create a column for the fold number
train_df['fold'] = -1

# Assign the fold number to each row
for fold, (train_index, val_index) in enumerate(skf.split(train_df, train_df['score_discrete'])):
    train_df.loc[val_index, 'fold'] = fold
# Drop the temporary 'score_discrete' column
train_df = train_df.drop(columns=['score_discrete'])

#ここより上は修正しない方が良いよ！------------------------------------------------------------------------------------------------------------------------------------

# Apply the function to the duration column　いろんな特徴量を追加しているよ！(実際に使うかは別として)
anime_df['duration'] = anime_df['duration'].apply(extract_duration_in_minutes)
# Apply the function to the episodes column　いろんな特徴量を追加しているよ！(実際に使うかは別として)
anime_df['episodes'] = anime_df['episodes'].apply(convert_episodes_to_int)
# Create new feature by multiplying duration_in_minutes with episodes　いろんな特徴量を追加しているよ！(実際に使うかは別として)
anime_df['total_duration'] = anime_df['duration'] * anime_df['episodes']

# Create a new feature based on the quarter of the year　いろんな特徴量を追加しているよ！(実際に使うかは別として)
anime_df[["aired_start", "aired_end"]] = anime_df["aired"].str.split(" to ", expand=True)
# Convert these new columns to datetime format　いろんな特徴量を追加しているよ！(実際に使うかは別として)
anime_df["aired_start"] = pd.to_datetime(anime_df["aired_start"], errors='coerce')
anime_df["aired_start_month"] = anime_df["aired_start"].dt.month
anime_df["aired_start_quarter"] = pd.cut(anime_df["aired_start_month"], bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4'])
# Create new features based on the ratio of watching to completed and watching to dropped　いろんな特徴量を追加しているよ！(実際に使うかは別として)
anime_df["watching_completed_ratio"] = anime_df["watching"] / anime_df["completed"]
anime_df["watching_dropped_ratio"] = anime_df["watching"] / anime_df["dropped"]
# Handle potential division by zero errors　いろんな特徴量を追加しているよ！(実際に使うかは別として)
anime_df["watching_completed_ratio"].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
anime_df["watching_dropped_ratio"].replace([np.inf, -np.inf, np.nan], 0, inplace=True)

# Extract start and end dates from 'aired' column　いろんな特徴量を追加しているよ！(実際に使うかは別として)
anime_df['start_date'], anime_df['end_date'] = anime_df['aired'].str.split(' to ', 1).str
# Convert the start and end dates to datetime format　いろんな特徴量を追加しているよ！(実際に使うかは別として)
anime_df['start_date'] = pd.to_datetime(anime_df['start_date'], errors='coerce')
anime_df['end_date'] = pd.to_datetime(anime_df['end_date'], errors='coerce')
# Calculate the airing duration and store it in a new column 'airing_duration'　いろんな特徴量を追加しているよ！(実際に使うかは別として)
anime_df['airing_duration'] = (anime_df['end_date'] - anime_df['start_date']).dt.days

#アニメの特徴量を加えたい場合はこの下に書いてね！




#使わない特徴量はここに書いてね！．
anime_df = anime_df.drop(['aired_start', 'aired_end', 'japanese_name', 'aired', 'start_date', 'end_date', 'producers', 'licensors', 'studios', ], axis=1)

'''
#評価の件数を追加 7/21
# Count the number of ratings per anime
anime_rating_counts = train_df.groupby('anime_id').size().reset_index(name='anime_rating_counts')
# Count the number of ratings per user
user_rating_counts = train_df.groupby('user_id').size().reset_index(name='user_rating_counts')
# Merge the count data with the original data
train_df = pd.merge(train_df, anime_rating_counts, how='left', on='anime_id')
train_df = pd.merge(train_df, user_rating_counts, how='left', on='user_id')
test_df = pd.merge(test_df, anime_rating_counts, how='left', on='anime_id')
test_df = pd.merge(test_df, user_rating_counts, how='left', on='user_id')
'''

#ユーザの評価の情報とtrain, testをマージしているよ！ここは修正しなくてOK!
train_merged = train_df.merge(anime_df, on='anime_id', how='left')
test_merged = test_df.merge(anime_df, on='anime_id', how='left')

#ユーザごとの評価平均や分散，最大最小，変動係数を追加しているよ！
user_stats = train_merged.groupby('user_id').filter(lambda x: len(x) >= 30).groupby('user_id')['score'].agg(['mean', 'var', 'max', 'min'])
user_stats['cv'] = user_stats['var'] ** 0.5 / user_stats['mean']  # coefficient of variation = std/mean

#アニメごとの評価平均や分散，最大最小，変動係数を追加しているよ！
anime_stats = train_merged.groupby('anime_id').filter(lambda x: len(x) >= 30).groupby('anime_id')['score'].agg(['mean', 'var', 'max', 'min'])
anime_stats['cv'] = anime_stats['var'] ** 0.5 / anime_stats['mean']  # coefficient of variation = std/mean

# Merge the train data with the anime meta data　ここは修正しなくてOK!
train_merged = train_merged.merge(anime_stats, on='anime_id', how='left', suffixes=("", "_anime"))
train_merged = train_merged.merge(user_stats, on='user_id', how='left', suffixes=("", "_user"))
# Prepare the test data ここは修正しなくてOK!
test_merged = test_merged.merge(anime_stats, on='anime_id', how='left', suffixes=("", "_anime"))
test_merged = test_merged.merge(user_stats, on='user_id', how='left', suffixes=("", "_user"))

# Filter users who have rated 30 or more animes
user_rating_counts = train_df['user_id'].value_counts()
users_with_30_or_more_ratings = user_rating_counts[user_rating_counts >= 30].index
filtered_train_df = train_df[train_df['user_id'].isin(users_with_30_or_more_ratings)]
# Compute the median, mode, range, skewness and kurtosis for each user
user_stats_df = filtered_train_df.groupby('user_id')['score'].agg(['median', lambda x: mode(x)[0][0], 'max', 'min', skew, kurtosis])
# Calculate the range (max - min)
user_stats_df['range'] = user_stats_df['max'] - user_stats_df['min']
# Drop the original 'max' and 'min' columns
user_stats_df = user_stats_df.drop(['max', 'min'], axis=1)
# Rename the columns
user_stats_df.columns = ['median', 'mode', 'skewness', 'kurtosis', 'range']
user_stats_df.reset_index(inplace=True)
# Merge the user statistics with the train and test data
train_merged = train_merged.merge(user_stats_df, how='left', on='user_id')
test_merged = test_merged.merge(user_stats_df, how='left', on='user_id')






# Encode the categorical variables  カテゴリデータを追加したらここに書いてね！．
cat_cols = ['user_id', 'anime_id', 'genres', 'type', 'source', 'rating']

#ここより下は修正しない方が良いよ！------------------------------------------------------------------------------------------------------------------------------------
les = []
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train_merged[col], test_merged[col]]).fillna(''))
    train_merged[col] = le.transform(train_merged[col].fillna(''))
    test_merged[col] = le.transform(test_merged[col].fillna(''))
    les.append(le)

# Training and evaluation with LightGBM
scores_lgb = []
models_lgb = []

for fold in range(5):
    print(f"Training for fold {fold}...")

    # Prepare the train and validation data
    train_data = train_merged[train_merged['fold'] != fold]
    val_data = train_merged[train_merged['fold'] == fold]
    train_data_x = train_data.drop(['score', 'fold'], axis=1)
    val_data_x = val_data.drop(['score', 'fold'], axis=1)

    # Define the target
    target = 'score'

    # Prepare the LightGBM datasets
    lgb_train = lgb.Dataset(train_data_x, train_data[target])
    lgb_val = lgb.Dataset(val_data_x, val_data[target])

    # Define the parameters
    params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.1,
            # 'reg_lambda': 1.0
    }

    # Train the model
    callbacks = [
        lgb.early_stopping(stopping_rounds=20), 
        lgb.log_evaluation(period=20)
    ]
    model_lgb = lgb.train(params, lgb_train, valid_sets=[lgb_val], callbacks=callbacks)

    # Save the model
    with open(f'model_lgb_{fold}.pkl', 'wb') as f:
        pickle.dump(model_lgb, f)

    # Predict the validation data
    val_pred_lgb = model_lgb.predict(val_data_x, num_iteration=model_lgb.best_iteration)

    # Evaluate the model
    score_lgb = np.sqrt(mean_squared_error(val_data[target], val_pred_lgb))
    scores_lgb.append(score_lgb)

    print(f"RMSE for fold {fold}: {score_lgb}")

# Calculate the average score
average_score_lgb = np.mean(scores_lgb)

print(f"Average RMSE: {average_score_lgb}")   #これが最小になることを目指してみて！

# Predict the test data and create the submission file
submission_df = pd.read_csv('sample_submission.csv')
submission_df['score'] = 0

for fold in range(5):
    with open(f'model_lgb_{fold}.pkl', 'rb') as f:
        model_lgb = pickle.load(f)
    test_pred_lgb = model_lgb.predict(test_merged, num_iteration=model_lgb.best_iteration)
    submission_df['score'] += test_pred_lgb / 5

submission_df.to_csv('submission_baseline_ver7_30.csv', index=False)

WITH Rank1 AS (
    SELECT id, 
           RANK() OVER (ORDER BY score1 DESC) as rank1
    FROM your_table
),

Rank2 AS (
    SELECT id, 
           RANK() OVER (ORDER BY score2 DESC) as rank2
    FROM your_table
),

CombinedRanks AS (
    SELECT r1.id, 
           r1.rank1, 
           r2.rank2, 
           (r1.rank1 + r2.rank2) as total_rank
    FROM Rank1 r1
    JOIN Rank2 r2 ON r1.id = r2.id
)

SELECT *,
       RANK() OVER (ORDER BY total_rank ASC) as final_rank
FROM CombinedRanks
ORDER BY final_rank;
