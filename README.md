import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

sns.set_style("whitegrid")

DRIVE_PATH = '/content/drive/MyDrive/ML/'
#memory_optimized
def optimize_mem(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'-> Memory usage before: {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'-> Memory usage after: {end_mem:.2f} MB')
    return df

# data

try:
    df_orders = optimize_mem(pd.read_csv(DRIVE_PATH + 'orders.csv'))
    df_priors = optimize_mem(pd.read_csv(DRIVE_PATH + 'order_products__prior.csv.xlsx')) 
    df_train = optimize_mem(pd.read_csv(DRIVE_PATH + 'order_products__train.csv'))
    df_products = optimize_mem(pd.read_csv(DRIVE_PATH + 'products.csv'))
    print("Files loaded successfully!")
except:
    print("Error loading files! Check the path.")


#2 feature_engineering

df_priors = df_priors.merge(df_orders[['order_id', 'user_id', 'order_number']], on='order_id', how='left')

user_feats = df_priors.groupby('user_id').agg({
    'order_number': 'max',
    'product_id': 'count',
    'reordered': 'mean'
}).rename(columns={
    'order_number': 'u_total_orders',
    'product_id': 'u_total_items',
    'reordered': 'u_reorder_ratio'
})

print("- Processing product stats...")
prod_feats = df_priors.groupby('product_id').agg({
    'order_id': 'count',
    'reordered': 'mean'
}).rename(columns={
    'order_id': 'p_total_purchases',
    'reordered': 'p_reorder_prob'
})


inter_feats = df_priors.groupby(['user_id', 'product_id']).agg({
    'order_id': 'count',
}).rename(columns={'order_id': 'up_times_bought'})


train_orders = df_orders[df_orders['eval_set'] == 'train']


data = df_train.merge(train_orders[['order_id', 'user_id', 'days_since_prior_order']], on='order_id', how='left')


data = data.merge(user_feats, on='user_id', how='left')
data = data.merge(prod_feats, on='product_id', how='left')
data = data.merge(inter_feats, on=['user_id', 'product_id'], how='left')

fill_cols = ['u_total_orders', 'u_total_items', 'u_reorder_ratio', 'p_total_purchases', 'p_reorder_prob', 'up_times_bought']
my_imputer = SimpleImputer(strategy='mean')
data[fill_cols] = my_imputer.fit_transform(data[fill_cols])

data['days_since_prior_order'] = data['days_since_prior_order'].fillna(data['days_since_prior_order'].mean())

print(f"data shape: {data.shape}")

del df_orders, df_priors, df_train, df_products, user_feats, prod_feats, inter_feats
gc.collect()

sample_size = 5000
print(f"Taking a sample of {sample_size} rows.")
df_sample = data.sample(n=sample_size, random_state=42)

features_cols = ['u_total_orders', 'u_total_items', 'u_reorder_ratio', 
                 'p_total_purchases', 'p_reorder_prob', 'up_times_bought', 'days_since_prior_order']

X = df_sample[features_cols]
y_class = df_sample['reordered']
y_reg = df_sample['days_since_prior_order']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X_scaled, y_class, y_reg, test_size=0.2, random_state=42
)


#Classification


#SVM
print("SVM")
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train, y_cls_train)
svm_preds = svm_model.predict(X_test)
print(f"SVM Acc: {accuracy_score(y_cls_test, svm_preds):.4f}")
print(f"SVM:  {f1_score(y_cls_test, svm_preds):.4f}")


#KNN
print("KNN")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_cls_train)
knn_preds = knn_model.predict(X_test)
print(f"KNN Acc: {accuracy_score(y_cls_test, knn_preds):.4f}")
print(f"KNN:  {f1_score(y_cls_test, knn_preds):.4f}")


#regression
print("\n Regression Results")

#SVR
print("SVR")
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_reg_train)
svr_preds = svr_model.predict(X_test)
print(f"SVR RMSE: {np.sqrt(mean_squared_error(y_reg_test, svr_preds)):.4f}")
print(f"SVR Regressor:   {r2_score(y_reg_test, svr_preds):.4f}")

#KNN_Regressor
print("KNN")
knn_reg_model = KNeighborsRegressor(n_neighbors=5)
knn_reg_model.fit(X_train, y_reg_train)
knn_reg_preds = knn_reg_model.predict(X_test)
print(f"KNN RMSE: {np.sqrt(mean_squared_error(y_reg_test, knn_reg_preds)):.4f}")
print(f"KNN Regressor:   {r2_score(y_reg_test, knn_reg_preds):.4f}")















