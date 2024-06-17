import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


combined_df = pd.read_csv('house_price.csv')

cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
combined_df.drop(columns=cols_to_drop, inplace=True)

num_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    combined_df[col].fillna(combined_df[col].median(), inplace=True)

cat_cols = combined_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    combined_df[col].fillna('None', inplace=True)


train_df = combined_df[combined_df['source'] == 'train'].drop(['source'], axis=1)
test_df = combined_df[combined_df['source'] == 'test'].drop(['source', 'SalePrice'], axis=1)

numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

train_df[categorical_features] = train_df[categorical_features].fillna('None')
train_df[numeric_features] = train_df[numeric_features].fillna(0)
test_df[categorical_features] = test_df[categorical_features].fillna('None')
test_df[numeric_features] = test_df[numeric_features].fillna(0)


plt.figure(figsize=(10, 6))
sns.histplot(train_df['SalePrice'], bins=30, kde=True)
plt.title('Distribuição de SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequência')
plt.show()

numeric_train_df = train_df[numeric_features + ['SalePrice']]
correlation_matrix = numeric_train_df.corr()


plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação')
plt.show()


correlation_with_saleprice = correlation_matrix['SalePrice'].sort_values(ascending=False)
print(f"A correlação de preço para cada atributo é: {correlation_with_saleprice}")


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


X_train = train_df.drop('SalePrice', axis=1)
y_train = train_df['SalePrice']


X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(test_df)


model = LinearRegression()
model.fit(X_train_processed, y_train)


train_predictions = model.predict(X_train_processed)


mse = mean_squared_error(y_train, train_predictions)
rmse = mse ** 0.5
r2 = r2_score(y_train, train_predictions)
print(f'RMSE do modelo: {rmse}')
print(f'R^2 do modelo: {r2}')


plt.figure(figsize=(10, 6))
plt.scatter(y_train, train_predictions)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')
plt.title('Previsões vs Valores Reais')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.show()


test_predictions = model.predict(X_test_processed)


submission_df = test_df[['Id']].copy()
submission_df['SalePrice'] = test_predictions
submission_df.to_csv('house_price_predictions.csv', index=False)
