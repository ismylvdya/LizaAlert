# model_training.py

#%%
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. Подготовка данных
# ==============================
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Используем уже обработанный пол
    df['gender'] = df.apply(lambda x: x['gender_new'] if x['gender'] in ['male', 'female'] else x['gender'], axis=1)

    # Удаляем ненужные колонки
    drop_cols = ['Unnamed: 0', 'id_hash', 'gender_new', 'head', 'content', 'date_of_find']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Преобразуем строки с возрастами в списки
    def parse_age(x):
        if isinstance(x, str) and x.startswith('['):
            return ast.literal_eval(x)
        return x
    df['age'] = df['age'].apply(parse_age)
    df = df.explode('age')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Разделяем даты на компоненты
    def split_date(r_date):
        if pd.notna(r_date):
            try:
                year, month, day_hour = r_date.split('-', 2)
                day, hour = day_hour.split()
                return int(year), int(month), int(day), int(hour.split(':')[0])
            except:
                return np.nan, np.nan, np.nan, np.nan
        return np.nan, np.nan, np.nan, np.nan

    date_cols = [c for c in df.columns if 'date' in c.lower()]
    for col in date_cols:
        df[[f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_hour']] = (
            df[col].apply(split_date).apply(pd.Series)
        )

    # search_period как число
    if 'search_period' in df.columns:
        df['search_period'] = df['search_period'].apply(lambda x: int(str(x).split()[0]) if pd.notna(x) else x)

    # Удаляем исходные колонки даты и search_period
    for c in ['date_search', 'date_of_loss', 'last_search_date', 'search_period']:
        if c in df.columns:
            df = df.drop(columns=c)

    # Заполнение пропусков
    df['gender'] = df['gender'].fillna('неопр')
    df['location'] = df['location'].fillna('неопр')
    mean_age = df['age'].mean()
    df['age'] = df['age'].fillna(mean_age)

    # Отбор только жив/мертв
    df = df[df['status'].isin(['жив(а)', 'погиб(ла)'])]

    # Заполнение пропусков в колонках с датой медианой
    date_cols_new = [c for c in df.columns if 'year' in c or 'month' in c or 'day' in c or 'hour' in c]
    for col in date_cols_new:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    return df

# ==============================
# 2. Кодирование категориальных данных
# ==============================
def encode_features(df, categorical_cols=['gender', 'location', 'status']):
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# ==============================
# 3. Балансировка классов и подготовка выборок
# ==============================
def prepare_train_test(df, target_col='status', sample_size=2000, test_size=0.3):
    # Берем по sample_size экземпляров для каждого класса
    sampled_df = df.groupby(target_col).apply(lambda x: x.sample(sample_size, random_state=42)).reset_index(drop=True)
    X = sampled_df.drop(columns=[target_col])
    y = sampled_df[target_col]

    # Делим на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=42)

    # Масштабирование
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ==============================
# 4. Обучение моделей и вывод метрик
# ==============================
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

# ==============================
# 5. Визуализация корреляции
# ==============================
def plot_correlation(df):
    corr = df.corr()
    plt.figure(figsize=(12, 12))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

# ==============================
# 6. Основной запуск
# ==============================
if __name__ == "__main__":
    df = load_and_prepare_data('filled_all_data.csv')
    plot_correlation(df)
    df, encoders = encode_features(df)
    X_train, X_test, y_train, y_test, scaler = prepare_train_test(df)
    train_and_evaluate_models(X_train, X_test, y_train, y_test)
