# data_processing_and_visualization.py

# %%
import pandas as pd
import numpy as np
import re
import ast
from tqdm import tqdm
from pymorphy3 import MorphAnalyzer
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import seaborn as sns

tqdm.pandas()

# ==============================
# 1. Очистка дубликатов и нормализация id_hash
# ==============================
GEO_WORDS = {
    'область', 'район', 'город', 'деревня', 'село', 'посёлок',
    'республика', 'край', 'округ', 'улица', 'проспект', 'аллея',
    'переулок', 'площадь', 'шоссе', 'набережная', 'бульвар', 'микрорайон'
}

REGIONS = {
    'москва', 'санкт-петербург', 'новосибирск', 'екатеринбург', 'казань',
    'нижний новгород', 'челябинск', 'самара', 'омск', 'ростов-на-дону',
    'красноярск', 'пермь', 'воронеж', 'волгоград', 'краснодар'
}


def is_geo_name(text):
    """Проверяет, является ли текст географическим названием"""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    words = text_lower.split()
    if any(geo_word in text_lower for geo_word in GEO_WORDS):
        return True
    if len(words) == 1 and words[0] in REGIONS:
        return True
    return False


def extract_name_from_text(text):
    """Извлекает имя из текста с фильтрацией гео-названий"""
    if not isinstance(text, str) or not text.strip():
        return None
    text = re.sub(r'^\s*Re:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[«»"“”]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(
        r'\b(пропал[а-яё]*|жив[а-яё]*|погиб[а-яё]*|пропаж[а-яё]*|стоп\b)',
        '', flags=re.IGNORECASE, string=text
    )
    patterns = [
        r'([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})',
        r'([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.)',
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            name = matches[-1].group(0).strip()
            name = re.sub(r'([А-ЯЁ])\.\s*([А-ЯЁ])\.', r'\1. \2.', name)
            if not is_geo_name(name):
                return name
    return None


def clean_id_hash(text):
    """Удаляет статусные слова из id_hash"""
    if not isinstance(text, str):
        return text
    text = re.sub(
        r'\b(пропал[а-яё]*|жив[а-яё]*|погиб[а-яё]*|пропаж[а-яё]*)\b',
        '', flags=re.IGNORECASE, string=text
    )
    return text.strip()


def detect_status(text):
    """Определяет статус по ключевым словам"""
    if not isinstance(text, str):
        return None
    text = text.lower()
    if re.search(r'\bжив(ая|ой)?\b|\bжив[ёуыаоэяию]\b', text):
        return 'жив(а)'
    elif re.search(r'\bпогиб(ла|ший|шая)?\b|\bпогиб[ёуыаоэяию]\b|\bгибел\b', text):
        return 'погиб(ла)'
    elif re.search(r'\bпропал(а|и)?\b|\bпропаж(а|и)\b|\bпропавш\b', text):
        return 'пропал(а)'
    return None


def process_data(file_path, output_path='updated_file.csv'):
    """Полная обработка данных"""
    try:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp1251')

        # Заполнение и очистка id_hash
        df['name_from_head'] = df['head'].progress_apply(extract_name_from_text)
        valid_mask = df['name_from_head'].notna() & ~df['name_from_head'].apply(is_geo_name)
        df.loc[valid_mask, 'id_hash'] = df.loc[valid_mask, 'name_from_head']
        df['id_hash'] = df['id_hash'].apply(clean_id_hash)

        # Обновление статуса
        for index, row in df.iterrows():
            content_status = detect_status(row['content'])
            head_status = detect_status(row['head'])
            if content_status != 'пропал(а)' and content_status:
                df.at[index, 'status'] = content_status
            elif head_status:
                df.at[index, 'status'] = head_status

        # Очистка пола
        morph = MorphAnalyzer()

        def determine_gender(head_line):
            first_check = 'неопр'
            if not pd.isna(head_line):
                tokens = [w.lower() for w in re.findall(r'\w+', head_line)]
                for token in tokens:
                    tag = str(morph.parse(token)[0].tag)
                    if 'femn' in tag: return 'жен'
                    if 'masc' in tag: return 'муж'
                if any('plur' in str(morph.parse(w)[0].tag) for w in tokens):
                    return 'мн'
            return first_check

        df['gender_new'] = df['head'].progress_apply(determine_gender)

        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Файл обработан и сохранен в {output_path}")
        return df
    except Exception as e:
        print(f"Ошибка обработки: {e}")
        return None


# ==============================
# 2. Преобразование дат
# ==============================
def split_date(r_date):
    if not pd.isna(r_date):
        try:
            year, month, day_hour = r_date.split('-', 2)
            day, hour = day_hour.split()
            return int(year), int(month), int(day), int(hour.split(':')[0])
        except:
            return np.nan, np.nan, np.nan, np.nan
    return np.nan, np.nan, np.nan, np.nan


def process_dates(df):
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[[f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_hour']] = (
            df[col].apply(split_date).apply(pd.Series)
        )
    if 'search_period' in df.columns:
        df['search_period'] = df['search_period'].apply(lambda x: int(str(x).split()[0]) if pd.notna(x) else x)
    return df


# ==============================
# 3. Визуализация
# ==============================
def plot_age_distribution(df):
    def parse_age(age_str):
        if pd.isna(age_str):
            return []
        try:
            ages = ast.literal_eval(age_str)
            return [ages] if isinstance(ages, int) else ages
        except:
            return []

    df['age_list'] = df['age'].apply(parse_age)
    df_expanded = df.explode('age_list')
    df_expanded = df_expanded[df_expanded['age_list'].notna()]
    df_expanded['age_list'] = df_expanded['age_list'].astype(int)
    counts = df_expanded.groupby(['age_list', 'status']).size().unstack(fill_value=0)
    for col in ['пропал(а)', 'жив(а)', 'погиб(ла)']:
        if col not in counts.columns: counts[col] = 0
    counts = counts.sort_index()
    ages = counts.index.tolist()
    dead = counts['погиб(ла)'].tolist()
    missing = counts['пропал(а)'].tolist()
    alive = counts['жив(а)'].tolist()
    plt.figure(figsize=(18, 6))
    plt.bar(ages, dead, label='Погиб(ла)', color='#EA3C2D')
    plt.bar(ages, missing, bottom=dead, label='Пропал(а)', color='#F6C944')
    plt.bar(ages, alive, bottom=[d + m for d, m in zip(dead, missing)], label='Жив(а)', color='#85BA38')
    plt.xlabel('Возраст')
    plt.ylabel('Количество пропавших')
    plt.title('Распределение по возрастам и статусам')
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    custom_legend = [Patch(color='#85BA38', label='Жив(а)'),
                     Patch(color='#F6C944', label='Пропал(а)'),
                     Patch(color='#EA3C2D', label='Погиб(ла)')]
    plt.legend(handles=custom_legend, loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_gender_distribution(df):
    gender_counts = df['gender_new'].value_counts().drop(labels=['неопр', None], errors='ignore')
    labels_map = {'муж': 'Мужчины', 'жен': 'Женщины', 'мн': 'Несколько человек'}
    colors = {'муж': '#97CCE8', 'жен': '#F4B9C1', 'мн': '#9E9E9E'}
    labels = [labels_map.get(g, g) for g in gender_counts.index]
    values = gender_counts.values
    colors_used = [colors.get(g, '#CCCCCC') for g in gender_counts.index]
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(values, labels=None, colors=colors_used, autopct='%1.1f%%', startangle=90,
                                       counterclock=False, textprops={'color': 'black', 'fontsize': 14},
                                       pctdistance=0.75)
    for i, wedge in enumerate(wedges):
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = 1.1 * np.cos(np.deg2rad(ang))
        y = 1.1 * np.sin(np.deg2rad(ang))
        ha = 'left' if x > 0 else 'right'
        plt.text(x, y, labels[i], ha=ha, va='center', fontsize=14)
    plt.setp(autotexts, weight='normal')
    plt.axis('equal')
    plt.title('Распределение по полу')
    plt.tight_layout()
    plt.show()


def plot_demographic_pyramid(df):
    df['age'] = df['age'].astype(str).str.extract(r'(\d+)').astype('Int64')
    df = df[df['age'].notna()]
    df = df[df['gender_new'].isin(['муж', 'жен'])]
    df['status'] = df['status'].fillna('nan')

    def simplify_status(s):
        if s == 'жив(а)':
            return 'жив'
        elif s == 'погиб(ла)':
            return 'мертв'
        else:
            return 'пропал/nan'

    df['status_simple'] = df['status'].apply(simplify_status)
    grouped = df.groupby(['age', 'gender_new', 'status_simple']).size().reset_index(name='count')
    pivot = grouped.pivot_table(index='age', columns=['gender_new', 'status_simple'], values='count', fill_value=0)
    ages = sorted(pivot.index)
    colors_male = {'жив': mcolors.to_rgba('#5EB8E7', alpha=1),
                   'пропал/nan': mcolors.to_rgba('#5EB8E7', alpha=0.6),
                   'мертв': mcolors.to_rgba('#5EB8E7', alpha=0.3)}
    colors_female = {'жив': mcolors.to_rgba('#FAA4B0', alpha=1),
                     'пропал/nan': mcolors.to_rgba('#FAA4B0', alpha=0.6),
                     'мертв': mcolors.to_rgba('#FAA4B0', alpha=0.3)}
    fig, ax = plt.subplots(figsize=(12, 18))
    widths_male = np.zeros(len(ages))
    widths_female = np.zeros(len(ages))
    female_handles = []
    female_labels = []
    male_handles = []
    male_labels = []
    for status in ['жив', 'пропал/nan', 'мертв']:
        male_vals = pivot.get(('муж', status), pd.Series([0] * len(ages), index=ages))
        female_vals = pivot.get(('жен', status), pd.Series([0] * len(ages), index=ages))
        h_m = ax.barh(ages, male_vals, left=widths_male, color=colors_male[status], align='center')
        widths_male += male_vals.values
        h_f = ax.barh(ages, -female_vals, left=-widths_female, color=colors_female[status], align='center')
        widths_female += female_vals.values
        male_handles.append(h_m[0])
        male_labels.append(status)
        female_handles.append(h_f[0])
        female_labels.append(status)
    ax.set_yticks(ages)
    ax.set_yticklabels([str(a) for a in ages])
    ax.set_xlabel('Количество пропавших')
    ax.set_ylabel('Возраст')
    ax.set_title('Распределение по возрастам, статусу и полу')
    max_count = max(widths_male.max(), widths_female.max())
    step = 25
    xticks_pos = list(range(0, int(max_count) + step, step))
    xticks_full = [-x for x in xticks_pos[::-1]] + xticks_pos
    ax.set_xticks(xticks_full)
    ax.set_xticklabels([str(abs(x)) for x in xticks_full])
    ax.axvline(0, color='black', linewidth=0.8)
    leg_female = ax.legend(female_handles, female_labels, loc='upper left', title='Женщины')
    leg_male = ax.legend(male_handles, male_labels, loc='upper right', title='Мужчины')
    ax.add_artist(leg_female)
    plt.tight_layout()
    plt.show()


# ==============================
# 4. Основной запуск
# ==============================
if __name__ == "__main__":
    df = process_data("LizaAlert.csv", "updated_file.csv")
    if df is not None:
        df = process_dates(df)
        plot_age_distribution(df)
        plot_gender_distribution(df)
        plot_demographic_pyramid(df)
