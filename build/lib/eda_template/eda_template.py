# =========================================================
# УНИВЕРСАЛЬНЫЙ EDA-ШАБЛОН ДЛЯ JUPYTER
# quick_eda(df)  -> быстрый обзор датасета
# full_eda(df)   -> расширенный EDA с графиками
#
# Как использовать:
# 1) Загрузить DataFrame в переменную df
# 2) quick_eda(df)
# 3) full_eda(df, target='target', task='classification')
#
# task:
# - 'classification'
# - 'regression'
# - None
# =========================================================

import warnings

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import phik 
from IPython.display import display, Markdown

# ---------------------------------------------------------
# НАСТРОЙКИ НОУТБУКА
# ---------------------------------------------------------
def _configure_notebook():
    """Настраивает вывод таблиц и стиль графиков."""
    pd.set_option("display.max_columns", None) # сколько колонок DataFrame pandas показывает при выводе
    pd.set_option("display.max_rows", 200) # максимальным числом строк, которое pandas показывает при выводе DataFrame
    sns.set_theme(style="whitegrid") # белый фон с сеткой (grid) по осям
    warnings.filterwarnings("ignore") # отключение предупреждений, чтобы не засорять вывод в Jupyter

# ---------------------------------------------------------
# СЛУЖЕБНЫЕ ФУНКЦИИ
# ---------------------------------------------------------

def _md(text):
    # Красивый текстовый заголовок внутри Jupyter
    display(Markdown(text))


def _get_types(df, cat_threshold=12):
    """
    Определяем типы колонок:
    - numeric: числовые признаки
    - categorical: категориальные признаки
    - datetime: даты
    - numeric_as_cat: числовые колонки с малым числом уникальных значений,
      которые в EDA часто удобнее смотреть как категории
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()

    numeric_as_cat = [col for col in num_cols if df[col].nunique(dropna=False) <= cat_threshold]
    num_cols = [col for col in num_cols if col not in numeric_as_cat]
    cat_cols = list(dict.fromkeys(cat_cols + numeric_as_cat))

    return num_cols, cat_cols, dt_cols, numeric_as_cat


def _exclude_id_columns(
    df: pd.DataFrame,
    exclude_cols=None,
    *,
    target=None,
) -> pd.DataFrame:
    """
    Возвращает копию DataFrame без явно указанных столбцов.

    Защита:
    - проверяет наличие target;
    - запрещает исключать target;
    - сообщает об опечатках в названиях столбцов;
    - не изменяет исходный DataFrame.
    """
    if not df.columns.is_unique:
        raise ValueError("Названия столбцов должны быть уникальными.")

    if target is not None and target not in df.columns:
        raise ValueError(
            f"Целевая переменная {target!r} отсутствует в DataFrame."
        )

    if exclude_cols is None:
        exclude_cols = []
    elif isinstance(exclude_cols, str):
        exclude_cols = [exclude_cols]
    else:
        exclude_cols = list(exclude_cols)

    if target is not None and target in exclude_cols:
        raise ValueError(
            f"Нельзя исключать целевую переменную {target!r}. "
            "Уберите её из exclude_cols."
        )

    unknown_cols = [
        col for col in exclude_cols
        if col not in df.columns
    ]

    if unknown_cols:
        raise ValueError(
            f"В exclude_cols указаны отсутствующие столбцы: {unknown_cols}"
        )

    return df.drop(columns=exclude_cols).copy()


def _summary_table(df):
    """
    Строим компактную сводную таблицу по колонкам:
    тип, число уникальных значений, пропуски и доля пропусков
    """
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(dropna=False),
        "n_missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2)
    })
    return info.sort_values(["missing_%", "n_unique"], ascending=[False, True])


def _plot_missing(df):
    """
    График по пропускам.
    Показываем только те колонки, где пропуски действительно есть.
    """
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]

    if miss.empty:
        print("Пропусков не найдено.")
        return

    plt.figure(figsize=(10, max(4, len(miss) * 0.35)))
    sns.barplot(x=miss.values, y=miss.index, orient="h")
    plt.title("Доля пропусков по колонкам")
    plt.xlabel("Процент пропусков")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def _plot_numeric(df, cols, max_cols=10):
    """
    Гистограммы для числовых признаков.
    Ограничиваем число графиков, чтобы ноутбук не тормозил.
    """
    cols = cols[:max_cols]
    if not cols:
        print("Нет числовых признаков для графиков.")
        return

    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), kde=True, bins=30, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def _plot_boxplots(df, cols, max_cols=10):
    """
    Boxplot для числовых признаков.
    Для быстрой проверки выбросов.
    """
    cols = cols[:max_cols]
    if not cols:
        return

    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def _plot_lineplots(df, date_col, cols, max_cols=10):
    """
    Line plot для временных рядов.
    Подходит для просмотра динамики признаков по дате.
    """
    cols = cols[:max_cols]
    if not cols:
        return

    df_plot = df.copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col])
    df_plot = df_plot.sort_values(date_col)

    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        sns.lineplot(data=df_plot, x=date_col, y=col, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def _plot_lineplots_agg(df, date_col, cols, agg='mean', max_cols=10):
    """
    Line plot для временных рядов с агрегацией по дате.
    
    Параметры:
    - df: DataFrame
    - date_col: название столбца с датой
    - cols: список числовых столбцов для графиков
    - agg: способ агрегации ('mean', 'sum', 'median', ...)
    - max_cols: максимум графиков
    """
    cols = cols[:max_cols]
    if not cols:
        return

    df_plot = df.copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col])

    df_agg = (
        df_plot
        .groupby(date_col, as_index=False)[cols]
        .agg(agg)
        .sort_values(date_col)
    )

    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        sns.lineplot(data=df_agg, x=date_col, y=col, ax=ax)
        ax.set_title(f"{col} ({agg})")
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def _plot_corr(df, numeric_cols, max_cols=20, method="pearson"):
    """
    Матрица корреляций по числовым признакам.

    Параметры:
    - df: DataFrame
    - numeric_cols: список числовых колонок
    - max_cols: максимум колонок для матрицы
    - method: 'pearson' или 'spearman'
      pearson  -> линейная связь, чувствителен к выбросам, предполагает ~нормальность
      spearman -> ранговая корреляция, устойчивее к выбросам и нелинейным монотонным связям
    """
    cols = numeric_cols[:max_cols]
    if len(cols) < 2:
        print("Недостаточно числовых признаков для матрицы корреляций.")
        return

    corr = df[cols].corr(method=method, numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(f"Матрица корреляций ({method}) по числовым признакам")
    plt.tight_layout()
    plt.show()

def _plot_phik_corr(df):
    """
    Матрица PhiK-корреляций для любых типов признаков.
    """

    df_phik = df.copy()
    timedelta_cols = df_phik.select_dtypes(include=["timedelta64[ns]"]).columns
    cols_to_drop = timedelta_cols.tolist()
    df_phik = df_phik.drop(columns=cols_to_drop)
    
    corr = df_phik.phik_matrix()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Матрица PhiK-корреляций (смешанные типы признаков)")
    plt.tight_layout()
    plt.show()

def _plot_categorical(df, cols, max_cols=8, top_n=12):
    """
    Countplot для категориальных признаков.
    Для каждой колонки показываем только top-N самых частых значений.
    """
    cols = cols[:max_cols]

    if not cols:
        print("Нет категориальных признаков для графиков.")
        return

    for col in cols:
        plt.figure(figsize=(10, 4))
        
        # Приводим значения к строковому типу и отдельно помечаем пропуски
        s = df[col].astype("string").fillna("<NA>")

        # Считаем частоты значений
        value_counts = s.value_counts()
        n_unique = len(value_counts)
        
        # Определяем порядок отображения категорий
        order = value_counts.head(top_n).index
        
        sns.countplot(y=s, order=order)

        # Если категорий мало, пишем обычное распределение значений
        if n_unique < top_n:
            plt.title(f"{col} – распределение значений")
        else:
            plt.title(f"{col} – top {top_n} категорий")
            
        plt.xlabel("Количество")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()


def _target_eda(df, target, task, num_cols, cat_cols):
    """
    Анализ целевой переменной:
    - classification: баланс классов + boxplot по числовым признакам
    - regression: распределение target + scatterplot с числовыми признаками
    """
    if target is None:
        return

    if target not in df.columns:
        print(f"Колонка target='{target}' не найдена.")
        return

    _md(f"###### Анализ target: `{target}`")

    if task == "classification":
        display(df[target].value_counts(dropna=False))

        plt.figure(figsize=(8, 4))
        sns.countplot(x=df[target].astype("string").fillna("<NA>"))
        plt.title("Распределение классов")
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.show()

        # Показываем boxplot только по нескольким первым числовым признакам
        for col in [c for c in num_cols if c != target][:5]:
            plt.figure(figsize=(7, 4))
            sns.boxplot(data=df, x=target, y=col)
            plt.title(f"{col} по классам target")
            plt.tight_layout()
            plt.show()

    elif task == "regression":
        display(df[target].describe())

        plt.figure(figsize=(8, 4))
        sns.histplot(df[target].dropna(), kde=True, bins=30)
        plt.title("Распределение target")
        plt.tight_layout()
        plt.show()

        for col in [c for c in num_cols if c != target][:5]:
            plt.figure(figsize=(7, 4))
            sns.scatterplot(data=df, x=col, y=target, alpha=0.7)
            plt.title(f"{col} vs {target}")
            plt.tight_layout()
            plt.show()

    else:
        print("Для анализа target укажите task='classification' или task='regression'.")


def _target_relations(df,
    target,
    top_n=15,
    corr_method="pearson",
    use_phik=True,
    phik_interval_cols=None
):
    """
    Анализ связи признаков с целевой переменной.
     Что делает:
    1) Пирсон / Спирмен:
       - только для числового target и числовых признаков
       - полезно для регрессии и анализа линейной / монотонной связи
    2) PhiK:
       - работает со смешанными типами данных
       - позволяет оценить связь признаков с target даже если
         среди признаков есть категориальные и дискретные переменные
    3) выводит таблицы и графики   
    
    Параметры:
    - df: pandas DataFrame
    - target: имя целевой колонки
    - top_n: сколько признаков показывать
    - corr_method: 'pearson' или 'spearman' для числового блока
    - use_phik: считать ли PhiK
    - phik_interval_cols: список числовых interval-признаков для phik;
      если None, phik попробует обработать автоматически
    """
    if target not in df.columns:
        print(f"Колонка target='{target}' не найдена.")
        return

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    _md("###### Связь признаков с target")

    # ------------------------------------------------------
    # 1. Блок обычной корреляции: только num-target
    # ------------------------------------------------------
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    # Пирсон / Спирмен считаем только если target числовой
    if pd.api.types.is_numeric_dtype(y):
        if num_cols:
            _md(f"###### Корреляция ({corr_method}) числовых признаков с target")

            # Берём только числовые колонки и сам target
            corr_df = df[num_cols + [target]].copy()

            # corrwith удобен именно для расчёта корреляции каждого признака с target
            corr_series = (
                corr_df[num_cols]
                .corrwith(corr_df[target], method=corr_method)
                .sort_values(key=lambda s: s.abs(), ascending=False)
            )

            display(corr_series.to_frame(name=f"{corr_method}_corr_with_target").head(top_n))

            plt.figure(figsize=(8, max(4, top_n * 0.35)))
            sns.barplot(
                x=corr_series.head(top_n).values,
                y=corr_series.head(top_n).index,
                orient="h"
            )
            plt.title(f"Топ числовых признаков по {corr_method}-корреляции с target")
            plt.xlabel("Коэффициент корреляции")
            plt.ylabel("")
            plt.tight_layout()
            plt.show()
        else:
            print("Нет числовых признаков для расчёта стандартной корреляции с target.")
    else:
        print("Target не является числовым: стандартная корреляция Пирсона/Спирмена пропущена.")

    # ------------------------------------------------------
    # 2. Блок PhiK: mixed data / общая сила связи
    # ------------------------------------------------------
    if use_phik:
        try:
            _md("###### PhiK: связь признаков с target")

            df_phik = df.copy()
            timedelta_cols = df_phik.select_dtypes(include=["timedelta64[ns]"]).columns
            cols_to_drop = timedelta_cols.tolist()
            df_phik = df_phik.drop(columns=cols_to_drop)
            
            # interval_cols можно передать явно
            if phik_interval_cols is None:
                phik_corr = df_phik.phik_matrix()
            else:
                phik_corr = df_phik.phik_matrix(interval_cols=phik_interval_cols)

            if target not in phik_corr.columns:
                print("Не удалось получить PhiK для target.")
                return

            phik_target = (
                phik_corr[target]
                .drop(labels=[target], errors="ignore")
                .sort_values(ascending=False)
            )

            display(phik_target.to_frame(name="phi_k_with_target").head(top_n))

            plt.figure(figsize=(8, max(4, top_n * 0.35)))
            sns.barplot(
                x=phik_target.head(top_n).values,
                y=phik_target.head(top_n).index,
                orient="h"
            )
            plt.title("Топ признаков по PhiK-связи с target")
            plt.xlabel("PhiK")
            plt.ylabel("")
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Библиотека phik не установлена. Установи её: pip install phik")
        except Exception as e:
            print("Не удалось посчитать PhiK:", e)


# ---------------------------------------------------------
# ОСНОВНЫЕ ФУНКЦИИ
# ---------------------------------------------------------
def quick_eda(df, cat_threshold=12, exclude_cols=None, target=None):
    """
    БЫСТРЫЙ EDA:
    Используется для первого знакомства с датасетом.
    Даёт компактную сводку без перегруза графиками.
    """
    df_copy = _exclude_id_columns(
            df,
            exclude_cols=exclude_cols,
            target=target,
        ) # иключение колонок с 'id
    
    _configure_notebook() # настройка ноутбука

    _md("#### Quick EDA")

    print("Размер датасета:", df.shape)
    print("Информация по датасету:")
    display(df.info())
    print("\nПервые строки:")
    display(df.head())

    print("\nСводка по колонкам:")
    display(_summary_table(df))

    print("\nЧисло дубликатов:", df.duplicated().sum())
    
    num_cols, cat_cols, dt_cols, numeric_as_cat = _get_types(df_copy, cat_threshold=cat_threshold)

    print("\nТипы признаков:")
    print("Числовые:", len(num_cols))
    print("Категориальные:", len(cat_cols))
    print("Datetime:", len(dt_cols))
    print("Числовые, трактуемые как категории:", numeric_as_cat)

    if num_cols:
        _md("##### Числовые признаки: describe()")
        display(df_copy[num_cols].describe().T)

    if cat_cols:
        _md("##### Категориальные признаки")
        cat_info = pd.DataFrame({
            "n_unique": [df_copy[col].nunique(dropna=False) for col in cat_cols],
            "top_value": [
                df_copy[col].astype("string").fillna("<NA>").value_counts(dropna=False).index[0]
                for col in cat_cols
            ]
        }, index=cat_cols).sort_values("n_unique")
        display(cat_info)

    _md("##### Пропуски")
    _plot_missing(df_copy)


def full_eda(
    df,
    target=None,
    task=None,
    cat_threshold=12,
    max_numeric_plots=10,
    max_categorical_plots=6,
    top_n_categories=12,
    exclude_cols=None
):
    """
    ПОЛНЫЙ EDA:
    Подходит для более детального анализа данных.
    Что делает:
    - показывает обзор датасета
    - показывает качество данных
    - анализирует числовые признаки
    - анализирует категориальные признаки
    - строит корреляции
    - при наличии target делает его отдельный анализ
    """
    df_copy = _exclude_id_columns(
        df,
        exclude_cols=exclude_cols,
        target=target,
    ) # иключение колонок с 'id'
    
    _configure_notebook() # настройка ноутбука
    _md("#### Full EDA")

    print("Размер датасета:", df.shape)
    print("Информация по датасету:")
    display(df.info())
    print("\nПервые строки:")
    display(df.head())

    _md("##### Сводка по колонкам")
    display(_summary_table(df))

    print("Число дубликатов:", df.duplicated().sum())
    
    constant_cols = [col for col in df_copy.columns if df_copy[col].nunique(dropna=False) <= 1]
    print("Константные колонки:", constant_cols if constant_cols else "Нет")

    num_cols, cat_cols, dt_cols, numeric_as_cat = _get_types(df_copy, cat_threshold=cat_threshold)

    # Если target указан, убираем его из списков признаков для отдельных графиков
    if target in num_cols:
        num_cols.remove(target)
    if target in cat_cols:
        cat_cols.remove(target)
    if target in dt_cols:
        dt_cols.remove(target)

    _md("##### Группы признаков")
    print("Числовые:", num_cols)
    print("Категориальные:", cat_cols)
    print("Datetime:", dt_cols)

    _md("##### Пропуски")
    _plot_missing(df_copy)

    _md("##### Числовые признаки")
    if num_cols:
        display(df_copy[num_cols].describe().T)

        # Простейшая оценка выбросов по правилу IQR
        q1 = df_copy[num_cols].quantile(0.25)
        q3 = df_copy[num_cols].quantile(0.75)
        iqr = q3 - q1
        
        outliers = (((df_copy[num_cols] < (q1 - 1.5 * iqr)) | (df_copy[num_cols] > (q3 + 1.5 * iqr))).sum())
        outlier_info = pd.DataFrame({
            "outliers_count": outliers,
            "outliers_%": (outliers / len(df_copy) * 100).round(2)
        }).sort_values("outliers_%", ascending=False)
        print("Оценка выбросов по правилу IQR")
        display(outlier_info)

        _plot_numeric(df_copy, num_cols, max_cols=max_numeric_plots)
        _plot_boxplots(df_copy, num_cols, max_cols=max_numeric_plots)
        _plot_corr(df_copy, num_cols)
    else:
        print("Числовые признаки не найдены.")

    _md("##### Категориальные признаки")
    if cat_cols:
        cat_info = pd.DataFrame({
            "n_unique": [df_copy[col].nunique(dropna=False) for col in cat_cols],
            "missing_%": [(df_copy[col].isna().mean() * 100).round(2) for col in cat_cols],
            "top_values": [
                ", ".join(
                    map(str, df_copy[col].astype("string").fillna("<NA>").value_counts(dropna=False).head(3).index.tolist())
                )
                for col in cat_cols
            ]
        }, index=cat_cols).sort_values(["n_unique", "missing_%"])

        display(cat_info)
        _plot_categorical(df_copy, cat_cols, max_cols=max_categorical_plots, top_n=top_n_categories)
    else:
        print("Категориальные признаки не найдены.")

    _md("##### Datetime-признаки")
    if dt_cols:
        dt_info = pd.DataFrame({
            "min": [df_copy[col].min() for col in dt_cols],
            "max": [df_copy[col].max() for col in dt_cols],
            "missing_%": [(df_copy[col].isna().mean() * 100).round(2) for col in dt_cols]
        }, index=dt_cols)
        display(dt_info)
    else:
        print("Datetime-признаки не найдены.")

    _plot_phik_corr(df_copy)
    
    if target is not None:
        _md("##### Target")
        _target_eda(df_copy, target, task, num_cols, cat_cols)
        _target_relations(df_copy, target, corr_method="pearson", use_phik=True)
