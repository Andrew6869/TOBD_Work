import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. НАСТРОЙКА СТРАНИЦЫ
# ============================================================
st.set_page_config(
    page_title="ТОБД 2.6 — Аналитический дашборд",
    layout="wide"
)

st.title("Анализ потребительских предпочтений")
st.markdown("Итоговый интерактивный дашборд по этапам 2.2–2.5: проверка гипотез **H1–H5**, **RFM-сегментация**, результаты **ML-моделей** и **бенчмарки производительности**.")

# ============================================================
# 2. ЗАГРУЗКА И СИНХРОНИЗАЦИЯ ДАННЫХ (как в п.2.3 и п.2.4)
# ============================================================
def assign_category(desc):
    if pd.isna(desc):
        return 'Other'
    d = str(desc).lower()
    if any(w in d for w in ['christmas', 'xmas', 'santa', 'advent', 'reindeer']):
        return 'Christmas'
    if any(w in d for w in ['garden', 'flower', 'plant', 'bird', 'outdoor']):
        return 'Garden'
    if any(w in d for w in ['kitchen', 'bowl', 'cup', 'mug', 'plate', 'teapot']):
        return 'Kitchen'
    if any(w in d for w in ['bag', 'purse', 'handbag', 'wallet']):
        return 'Bags'
    if any(w in d for w in ['toy', 'game', 'play', 'doll']):
        return 'Toys'
    if any(w in d for w in ['light', 'lamp', 'candle', 'holder']):
        return 'Lighting'
    return 'Other'

@st.cache_data
def load_data():
    fact = pd.read_parquet('fact_monthly.parquet')
    retail = pd.read_parquet('retail_temp.parquet')
    
    # Применяем строго те же правила очистки, что в 2.3 / 2.4,
    # чтобы метрики полностью совпадали с отчётом 2.4
    retail['Invoice'] = retail['Invoice'].astype(str)
    retail = retail[~retail['Invoice'].str.startswith('C')]
    retail = retail[(retail['Quantity'] > 0) & (retail['Price'] > 0)]
    retail = retail.drop_duplicates()
    retail = retail[retail['Customer ID'].notna()].copy()
    retail['Customer ID'] = retail['Customer ID'].astype(int)
    
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])
    retail['Revenue'] = retail['Quantity'] * retail['Price']
    retail['YearMonth'] = retail['InvoiceDate'].dt.to_period('M').astype(str)
    retail['Month'] = retail['InvoiceDate'].dt.month
    retail['DayOfWeek'] = retail['InvoiceDate'].dt.dayofweek
    retail['Hour'] = retail['InvoiceDate'].dt.hour
    retail['Category'] = retail['Description'].apply(assign_category)
    
    return fact, retail

fact, retail = load_data()

# ============================================================
# 3. БОКОВАЯ ПАНЕЛЬ (ФИЛЬТРЫ)
# ============================================================
st.sidebar.header("Фильтры (Retail)")

all_months = sorted(retail['Month'].unique())
selected_months = st.sidebar.multiselect(
    "Месяцы",
    options=all_months,
    default=all_months,
    format_func=lambda x: f"{x:02d}"
)

all_categories = sorted(retail['Category'].unique())
selected_categories = st.sidebar.multiselect(
    "Категории товаров",
    options=all_categories,
    default=all_categories
)

all_countries = sorted(retail['Country'].unique())
selected_countries = st.sidebar.multiselect(
    "Страны (по умолчанию — все)",
    options=all_countries,
    default=all_countries
)

# Фильтрация
retail_f = retail[
    (retail['Month'].isin(selected_months)) &
    (retail['Category'].isin(selected_categories)) &
    (retail['Country'].isin(selected_countries))
].copy()

if retail_f.empty:
    st.warning("По выбранным фильтрам нет данных. Измените параметры на боковой панели.")
    st.stop()

# ============================================================
# 4. ВЕРХНИЕ KPI-КАРТОЧКИ
# ============================================================
st.subheader("Ключевые показатели (по отфильтрованным данным)")

total_revenue = retail_f['Revenue'].sum()
n_customers = retail_f['Customer ID'].nunique()
n_orders = retail_f['Invoice'].nunique()
avg_check = total_revenue / n_orders if n_orders > 0 else 0

# Расчёт Gini и доли Парето (как в 2.4)
cust_rev = retail_f.groupby('Customer ID')['Revenue'].sum().values
if len(cust_rev) > 1 and cust_rev.sum() > 0:
    x_asc = np.sort(cust_rev)
    n_c = len(x_asc)
    gini = (2 * np.sum(np.arange(1, n_c + 1) * x_asc)) / (n_c * x_asc.sum()) - (n_c + 1) / n_c
    
    x_desc = x_asc[::-1]
    cum_rev = np.cumsum(x_desc) / x_desc.sum()
    idx80 = np.searchsorted(cum_rev, 0.80) + 1
    share80 = idx80 / n_c
else:
    gini = 0.0
    share80 = 0.0
    cum_rev = np.array([1.0])

freq = retail_f.groupby('Customer ID')['Invoice'].nunique()
share1 = (freq == 1).mean() if len(freq) > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Выручка", f"{total_revenue:,.0f} £")
c2.metric("Клиентов / Заказов", f"{n_customers:,} / {n_orders:,}")
c3.metric("Индекс Gini", f"{gini:.3f}", help="0 = полное равенство, 1 = вся выручка у 1 клиента (в 2.4: 0.716)")
c4.metric("Доля для 80% выручки", f"{share80*100:.1f}%", help="Правило Парето (в 2.4: 26.0%)")
c5.metric("Доля однократных", f"{share1*100:.1f}%", help="Клиенты с 1 заказом (в 2.4: 34.4%)")

st.divider()

# ============================================================
# 5. НАВИГАЦИЯ ПО ВКЛАДКАМ
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Гипотезы H1–H5",
    "RFM-сегментация",
    "ML и Макро-контекст",
    "Бенчмарки (п.2.5)"
])

# ------------------------------------------------------------
# ВКЛАДКА 1: ГИПОТЕЗЫ H1–H5
# ------------------------------------------------------------
with tab1:
    # --- H1 ---
    st.markdown("### H1: Временные паттерны и сезонные индексы")
    col_h1_1, col_h1_2 = st.columns([2, 1])
    with col_h1_1:
        h1_data = retail_f.groupby('Month')['Revenue'].sum().reset_index().sort_values('Month')
        h1_data['Seasonal_Index'] = h1_data['Revenue'] / h1_data['Revenue'].mean()
        fig_h1 = px.bar(
            h1_data, x='Month', y='Revenue',
            title="Выручка и сезонный индекс по месяцам (Retail)",
            labels={'Month': 'Месяц', 'Revenue': 'Выручка, £', 'Seasonal_Index': 'Индекс'},
            color='Seasonal_Index', color_continuous_scale='Blues',
            hover_data={'Seasonal_Index': ':.2f'},
            text_auto='.2s'
        )
        fig_h1.update_xaxes(type='category')
        fig_h1.update_layout(bargap=0.3)
        st.plotly_chart(fig_h1, use_container_width=True)
    with col_h1_2:
        st.info(
            "**Вывод по H1 (Подтверждается):**\n\n"
            "* **Retail:** пик выручки приходится на ноябрь (индекс **1.69**), спад — на февраль (**0.65**).\n"
            "* **Walmart (витрина 2.3):** пик в декабре (**1.22**), спад в январе (**0.88**).\n"
            "* Закупки и промо-акции оптимально запускать за 1–2 месяца до пика."
        )

    st.divider()

    # --- H2 и H5 ---
    col_h2, col_h5 = st.columns(2)
    with col_h2:
        st.markdown("### H2: Правило Парето (20/80)")
        n_l = len(cum_rev)
        x_lorenz = np.concatenate(([0.0], np.arange(1, n_l + 1) / n_l))
        y_lorenz = np.concatenate(([0.0], cum_rev))

        fig_h2 = go.Figure()
        fig_h2.add_trace(go.Scatter(x=x_lorenz, y=y_lorenz, mode='lines', name='Кривая Лоренца', line=dict(color='steelblue', width=2.5)))
        fig_h2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Равномерное', line=dict(dash='dash', color='gray')))
        fig_h2.add_hline(y=0.8, line_dash="dot", line_color="red", annotation_text="80% выручки")
        fig_h2.add_vline(x=share80, line_dash="dot", line_color="red", annotation_text=f"{share80*100:.1f}% клиентов")
        fig_h2.update_layout(
            title=f"Кривая Лоренца (Gini = {gini:.3f})",
            xaxis_title="Доля клиентов (накопленная)",
            yaxis_title="Доля выручки (накопленная)"
        )
        st.plotly_chart(fig_h2, use_container_width=True)

    with col_h5:
        st.markdown("### H5: Частота заказов на клиента")
        freq_hist = freq.value_counts().sort_index().reset_index()
        freq_hist.columns = ['orders', 'count']
        fig_h5 = px.bar(
            freq_hist[freq_hist['orders'] <= 20],
            x='orders', y='count',
            title=f"Распределение числа заказов (доля однократных = {share1*100:.1f}%)",
            labels={'orders': 'Число заказов', 'count': 'Клиентов'},
            color_discrete_sequence=['teal']
        )
        fig_h5.add_vline(x=freq.mean(), line_dash="dash", line_color="red", annotation_text=f"Среднее: {freq.mean():.1f}")
        st.plotly_chart(fig_h5, use_container_width=True)

    st.divider()

    # --- H3 ---
    st.markdown("### H3: Динамика среднего чека по дням недели и часам")
    inv = retail_f.groupby('Invoice').agg(
        revenue=('Revenue', 'sum'),
        dow=('DayOfWeek', 'first'),
        hour=('Hour', 'first')
    ).reset_index()

    col_h3_1, col_h3_2 = st.columns(2)
    with col_h3_1:
        dow_map = {0: 'Пн', 1: 'Вт', 2: 'Ср', 3: 'Чт', 4: 'Пт', 5: 'Сб', 6: 'Вс'}
        dow_avg = inv.groupby('dow')['revenue'].agg(['mean', 'median', 'count']).reset_index()
        dow_avg['day_name'] = dow_avg['dow'].map(dow_map)
        fig_h3_dow = px.bar(
            dow_avg, x='day_name', y='mean',
            title="Средний чек по дням недели (Kruskal p = 2.65e-06)",
            labels={'day_name': 'День недели', 'mean': 'Средний чек, £', 'count': 'Заказов'},
            hover_data=['median', 'count'],
            color_discrete_sequence=['#2b5c8f']
        )
        st.plotly_chart(fig_h3_dow, use_container_width=True)

    with col_h3_2:
        hour_avg = inv.groupby('hour')['revenue'].agg(['mean', 'count']).reset_index()
        fig_h3_hour = px.bar(
            hour_avg, x='hour', y='mean',
            title="Средний чек по часам суток (Kruskal p = 3.46e-24)",
            labels={'hour': 'Час дня', 'mean': 'Средний чек, £', 'count': 'Заказов'},
            hover_data=['count'],
            color_discrete_sequence=['coral']
        )
        st.plotly_chart(fig_h3_hour, use_container_width=True)

    st.divider()

    # --- H4 ---
    st.markdown("### H4: Сезонность категорий товаров")
    pivot = retail_f.pivot_table(index='Category', columns='Month', values='Revenue', aggfunc='sum', fill_value=0)
    pivot_share = pivot.div(pivot.sum(axis=1), axis=0)
    cv = (pivot_share.std(axis=1) / pivot_share.mean(axis=1)).sort_values(ascending=True)

    col_h4_1, col_h4_2 = st.columns([3, 2])
    with col_h4_1:
        fig_h4 = px.imshow(
            pivot_share,
            aspect='auto',
            title="Доля месяца в годовой выручке категории",
            labels=dict(x="Месяц", y="Категория", color="Доля"),
            color_continuous_scale='YlOrRd'
        )
        st.plotly_chart(fig_h4, use_container_width=True)

    with col_h4_2:
        cv_df = cv.reset_index()
        cv_df.columns = ['Category', 'CV']
        fig_cv = px.bar(
            cv_df, x='CV', y='Category', orientation='h',
            title="Коэффициент вариации (мера сезонности)",
            labels={'CV': 'CV (ст. откл. / среднее)', 'Category': ''},
            color='CV', color_continuous_scale='OrRd',
            text_auto='.2f'
        )
        st.plotly_chart(fig_cv, use_container_width=True)

# ------------------------------------------------------------
# ВКЛАДКА 2: RFM-СЕГМЕНТАЦИЯ
# ------------------------------------------------------------
with tab2:
    st.markdown("### RFM-кластеризация клиентов (KMeans, k = 4)")
    snap = retail_f['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = retail_f.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda s: (snap - s.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('Revenue', 'sum')
    ).reset_index()
    rfm = rfm[(rfm['Recency'] >= 0) & (rfm['Frequency'] > 0) & (rfm['Monetary'] > 0)]

    if len(rfm) >= 4:
        X = rfm[['Recency', 'Frequency', 'Monetary']].values
        X_log = np.log1p(X)
        X_scaled = StandardScaler().fit_transform(X_log)
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm['cluster'] = km.fit_predict(X_scaled)

        # Автоматическое именование как в п.2.4
        prof = rfm.groupby('cluster').agg(
            n=('Customer ID', 'count'),
            Recency=('Recency', 'mean'),
            Frequency=('Frequency', 'mean'),
            Monetary=('Monetary', 'mean'),
            Total_Rev=('Monetary', 'sum')
        )
        sorted_by_money = prof.sort_values('Monetary', ascending=False).index.tolist()
        seg_names = {
            sorted_by_money[0]: 'VIP',
            sorted_by_money[1]: 'Активные'
        }
        low_c = sorted_by_money[2:]
        if prof.loc[low_c[0], 'Recency'] < prof.loc[low_c[1], 'Recency']:
            seg_names[low_c[0]] = 'Новые'
            seg_names[low_c[1]] = 'Разовые / спящие'
        else:
            seg_names[low_c[1]] = 'Новые'
            seg_names[low_c[0]] = 'Разовые / спящие'

        rfm['Сегмент'] = rfm['cluster'].map(seg_names)
        prof['Сегмент'] = prof.index.map(seg_names)
        prof['Доля клиентов, %'] = (prof['n'] / prof['n'].sum() * 100).round(1)
        prof['Доля выручки, %'] = (prof['Total_Rev'] / prof['Total_Rev'].sum() * 100).round(1)

        col_rfm_1, col_rfm_2 = st.columns([3, 2])
        with col_rfm_1:
            fig_rfm = px.scatter(
                rfm, x='Frequency', y='Monetary', color='Сегмент',
                log_x=True, log_y=True,
                title="Карта сегментов: Frequency vs Monetary (лог. шкалы)",
                labels={'Frequency': 'Число заказов (log)', 'Monetary': 'Выручка, £ (log)'},
                hover_data=['Customer ID', 'Recency'],
                color_discrete_map={
                    'VIP': '#1f77b4',
                    'Активные': '#d62728',
                    'Новые': '#2ca02c',
                    'Разовые / спящие': '#ff7f0e'
                }
            )
            st.plotly_chart(fig_rfm, use_container_width=True)

        with col_rfm_2:
            st.markdown("#### Профиль сегментов")
            disp_prof = prof[['Сегмент', 'n', 'Доля клиентов, %', 'Доля выручки, %', 'Recency', 'Frequency', 'Monetary']].copy()
            disp_prof = disp_prof.sort_values('Monetary', ascending=False).round({'Recency': 1, 'Frequency': 1, 'Monetary': 0})
            st.dataframe(disp_prof, hide_index=True, use_container_width=True)
            
            fig_pie = px.pie(
                disp_prof, values='Доля выручки, %', names='Сегмент',
                title="Вклад сегментов в общую выручку",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Недостаточно клиентов для построения 4 кластеров при текущих фильтрах.")

# ------------------------------------------------------------
# ВКЛАДКА 3: ML-МОДЕЛИ И ВИТРИНА ДАННЫХ
# ------------------------------------------------------------
with tab3:
    st.markdown("### Результаты машинного обучения (п.2.4)")
    col_ml1, col_ml2 = st.columns(2)

    with col_ml1:
        st.markdown("#### ML-1: Классификация тональности (Amazon Reviews)")
        st.caption("Признаки: TF-IDF (10 000 слов) → TruncatedSVD (100 компонент), сэмпл 50 000 отзывов.")
        ml1_df = pd.DataFrame([
            {'Модель': 'LogisticRegression', 'Accuracy': 0.675, 'F1-macro': 0.532, 'F1-weighted': 0.721, 'Обучение, сек': 1.89},
            {'Модель': 'LinearSVC (лучшая)', 'Accuracy': 0.789, 'F1-macro': 0.549, 'F1-weighted': 0.787, 'Обучение, сек': 1.34},
            {'Модель': 'RandomForest', 'Accuracy': 0.742, 'F1-macro': 0.467, 'F1-weighted': 0.730, 'Обучение, сек': 5.52},
        ])
        st.dataframe(ml1_df, hide_index=True, use_container_width=True)
        st.success("**Выбор для продакшена:** `LinearSVC` — наивысший F1-macro (**0.549**) и минимальное время обучения (**1.34 сек**).")

    with col_ml2:
        st.markdown("#### ML-2: Прогноз недельных продаж (Walmart Sales)")
        st.caption("Признаки: Store (one-hot), Month, Holiday_Flag, Temperature, Unemployment (без утечки CPI/Fuel_Price).")
        ml2_df = pd.DataFrame([
            {'Модель': 'LinearRegression (лучшая)', 'R²': 0.952, 'RMSE, $': 117664, 'MAE, $': 86193, 'Обучение, сек': 0.23},
            {'Модель': 'RandomForestRegressor', 'R²': 0.941, 'RMSE, $': 130288, 'MAE, $': 80563, 'Обучение, сек': 1.41},
        ])
        st.dataframe(ml2_df, hide_index=True, use_container_width=True)
        st.info("**Вывод:** `LinearRegression` достигает **R² = 0.952**, так как зависимость продаж от магазина и месяца носит мультипликативный линейный характер.")

    st.divider()
    st.markdown("### Интегрированная витрина `fact_monthly.parquet` (п.2.3)")
    st.caption("Слияние по оси «Время» (13 общих месяцев 2010-12 — 2011-12): Retail (UK) + Walmart (US) + Росстат XML (РФ).")
    
    fig_fact = go.Figure()
    fig_fact.add_trace(go.Scatter(
        x=fact['YearMonth'], y=fact['retail_revenue'],
        name='Retail выручка (£)', yaxis='y1', mode='lines+markers', line=dict(color='steelblue', width=2.5)
    ))
    fig_fact.add_trace(go.Scatter(
        x=fact['YearMonth'], y=fact['walmart_sales'],
        name='Walmart продажи ($)', yaxis='y2', mode='lines+markers', line=dict(color='coral', width=2.5)
    ))
    fig_fact.update_layout(
        title="Совместная динамика Retail и Walmart (2010-12 — 2011-12)",
        yaxis=dict(title="Retail, £", side="left"),
        yaxis2=dict(title="Walmart, $", side="right", overlaying="y"),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig_fact, use_container_width=True)
    with st.expander("Показать таблицу fact_monthly"):
        st.dataframe(fact, use_container_width=True)

# ------------------------------------------------------------
# ВКЛАДКА 4: БЕНЧМАРКИ ПРОИЗВОДИТЕЛЬНОСТИ (п.2.5)
# ------------------------------------------------------------
with tab4:
    st.markdown("### Сравнительная оценка методов с замерами времени (п.2.5)")
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        load_bench = pd.DataFrame([
            {'Формат / Метод': 'XLSX (openpyxl, без кэша)', 'Время, сек': 113.07},
            {'Формат / Метод': 'XLSX (openpyxl, с кэшем ОС)', 'Время, сек': 27.36},
            {'Формат / Метод': 'CSV (pandas.read_csv)', 'Время, сек': 0.34},
            {'Формат / Метод': 'Parquet (pandas.read_parquet)', 'Время, сек': 0.11},
        ])
        fig_b1 = px.bar(
            load_bench, x='Формат / Метод', y='Время, сек',
            title="Скорость загрузки Retail (541 910 строк)",
            color='Формат / Метод', text_auto='.2f'
        )
        fig_b1.update_layout(showlegend=False)
        st.plotly_chart(fig_b1, use_container_width=True)

    with col_b2:
        agg_bench = pd.DataFrame([
            {'Метод агрегации': 'pandas groupby (in-memory)', 'Время, сек': 0.32},
            {'Метод агрегации': 'SQLite (только SQL-запрос)', 'Время, сек': 0.54},
            {'Метод агрегации': 'SQLite (загрузка в БД + запрос)', 'Время, сек': 2.05},
        ])
        fig_b2 = px.bar(
            agg_bench, x='Метод агрегации', y='Время, сек',
            title="Скорость месячной агрегации (pandas vs SQLite)",
            color='Метод агрегации', text_auto='.2f'
        )
        fig_b2.update_layout(showlegend=False)
        st.plotly_chart(fig_b2, use_container_width=True)

    st.success(
        "**Итоговая архитектура продакшена (п.2.5):** **Parquet + pandas + LinearSVC**.\n"
        "Переход с XLSX на Parquet ускорил чтение данных в **244 раза** (с 27.36 сек до 0.11 сек) и устранил главное узкое место пайплайна."
    )

st.caption("Данные: UCI Online Retail II, Walmart Sales, Amazon Reviews, ЕМИСС/Росстат (2010–2012). Выполнено по п.2.2–2.6.")
