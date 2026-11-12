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
st.set_page_config(page_title="ТОБД 2.6 — Дашборд", layout="wide")

st.title("Анализ потребительских предпочтений")
st.markdown("Интерактивный дашборд по результатам пунктов 2.3–2.4: гипотезы H1–H5, RFM-сегментация, ключевые метрики.")

# ============================================================
# 2. ЗАГРУЗКА ДАННЫХ (с кэшированием)
# ============================================================
@st.cache_data
def load_data():
    # Основные данные для графиков (из 2.3/2.4)
    fact = pd.read_parquet('fact_monthly.parquet')
    retail = pd.read_parquet('retail_temp.parquet')  # или 'retail_clean.parquet', если есть
    return fact, retail

fact, retail = load_data()

# Приводим типы (на всякий случай)
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])
retail['Revenue'] = retail['Quantity'] * retail['Price']
retail['YearMonth'] = retail['InvoiceDate'].dt.to_period('M').astype(str)
retail['Month'] = retail['InvoiceDate'].dt.month
retail['DayOfWeek'] = retail['InvoiceDate'].dt.dayofweek
retail['Hour'] = retail['InvoiceDate'].dt.hour

# ============================================================
# 3. БОКОВАЯ ПАНЕЛЬ (ФИЛЬТРЫ)
# ============================================================
st.sidebar.header("Фильтры")

# Фильтр по месяцам (для H1, H4)
all_months = sorted(retail['Month'].unique())
selected_months = st.sidebar.multiselect(
    "Месяцы",
    options=all_months,
    default=all_months,
    format_func=lambda x: f"{x:02d}"
)

# Фильтр по категориям (для H4)
# (категоризация как в 2.4)
def assign_category(desc):
    if pd.isna(desc): return 'Other'
    d = str(desc).lower()
    if any(w in d for w in ['christmas','xmas','santa']): return 'Christmas'
    if any(w in d for w in ['garden','flower','plant']): return 'Garden'
    if any(w in d for w in ['kitchen','bowl','cup']): return 'Kitchen'
    if any(w in d for w in ['bag','purse']): return 'Bags'
    if any(w in d for w in ['toy','game']): return 'Toys'
    if any(w in d for w in ['light','lamp','candle']): return 'Lighting'
    return 'Other'

retail['Category'] = retail['Description'].apply(assign_category)
all_categories = sorted(retail['Category'].unique())
selected_categories = st.sidebar.multiselect(
    "Категории товаров",
    options=all_categories,
    default=all_categories
)

# Применяем фильтры к retail
retail_f = retail[
    (retail['Month'].isin(selected_months)) &
    (retail['Category'].isin(selected_categories))
].copy()

# ============================================================
# 4. KPI-КАРТОЧКИ (СВЕРХУ)
# ============================================================
st.subheader("Ключевые показатели (по отфильтрованным данным)")

# Считаем на лету (или можно захардкодить из 2.4)
total_revenue = retail_f['Revenue'].sum()
n_customers = retail_f['Customer ID'].nunique()
n_orders = retail_f['Invoice'].nunique()
avg_check = total_revenue / n_orders if n_orders > 0 else 0

# Gini (упрощённо, прямая формула)
cust_rev = retail_f.groupby('Customer ID')['Revenue'].sum().values
if len(cust_rev) > 1:
    x = np.sort(cust_rev)
    n = len(x)
    gini = (2 * np.sum(np.arange(1, n+1) * x)) / (n * x.sum()) - (n+1)/n
else:
    gini = 0

# Доля однократных
freq = retail_f.groupby('Customer ID')['Invoice'].nunique()
share1 = (freq == 1).mean() if len(freq) > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Выручка", f"{total_revenue:,.0f} £")
col2.metric("Gini", f"{gini:.3f}", help="Концентрация выручки (0 = равенство, 1 = всё у одного)")
col3.metric("Доля однократных", f"{share1*100:.1f}%")
col4.metric("Средний чек", f"{avg_check:,.0f} £")

st.divider()

# ============================================================
# 5. ГРАФИКИ (5 РАЗРЕЗОВ)
# ============================================================

# --- H1: Сезонность (месяцы) ---
st.subheader("H1: Сезонность выручки по месяцам")
h1_data = retail_f.groupby('Month')['Revenue'].sum().reset_index()
h1_data = h1_data.sort_values('Month')

fig_h1 = px.bar(
    h1_data, x='Month', y='Revenue',
    title="Выручка по месяцам (Retail)",
    labels={'Month': 'Месяц', 'Revenue': 'Выручка, £'},
    color='Revenue', color_continuous_scale='Blues',
    text_auto='.2s'
)

fig_h1.update_xaxes(type='category') 

fig_h1.update_layout(bargap=0.3) # Делаем зазор, чтобы столбец не был слишком толстым
st.plotly_chart(fig_h1, use_container_width=True)

# --- H2: Кривая Лоренца (Парето) ---
st.subheader("H2: Правило Парето (20/80)")
cust_sorted = np.sort(retail_f.groupby('Customer ID')['Revenue'].sum().values)[::-1]
cum = np.cumsum(cust_sorted) / cust_sorted.sum()
n = len(cust_sorted)
x_lorenz = np.concatenate(([0.0], np.arange(1, n+1)/n))
y_lorenz = np.concatenate(([0.0], cum))

fig_h2 = go.Figure()
fig_h2.add_trace(go.Scatter(x=x_lorenz, y=y_lorenz, mode='lines', name='Кривая Лоренца', line=dict(color='steelblue', width=2)))
fig_h2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Равномерное', line=dict(dash='dash', color='gray')))
fig_h2.add_hline(y=0.8, line_dash="dot", line_color="red", annotation_text="80% выручки")
fig_h2.update_layout(title=f"Кривая Лоренца (Gini = {gini:.3f})", xaxis_title="Доля клиентов", yaxis_title="Доля выручки")
st.plotly_chart(fig_h2, use_container_width=True)

# --- H3: Средний чек по дням/часам ---
st.subheader("H3: Средний чек по дням недели и часам")
inv = retail_f.groupby('Invoice').agg(
    revenue=('Revenue', 'sum'),
    dow=('DayOfWeek', 'first'),
    hour=('Hour', 'first')
).reset_index()

col_h3_1, col_h3_2 = st.columns(2)
with col_h3_1:
    dow_avg = inv.groupby('dow')['revenue'].mean().reset_index()
    dow_avg['day_name'] = dow_avg['dow'].map({0:'Пн',1:'Вт',2:'Ср',3:'Чт',4:'Пт',5:'Сб',6:'Вс'})
    fig_h3_dow = px.bar(dow_avg, x='day_name', y='revenue', title="Средний чек по дням", color_discrete_sequence=['teal'])
    st.plotly_chart(fig_h3_dow, use_container_width=True)

with col_h3_2:
    hour_avg = inv.groupby('hour')['revenue'].mean().reset_index()
    fig_h3_hour = px.bar(hour_avg, x='hour', y='revenue', title="Средний чек по часам", color_discrete_sequence=['coral'])
    st.plotly_chart(fig_h3_hour, use_container_width=True)

# --- H4: Сезонность категорий ---
st.subheader("H4: Сезонность категорий товаров")
pivot = retail_f.pivot_table(index='Category', columns='Month', values='Revenue', aggfunc='sum', fill_value=0)
pivot_share = pivot.div(pivot.sum(axis=1), axis=0)

fig_h4 = px.imshow(
    pivot_share,
    aspect='auto',
    title="Доля месяца в годовой выручке категории",
    labels=dict(x="Месяц", y="Категория", color="Доля"),
    color_continuous_scale='YlOrRd'
)
st.plotly_chart(fig_h4, use_container_width=True)

# --- H5: Распределение частоты ---
st.subheader("H5: Распределение числа заказов на клиента")
freq_hist = freq.value_counts().sort_index().reset_index()
freq_hist.columns = ['orders', 'count']
fig_h5 = px.bar(
    freq_hist[freq_hist['orders'] <= 20],  # обрезаем хвост для читаемости
    x='orders', y='count',
    title="Сколько клиентов сделали N заказов",
    labels={'orders': 'Число заказов', 'count': 'Клиентов'},
    color_discrete_sequence=['steelblue']
)
st.plotly_chart(fig_h5, use_container_width=True)

st.divider()

# ============================================================
# 6. RFM-КЛАСТЕРЫ (ДОПОЛНИТЕЛЬНО)
# ============================================================
st.subheader("RFM-сегментация (KMeans, k=4)")

# Готовим RFM на отфильтрованных данных
snap = retail_f['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = retail_f.groupby('Customer ID').agg(
    Recency=('InvoiceDate', lambda s: (snap - s.max()).days),
    Frequency=('Invoice', 'nunique'),
    Monetary=('Revenue', 'sum')
).reset_index()
rfm = rfm.dropna(subset=['Recency', 'Frequency', 'Monetary'])
rfm = rfm[(rfm['Recency'] >= 0) & (rfm['Frequency'] > 0) & (rfm['Monetary'] > 0)]

if len(rfm) >= 4:
    X = rfm[['Recency', 'Frequency', 'Monetary']].values
    X_log = np.log1p(X)
    X_scaled = StandardScaler().fit_transform(X_log)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['cluster'] = km.fit_predict(X_scaled)

    # Названия кластеров (по среднему Monetary)
    cluster_means = rfm.groupby('cluster')['Monetary'].mean().sort_values(ascending=False)
    cluster_names = {c: name for c, name in zip(cluster_means.index, ['VIP', 'Активные', 'Новые', 'Разовые'])}
    rfm['segment'] = rfm['cluster'].map(cluster_names)

    fig_rfm = px.scatter(
        rfm, x='Frequency', y='Monetary', color='segment',
        title="RFM-кластеры (Frequency vs Monetary)",
        labels={'Frequency': 'Число заказов', 'Monetary': 'Выручка, £'},
        hover_data=['Recency']
    )
    st.plotly_chart(fig_rfm, use_container_width=True)

st.caption("Данные: Retail (2010–2011). Дашборд построен по результатам п.2.4.")
