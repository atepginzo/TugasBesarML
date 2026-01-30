# app.py
"""
Aplikasi Klasifikasi Kelayakan Air Minum
========================================
Aplikasi web untuk memprediksi kelayakan air minum
menggunakan Machine Learning (Random Forest dengan Hyperparameter Tuning).

Author: Atep Solihin - 301230038 - IF 5A
Version: 2.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# Import custom utilities
from src.utils import (
    load_artifacts,
    predict_single,
    predict_batch,
    get_feature_names,
    get_feature_ranges,
    load_dataset,
    get_dataset_stats,
    get_model_info,
    get_feature_importance,
    get_feature_statistics,
    FEATURE_INFO
)

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title="AquaCheck - Klasifikasi Kelayakan Air",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ PLOTLY DARK THEME ============
PLOTLY_TEMPLATE = "plotly_dark"
CHART_COLORS = {
    'primary': '#3B82F6',
    'secondary': '#60A5FA', 
    'success': '#10B981',
    'danger': '#EF4444',
    'warning': '#F59E0B',
    'potable': '#10B981',
    'not_potable': '#EF4444',
    'bg': '#1E293B',
    'grid': '#334155',
    'text': '#F1F5F9'
}

# ============ LOAD CUSTOM CSS ============
def load_css():
    """Load custom CSS styles."""
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ============ PLOTLY CHART HELPERS ============
def create_pie_chart(labels: list, values: list, colors: list, title: str) -> go.Figure:
    """Create a styled pie chart."""
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color='#1E293B', width=2)),
        textinfo='percent+label',
        textfont=dict(size=14, color='white'),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=CHART_COLORS['text']), x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=CHART_COLORS['text']),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        margin=dict(t=60, b=60, l=20, r=20),
        height=350
    )
    return fig


def create_bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str = None, orientation: str = 'v') -> go.Figure:
    """Create a styled bar chart."""
    if orientation == 'h':
        fig = go.Figure(go.Bar(
            x=df[y],
            y=df[x],
            orientation='h',
            marker=dict(
                color=df[y] if color else CHART_COLORS['primary'],
                colorscale='Blues' if color else None,
                line=dict(color=CHART_COLORS['primary'], width=1)
            ),
            hovertemplate=f"<b>%{{y}}</b><br>{y}: %{{x:.4f}}<extra></extra>"
        ))
    else:
        fig = go.Figure(go.Bar(
            x=df[x],
            y=df[y],
            marker=dict(
                color=CHART_COLORS['primary'],
                line=dict(color=CHART_COLORS['secondary'], width=1)
            ),
            hovertemplate=f"<b>%{{x}}</b><br>{y}: %{{y}}<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=CHART_COLORS['text']), x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=CHART_COLORS['text']),
        xaxis=dict(gridcolor=CHART_COLORS['grid'], zerolinecolor=CHART_COLORS['grid']),
        yaxis=dict(gridcolor=CHART_COLORS['grid'], zerolinecolor=CHART_COLORS['grid']),
        margin=dict(t=50, b=40, l=40, r=20),
        height=400
    )
    return fig


def create_histogram(df: pd.DataFrame, column: str, color_by: str = None) -> go.Figure:
    """Create a histogram with optional color grouping."""
    if color_by:
        fig = px.histogram(
            df, x=column, color=color_by,
            color_discrete_map={0: CHART_COLORS['danger'], 1: CHART_COLORS['success']},
            barmode='overlay',
            opacity=0.7,
            labels={color_by: 'Potability'}
        )
    else:
        fig = px.histogram(df, x=column, color_discrete_sequence=[CHART_COLORS['primary']])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=CHART_COLORS['text']),
        xaxis=dict(gridcolor=CHART_COLORS['grid']),
        yaxis=dict(gridcolor=CHART_COLORS['grid']),
        margin=dict(t=30, b=30, l=30, r=20),
        height=300,
        showlegend=True,
        legend=dict(
            title=dict(text='Status'),
            font=dict(size=11)
        )
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create an interactive correlation heatmap."""
    corr_matrix = df[get_feature_names()].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='Feature Correlation Matrix', font=dict(size=16, color=CHART_COLORS['text']), x=0.5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=CHART_COLORS['text']),
        margin=dict(t=60, b=40, l=100, r=40),
        height=500
    )
    return fig


def create_boxplot_comparison(df: pd.DataFrame, feature: str) -> go.Figure:
    """Create boxplot comparing potable vs non-potable."""
    fig = go.Figure()
    
    for potability, color, name in [(0, CHART_COLORS['danger'], 'Not Potable'), 
                                     (1, CHART_COLORS['success'], 'Potable')]:
        fig.add_trace(go.Box(
            y=df[df['Potability'] == potability][feature],
            name=name,
            marker_color=color,
            boxmean=True
        ))
    
    fig.update_layout(
        title=dict(text=f'{FEATURE_INFO[feature]["label"]} Distribution by Potability', 
                  font=dict(size=14, color=CHART_COLORS['text'])),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=CHART_COLORS['text']),
        yaxis=dict(gridcolor=CHART_COLORS['grid'], title=FEATURE_INFO[feature]['unit'] or feature),
        margin=dict(t=50, b=30, l=50, r=20),
        height=350,
        showlegend=True
    )
    return fig

# ============ HELPER FUNCTIONS ============
def render_metric_card(value: str, label: str, card_type: str = "primary") -> str:
    """Generate HTML for a metric card."""
    return f"""
    <div class="metric-card {card_type}">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
    </div>
    """

def render_alert_banner(is_potable: bool, confidence: float) -> str:
    """Generate HTML for prediction result alert banner."""
    if is_potable:
        return f"""
        <div class="alert-banner success">
            <span class="alert-icon">✓</span>
            <div class="alert-content">
                <h3>Air LAYAK MINUM</h3>
                <p>Sampel air ini memenuhi standar kualitas untuk air minum yang aman.</p>
            </div>
        </div>
        """
    else:
        return f"""
        <div class="alert-banner danger">
            <span class="alert-icon">✗</span>
            <div class="alert-content">
                <h3>Air TIDAK LAYAK MINUM</h3>
                <p>Sampel air ini tidak memenuhi standar keamanan. Perlu pengolahan sebelum dikonsumsi.</p>
            </div>
        </div>
        """

def render_confidence_badge(confidence: float) -> str:
    """Generate HTML for confidence score badge."""
    pct = confidence * 100
    if pct >= 80:
        level = "high"
    elif pct >= 60:
        level = "medium"
    else:
        level = "low"
    
    return f"""
    <div class="confidence-badge {level}">
        Keyakinan: <strong>{pct:.1f}%</strong>
    </div>
    """

def render_header():
    """Render application header."""
    st.markdown("""
    <div class="app-header">
        <h1>AquaCheck - Klasifikasi Kelayakan Air Minum</h1>
        <p>Sistem analisis kualitas air berbasis Machine Learning dengan Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Render application footer."""
    st.markdown("""
    <div class="app-footer">
        <p>© 2025 AquaCheck - Sistem Analisis Kualitas Air</p>
        <p>Tugas Besar Pembelajaran Mesin | Atep Solihin - 301230038 - IF 5A</p>
    </div>
    """, unsafe_allow_html=True)

# ============ PAGE: DASHBOARD ============
def page_dashboard():
    """Dashboard page showing dataset statistics and overview."""
    render_header()
    
    # Load dataset
    df = load_dataset()
    
    if df is None:
        st.error("Dataset 'water_potability.csv' tidak ditemukan. Pastikan file tersebut ada di direktori proyek.")
        return
    
    stats = get_dataset_stats(df)
    
    # Metric Cards Row
    st.markdown("### Ringkasan Dataset")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(render_metric_card(
            f"{stats['total_samples']:,}",
            "Total Sampel",
            "primary"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(render_metric_card(
            f"{stats['potable_count']:,}",
            f"Layak Minum ({stats['potable_pct']}%)",
            "success"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(render_metric_card(
            f"{stats['not_potable_count']:,}",
            f"Tidak Layak ({stats['not_potable_pct']}%)",
            "danger"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(render_metric_card(
            f"{stats['missing_values']:,}",
            "Nilai Kosong",
            "warning"
        ), unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Distribusi", "Korelasi", "Analisis Fitur", "Info Model"])
    
    with tab1:
        col_left, col_right = st.columns([1, 1.2])
        
        with col_left:
            # Pie chart for class distribution
            pie_fig = create_pie_chart(
                labels=['Tidak Layak', 'Layak Minum'],
                values=[stats['not_potable_count'], stats['potable_count']],
                colors=[CHART_COLORS['danger'], CHART_COLORS['success']],
                title='Distribusi Kelas Target'
            )
            st.plotly_chart(pie_fig, use_container_width=True)
        
        with col_right:
            # Feature importance (if model loaded)
            try:
                model, _, _ = load_artifacts()
                importance_df = get_feature_importance(model)
                if importance_df is not None:
                    bar_fig = create_bar_chart(
                        importance_df, 'Feature', 'Importance',
                        'Tingkat Kepentingan Fitur (Random Forest)',
                        color='Importance', orientation='h'
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)
            except:
                st.info("Muat model untuk melihat tingkat kepentingan fitur")
    
    with tab2:
        # Correlation heatmap
        st.markdown("#### Matriks Korelasi Fitur")
        st.caption("Heatmap interaktif yang menunjukkan korelasi antar parameter kualitas air")
        corr_fig = create_correlation_heatmap(df)
        st.plotly_chart(corr_fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Analisis Distribusi Fitur")
        
        # Feature selector
        selected_feature = st.selectbox(
            "Pilih Fitur untuk Dianalisis",
            options=get_feature_names(),
            format_func=lambda x: f"{FEATURE_INFO[x]['label']} ({FEATURE_INFO[x]['unit']})" if FEATURE_INFO[x]['unit'] else FEATURE_INFO[x]['label']
        )
        
        col_hist, col_box = st.columns(2)
        
        with col_hist:
            hist_fig = create_histogram(df, selected_feature, 'Potability')
            hist_fig.update_layout(title=f'Distribusi {FEATURE_INFO[selected_feature]["label"]}')
            st.plotly_chart(hist_fig, use_container_width=True)
        
        with col_box:
            box_fig = create_boxplot_comparison(df, selected_feature)
            st.plotly_chart(box_fig, use_container_width=True)
        
        # Feature statistics table
        with st.expander("Statistik Detail Fitur", expanded=False):
            stats_df = get_feature_statistics(df)
            st.dataframe(
                stats_df.style.format({
                    'Mean': '{:.2f}',
                    'Std': '{:.2f}',
                    'Min': '{:.2f}',
                    'Max': '{:.2f}',
                    'Missing_Pct': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
    
    with tab4:
        st.markdown("#### Informasi Model")
        
        try:
            model_info = get_model_info()
            model, _, _ = load_artifacts()
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown(f"""
                <div class="custom-card">
                    <h4>Detail Model</h4>
                    <p><strong>Algoritma:</strong> {model_info.get('type', 'Random Forest')}</p>
                    <p><strong>Sumber:</strong> {model_info.get('source', 'train_model.py')}</p>
                    <p><strong>N Estimators:</strong> {model_info.get('n_estimators', 'N/A')}</p>
                    <p><strong>Max Depth:</strong> {model_info.get('max_depth', 'None')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_info2:
                st.markdown("""
                <div class="custom-card">
                    <h4>Pipeline Preprocessing</h4>
                    <p><strong>1.</strong> SimpleImputer (Strategi Mean)</p>
                    <p><strong>2.</strong> SMOTE (Penyeimbangan Kelas)</p>
                    <p><strong>3.</strong> StandardScaler (Normalisasi)</p>
                    <p><strong>Pembagian:</strong> 80% Train / 20% Test</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.success("Model berhasil dimuat!")
            
        except Exception as e:
            st.error(f"Model tidak ditemukan: {str(e)}")
            st.info("Jalankan `python train_model_script.py` untuk membuat model.")
    
    # Dataset Preview
    with st.expander("Lihat Dataset Mentah", expanded=False):
        st.dataframe(
            df.head(100),
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Menampilkan 100 baris pertama dari {stats['total_samples']:,} total data.")
    
    render_footer()


# ============ PAGE: SINGLE PREDICTION ============
def page_single_prediction():
    """Single prediction page with manual input form."""
    render_header()
    
    st.markdown("### Prediksi Sampel Tunggal")
    st.markdown("Masukkan parameter kualitas air di bawah untuk memprediksi kelayakan minum.")
    
    # Load model artifacts
    try:
        model, scaler, imputer = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"""
        **File model tidak ditemukan!**
        
        {str(e)}
        
        Jalankan `python train_model_script.py` terlebih dahulu untuk melatih model.
        """)
        return
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Input Form
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("#### Parameter Kualitas Air")
        
        # Create 3 columns for inputs
        col1, col2, col3 = st.columns(3)
        
        features = get_feature_names()
        feature_info = get_feature_ranges()
        input_values = {}
        
        for i, feature in enumerate(features):
            info = feature_info[feature]
            
            # Distribute inputs across columns
            if i % 3 == 0:
                target_col = col1
            elif i % 3 == 1:
                target_col = col2
            else:
                target_col = col3
            
            with target_col:
                label_text = f"{info['label']}"
                if info['unit']:
                    label_text += f" ({info['unit']})"
                
                input_values[feature] = st.number_input(
                    label=label_text,
                    min_value=float(info['min'] * 0.5),
                    max_value=float(info['max'] * 1.5),
                    value=float(info['default']),
                    step=0.01,
                    help=info['description'],
                    key=f"input_{feature}"
                )
        
        st.markdown("")
        submitted = st.form_submit_button(
            "Analisis Sampel Air",
            use_container_width=True
        )
    
    # Process prediction
    if submitted:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        with st.spinner("Menganalisis sampel air..."):
            try:
                prediction, confidence = predict_single(
                    input_values, model, scaler, imputer
                )
                
                is_potable = prediction == 1
                
                # Results Section
                st.markdown("#### Hasil Analisis")
                
                result_col1, result_col2 = st.columns([2, 1])
                
                with result_col1:
                    st.markdown(
                        render_alert_banner(is_potable, confidence),
                        unsafe_allow_html=True
                    )
                
                with result_col2:
                    st.markdown(
                        render_confidence_badge(confidence),
                        unsafe_allow_html=True
                    )
                    st.markdown("")
                    st.markdown(f"""
                    <div style="font-size: 0.875rem; color: #94A3B8; background: #1E293B; padding: 1rem; border-radius: 8px;">
                        <strong>Kelas Prediksi:</strong> {'Layak Minum (1)' if is_potable else 'Tidak Layak (0)'}<br>
                        <strong>Waktu:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Gauge chart for confidence
                st.markdown("#### Visualisasi Keyakinan")
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence * 100,
                    title={'text': "Tingkat Keyakinan Prediksi", 'font': {'size': 20, 'color': CHART_COLORS['text']}},
                    number={'suffix': '%', 'font': {'size': 40, 'color': CHART_COLORS['text']}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': CHART_COLORS['text']},
                        'bar': {'color': CHART_COLORS['success'] if is_potable else CHART_COLORS['danger']},
                        'bgcolor': CHART_COLORS['bg'],
                        'borderwidth': 2,
                        'bordercolor': CHART_COLORS['grid'],
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.3)'},
                            {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.3)'},
                            {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': CHART_COLORS['text'], 'width': 4},
                            'thickness': 0.75,
                            'value': confidence * 100
                        }
                    }
                ))
                gauge_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': CHART_COLORS['text']},
                    height=300,
                    margin=dict(t=50, b=30, l=30, r=30)
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Show input summary
                with st.expander("Ringkasan Nilai Input", expanded=False):
                    summary_data = []
                    for k, v in input_values.items():
                        summary_data.append({
                            'Parameter': feature_info[k]['label'],
                            'Nilai': f"{v:.4f}",
                            'Satuan': feature_info[k]['unit'] if feature_info[k]['unit'] else '-'
                        })
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, hide_index=True, use_container_width=True)
                
            except Exception as e:
                st.error(f"Kesalahan prediksi: {str(e)}")
    
    render_footer()


# ============ PAGE: BATCH PREDICTION ============
def page_batch_prediction():
    """Batch prediction page with CSV upload."""
    render_header()
    
    st.markdown("### Prediksi Batch")
    st.markdown("Unggah file CSV berisi beberapa sampel air untuk analisis massal.")
    
    # Load model artifacts
    try:
        model, scaler, imputer = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"""
        **File model tidak ditemukan!**
        
        {str(e)}
        
        Jalankan `python train_model_script.py` terlebih dahulu untuk melatih model.
        """)
        return
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # File format information
    with st.expander("Format CSV yang Diperlukan", expanded=True):
        st.markdown("""
        File CSV Anda harus berisi kolom-kolom berikut (nama persis, peka huruf besar/kecil):
        """)
        
        format_df = pd.DataFrame([
            {'Kolom': 'ph', 'Deskripsi': 'Tingkat pH (0-14)', 'Contoh': '7.5'},
            {'Kolom': 'Hardness', 'Deskripsi': 'Kesadahan air (mg/L)', 'Contoh': '196.0'},
            {'Kolom': 'Solids', 'Deskripsi': 'Total padatan terlarut (ppm)', 'Contoh': '20927.0'},
            {'Kolom': 'Chloramines', 'Deskripsi': 'Kadar kloramin (ppm)', 'Contoh': '7.12'},
            {'Kolom': 'Sulfate', 'Deskripsi': 'Konsentrasi sulfat (mg/L)', 'Contoh': '333.0'},
            {'Kolom': 'Conductivity', 'Deskripsi': 'Konduktivitas listrik (μS/cm)', 'Contoh': '426.0'},
            {'Kolom': 'Organic_carbon', 'Deskripsi': 'Karbon organik (ppm)', 'Contoh': '14.28'},
            {'Kolom': 'Trihalomethanes', 'Deskripsi': 'Kadar THM (μg/L)', 'Contoh': '66.4'},
            {'Kolom': 'Turbidity', 'Deskripsi': 'Kekeruhan air (NTU)', 'Contoh': '3.97'},
        ])
        st.dataframe(format_df, hide_index=True, use_container_width=True)
        st.info("**Catatan:** Nilai kosong akan diisi otomatis menggunakan imputer yang telah dilatih.")
    
    st.markdown("")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Unggah file CSV Anda",
        type=['csv'],
        help="Pilih file CSV berisi parameter kualitas air"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            df_input = pd.read_csv(uploaded_file)
            
            st.markdown("#### Pratinjau Data")
            st.dataframe(df_input.head(10), use_container_width=True, hide_index=True)
            st.caption(f"Menampilkan 10 baris pertama dari {len(df_input)} total sampel.")
            
            # Validate columns
            required_cols = set(get_feature_names())
            uploaded_cols = set(df_input.columns)
            missing_cols = required_cols - uploaded_cols
            
            if missing_cols:
                st.error(f"Kolom yang diperlukan tidak ditemukan: **{', '.join(missing_cols)}**")
                return
            
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Predict button
            if st.button("Jalankan Prediksi Batch", use_container_width=True):
                with st.spinner(f"Memproses {len(df_input)} sampel..."):
                    try:
                        # Run prediction
                        results_df = predict_batch(df_input, model, scaler, imputer)
                        
                        st.markdown("#### Hasil Prediksi")
                        
                        # Summary metrics
                        total = len(results_df)
                        potable = (results_df['Prediction'] == 1).sum()
                        not_potable = total - potable
                        avg_confidence = results_df['Confidence'].mean() * 100
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.markdown(render_metric_card(
                                f"{total:,}", "Total Sampel", "primary"
                            ), unsafe_allow_html=True)
                        
                        with metric_col2:
                            st.markdown(render_metric_card(
                                f"{potable:,}", "Layak Minum", "success"
                            ), unsafe_allow_html=True)
                        
                        with metric_col3:
                            st.markdown(render_metric_card(
                                f"{not_potable:,}", "Tidak Layak", "danger"
                            ), unsafe_allow_html=True)
                        
                        with metric_col4:
                            st.markdown(render_metric_card(
                                f"{avg_confidence:.1f}%", "Rata-rata Keyakinan", "warning"
                            ), unsafe_allow_html=True)
                        
                        st.markdown("")
                        
                        # Results visualization
                        col_pie, col_conf = st.columns(2)
                        
                        with col_pie:
                            batch_pie = create_pie_chart(
                                labels=['Tidak Layak', 'Layak Minum'],
                                values=[not_potable, potable],
                                colors=[CHART_COLORS['danger'], CHART_COLORS['success']],
                                title='Hasil Prediksi Batch'
                            )
                            st.plotly_chart(batch_pie, use_container_width=True)
                        
                        with col_conf:
                            # Confidence distribution histogram
                            conf_fig = go.Figure(data=[go.Histogram(
                                x=results_df['Confidence'] * 100,
                                nbinsx=20,
                                marker=dict(
                                    color=CHART_COLORS['primary'],
                                    line=dict(color=CHART_COLORS['secondary'], width=1)
                                ),
                                hovertemplate='Keyakinan: %{x:.1f}%<br>Jumlah: %{y}<extra></extra>'
                            )])
                            conf_fig.update_layout(
                                title=dict(text='Distribusi Skor Keyakinan', 
                                          font=dict(size=16, color=CHART_COLORS['text']), x=0.5),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color=CHART_COLORS['text']),
                                xaxis=dict(title='Keyakinan (%)', gridcolor=CHART_COLORS['grid']),
                                yaxis=dict(title='Jumlah', gridcolor=CHART_COLORS['grid']),
                                margin=dict(t=60, b=40, l=40, r=20),
                                height=350
                            )
                            st.plotly_chart(conf_fig, use_container_width=True)
                        
                        # Results table
                        st.markdown("##### Detail Hasil")
                        
                        # Select display columns
                        display_cols = get_feature_names() + ['Potability_Label', 'Confidence_Pct']
                        st.dataframe(
                            results_df[display_cols],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'Potability_Label': st.column_config.TextColumn(
                                    'Prediction',
                                    width='medium'
                                ),
                                'Confidence_Pct': st.column_config.TextColumn(
                                    'Confidence',
                                    width='small'
                                )
                            }
                        )
                        
                        # Download button
                        st.markdown("")
                        csv_output = results_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Unduh Hasil sebagai CSV",
                            data=csv_output,
                            file_name=f"hasil_kelayakan_air_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"Kesalahan prediksi: {str(e)}")
        
        except Exception as e:
            st.error(f"Kesalahan membaca file: {str(e)}")
    
    render_footer()


# ============ SIDEBAR NAVIGATION ============
def render_sidebar():
    """Render sidebar with navigation."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h2 style="margin: 0; font-size: 1.75rem;">AquaCheck</h2>
            <p style="font-size: 0.875rem; opacity: 0.8; margin: 0.5rem 0 0 0;">Analisis Kualitas Air</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("##### Navigasi")
        
        page = st.radio(
            "Pilih Halaman",
            options=["Dashboard", "Prediksi Tunggal", "Prediksi Batch"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model Status
        st.markdown("##### Status Sistem")
        try:
            model_info = get_model_info()
            load_artifacts()
            st.success("Model Siap")
            st.caption(f"Tipe: {model_info.get('type', 'RF')}")
            st.caption(f"Sumber: {model_info.get('source', 'train_model.py')}")
        except:
            st.error("Model tidak ditemukan")
            st.caption("Jalankan train_model_script.py")
        
        st.markdown("---")
        
        # Quick Info
        st.markdown("##### Informasi")
        st.markdown("""
        <div style="font-size: 0.8rem; opacity: 0.9; line-height: 1.6;">
        <p><strong>Model:</strong> Random Forest</p>
        <p><strong>Fitur:</strong> 9 parameter</p>
        <p><strong>Output:</strong> Biner (Layak/Tidak)</p>
        <p><strong>Tuning:</strong> RandomizedSearchCV</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Author info
        st.markdown("""
        <div style="text-align: center; font-size: 0.75rem; opacity: 0.7; padding: 1rem 0;">
            <p>Dikembangkan oleh</p>
            <p><strong>Atep Solihin</strong></p>
            <p>301230038 - IF 5A</p>
        </div>
        """, unsafe_allow_html=True)
        
        return page


# ============ MAIN APPLICATION ============
def main():
    """Main application entry point."""
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Route to selected page
    if page == "Dashboard":
        page_dashboard()
    elif page == "Prediksi Tunggal":
        page_single_prediction()
    elif page == "Prediksi Batch":
        page_batch_prediction()


if __name__ == "__main__":
    main()
