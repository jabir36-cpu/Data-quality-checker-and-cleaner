import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title='Data Quality Checker & Cleaner', layout='wide')

st.markdown("""
# 📊 Data Quality Checker & Cleaner
**Steps to use this app:**
1. Upload your CSV or Excel file.
2. Select cleaning options in the sidebar.
3. Explore your data quality score and analytics.
4. Download the cleaned data.
""")

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

st.sidebar.header('⚙ Controls')
uploaded_file = st.sidebar.file_uploader('Upload CSV or Excel', type=['csv', 'xlsx'])

if uploaded_file:
    df_raw = load_data(uploaded_file)

    if df_raw.empty:
        st.warning("The uploaded file is empty. Please upload a valid CSV/Excel file.")
        st.stop()

    st.success('File Uploaded Successfully ✅')

    # Save original for before/after comparison
    original_rows = df_raw.shape[0]
    original_missing = (df_raw.isnull().sum().sum() / (df_raw.shape[0] * df_raw.shape[1])) * 100

    df = df_raw.copy()

    # ==========================
    # SIDEBAR - DROP COLUMNS
    # ==========================
    st.sidebar.subheader('🗑 Drop Columns')
    cols_to_drop = st.sidebar.multiselect(
        'Select columns to remove',
        options=df.columns.tolist()
    )
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # ==========================
    # SIDEBAR - RENAME COLUMNS
    # ==========================
    st.sidebar.subheader('✏️ Rename a Column')
    col_to_rename = st.sidebar.selectbox('Select column to rename', ['None'] + df.columns.tolist())
    if col_to_rename != 'None':
        new_col_name = st.sidebar.text_input(f'New name for "{col_to_rename}"')
        if new_col_name and new_col_name != col_to_rename:
            df = df.rename(columns={col_to_rename: new_col_name})
            st.sidebar.success(f'Renamed to "{new_col_name}" ✅')

    # ==========================
    # SIDEBAR - CLEANING OPTIONS
    # ==========================
    st.sidebar.subheader('🧹 Data Cleaning')
    remove_dup = st.sidebar.checkbox('Remove Duplicate Rows')
    drop_na = st.sidebar.checkbox('Drop Rows with Missing Values')
    fill_option = st.sidebar.selectbox(
        'Fill Missing Values',
        ['None', 'Mean (numeric)', 'Median (numeric)', 'Mode (any)']
    )

    if remove_dup:
        df = df.drop_duplicates()
    if drop_na:
        df = df.dropna()
    if fill_option != 'None':
        num_cols = df.select_dtypes(include=np.number).columns
        if fill_option == 'Mean (numeric)':
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        elif fill_option == 'Median (numeric)':
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        elif fill_option == 'Mode (any)':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])

    # --- Pre-compute stats ---
    total_cells = df.shape[0] * df.shape[1]
    missing_percent = (df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 0
    duplicate_percent = (df.duplicated().sum() / df.shape[0]) * 100 if df.shape[0] > 0 else 0
    quality_score = max(0, min(100, 100 - (missing_percent * 0.6 + duplicate_percent * 0.4)))

    sample_df = df.sample(min(3000, len(df)))
    numeric_cols = sample_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = sample_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analytics", "💾 Download"])

    # ==========================
    # TAB 1 - OVERVIEW
    # ==========================
    with tab1:

        # Before vs After Summary
        st.subheader("🔄 Before vs After Cleaning")
        before_after = pd.DataFrame({
            'Metric': ['Total Rows', 'Missing %'],
            'Before Cleaning': [original_rows, f"{original_missing:.1f}%"],
            'After Cleaning': [df.shape[0], f"{missing_percent:.1f}%"]
        })
        st.dataframe(before_after, use_container_width=True)
        rows_removed = original_rows - df.shape[0]
        if rows_removed > 0:
            st.info(f"🗑 **{rows_removed} rows removed** after cleaning.")
        else:
            st.info("✅ No rows were removed.")

        st.divider()

        # Quality Score
        st.subheader("🏆 Data Quality Score")
        st.progress(int(quality_score))
        c1, c2, c3 = st.columns(3)
        c1.metric("Quality Score", f"{quality_score:.1f}%")
        c2.metric("Missing %", f"{missing_percent:.1f}%")
        c3.metric("Duplicate %", f"{duplicate_percent:.1f}%")

        st.divider()

        # Dataset Overview
        st.subheader("🗂 Dataset Overview")
        st.write(f"**Total Rows:** {df.shape[0]}")
        st.write(f"**Total Columns:** {df.shape[1]}")

        st.write("**Missing % per Column:**")
        missing_vals = (df.isnull().mean() * 100).round(2)
        missing_table = pd.DataFrame({
            'Column': df.columns.tolist(),
            'Missing %': missing_vals.values,
            'Status': ['⚠️ High (>20%)' if v > 20 else '✅ OK' for v in missing_vals.values]
        })
        st.dataframe(missing_table, use_container_width=True)

        st.write("**Column Data Types:**")
        dtype_table = pd.DataFrame({
            'Column': df.columns.tolist(),
            'Data Type': [str(t) for t in df.dtypes.tolist()]
        })
        st.dataframe(dtype_table, use_container_width=True)

        st.divider()

        # Summary Statistics
        st.subheader("📐 Summary Statistics")
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
        else:
            st.info("No numeric columns available for summary statistics.")

        st.divider()

        # Duplicate Rows
        st.subheader("👯 Duplicate Rows")
        dup_df = df[df.duplicated(keep=False)]
        if dup_df.empty:
            st.success("✅ No duplicate rows found.")
        else:
            st.warning(f"⚠️ {dup_df.shape[0]} duplicate rows found.")
            if st.button("👁 Preview Duplicate Rows"):
                st.dataframe(dup_df, use_container_width=True)

        st.divider()

        # Preview Data with Column Search
        st.subheader("🔍 Preview Data (First 100 Rows)")
        col_search = st.text_input("🔎 Search / filter columns by name", "")
        if col_search:
            matched_cols = [c for c in df.columns if col_search.lower() in c.lower()]
            if matched_cols:
                st.dataframe(df[matched_cols].head(100), use_container_width=True)
            else:
                st.warning("No columns matched your search.")
        else:
            st.dataframe(df.head(100), use_container_width=True)

    # ==========================
    # TAB 2 - ANALYTICS
    # ==========================
    with tab2:

        with st.expander("🔗 Correlation Matrix"):
            if len(numeric_cols) >= 2:
                corr = sample_df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Viridis')
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Not enough numeric columns.")

        with st.expander("📈 Scatter Plot"):
            if len(numeric_cols) >= 2:
                x_axis = st.selectbox('X-axis', numeric_cols, key='scatter_x')
                y_axis = st.selectbox('Y-axis', numeric_cols, key='scatter_y', index=1)
                fig_scatter = px.scatter(sample_df, x=x_axis, y=y_axis)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Not enough numeric columns.")

        with st.expander("📊 Histogram"):
            if numeric_cols:
                hist_col = st.selectbox('Select Column for Histogram', numeric_cols, key='hist_col')
                bins = st.slider('Number of Bins', min_value=5, max_value=100, value=20)
                fig_hist = px.histogram(sample_df, x=hist_col, nbins=bins, color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No numeric columns found.")

        with st.expander("📦 Outlier Detection"):
            if numeric_cols:
                selected_col = st.selectbox('Select Column', numeric_cols, key='box_col')
                fig_box = px.box(sample_df, y=selected_col, points="outliers", color_discrete_sequence=['red'])
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No numeric columns found.")

        with st.expander("🏷 Categorical Analysis"):
            if cat_cols:
                selected_cat = st.selectbox('Select Column', cat_cols, key='cat_col')
                counts = sample_df[selected_cat].value_counts()
                fig_bar = px.bar(x=counts.index, y=counts.values, labels={'x': selected_cat, 'y': 'Count'})
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No categorical columns found.")

        with st.expander("🌡 Missing Values Heatmap"):
            missing_counts = df.isnull().sum()
            missing_cols_only = missing_counts[missing_counts > 0]
            if missing_cols_only.empty:
                st.success("✅ No missing values in the dataset!")
            else:
                fig_missing = px.bar(
                    x=missing_cols_only.index,
                    y=missing_cols_only.values,
                    labels={'x': 'Column', 'y': 'Missing Count'},
                    color=missing_cols_only.values,
                    color_continuous_scale='Reds',
                    title='Missing Values per Column'
                )
                st.plotly_chart(fig_missing, use_container_width=True)

    # ==========================
    # TAB 3 - DOWNLOAD
    # ==========================
    with tab3:
        st.subheader("💾 Download Cleaned Data")

        # Row count comparison
        st.info(f"📋 Original rows: **{original_rows}** → Cleaned rows: **{df.shape[0]}** ({original_rows - df.shape[0]} removed)")

        # Select columns to download
        st.write("**Select Columns to Download:**")
        selected_download_cols = st.multiselect(
            'Choose columns (leave empty to download all)',
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )
        download_df = df[selected_download_cols] if selected_download_cols else df

        st.write(f"Downloading **{download_df.shape[0]} rows** and **{download_df.shape[1]} columns**")

        buffer = BytesIO()
        if uploaded_file.name.endswith('.csv'):
            download_df.to_csv(buffer, index=False)
            st.download_button(
                "⬇️ Download CSV",
                data=buffer.getvalue(),
                file_name='cleaned_data.csv',
                mime='text/csv'
            )
        else:
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                download_df.to_excel(writer, index=False)
            st.download_button(
                "⬇️ Download Excel",
                data=buffer.getvalue(),
                file_name='cleaned_data.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

else:
    st.info("Please upload a CSV or Excel file to start analysis.")