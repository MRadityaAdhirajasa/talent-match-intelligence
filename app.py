import streamlit as st
import pandas as pd
import warnings
import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np
import os

warnings.filterwarnings('ignore')

# --- 1. MEMUAT "DATABASE" ---
@st.cache_data
def load_database():
    print("Memuat 'database' dari CSV...")
    data = {
        'competencies_yearly': pd.read_csv(os.path.join(data_folder, "Study Case DA - competencies_yearly.csv")),
        'dim_areas': pd.read_csv(os.path.join(data_folder, "Study Case DA - dim_areas.csv")),
        'dim_companies': pd.read_csv(os.path.join(data_folder, "Study Case DA - dim_companies.csv")),
        'dim_competency_pillars': pd.read_csv(os.path.join(data_folder, "Study Case DA - dim_competency_pillars.csv")),
        'dim_departments': pd.read_csv(os.path.join(data_folder, "Study Case DA - dim_departments.csv")),
        'dim_directorates': pd.read_csv(os.path.join(data_folder, "Study Case DA - dim_directorates.csv")),
        'dim_divisions': pd.read_csv(os.path.join(data_folder, "Study Case DA - dim_divisions.csv")),
        'dim_education': pd.read_csv(os.path.join(data_folder, "Study Case DA - dim_education.csv")),
        'dim_grades': pd.read_csv(os.path.join(data_folder, "Study Case DA - dim_grades.csv")),
        'dim_majors': pd.read_csv(os.path.join(data_folder, "Study Case DA - dim_majors.csv")),
        'dim_positions': pd.read_csv(os.path.join(data_folder, "Study Case DA - dim_positions.csv")),
        'employees': pd.read_csv(os.path.join(data_folder, "Study Case DA - employees.csv")),
        'papi_scores': pd.read_csv(os.path.join(data_folder, "Study Case DA - papi_scores.csv")),
        'performance_yearly': pd.read_csv(os.path.join(data_folder, "Study Case DA - performance_yearly.csv")),
        'profiles_psych': pd.read_csv(os.path.join(data_folder, "Study Case DA - profiles_psych.csv")),
        'strengths': pd.read_csv(os.path.join(data_folder, "Study Case DA - strengths.csv"))
    }
    print("Database dimuat.")
    return data

DB_DATA = load_database()

# --- 2. FUNGSI LOGIKA ---
def run_dynamic_matching(benchmark_ids, all_data):
    print("Menjalankan simulasi SQL (logika Pandas)...")
    
    # --- FASE 1 : REPLIKASI SQL ---
    competencies_yearly = all_data['competencies_yearly']
    dim_areas = all_data['dim_areas']
    dim_companies = all_data['dim_companies']
    dim_competency_pillars = all_data['dim_competency_pillars']
    dim_departments = all_data['dim_departments']
    dim_directorates = all_data['dim_directorates']
    dim_divisions = all_data['dim_divisions']
    dim_education = all_data['dim_education']
    dim_grades = all_data['dim_grades']
    dim_majors = all_data['dim_majors']
    dim_positions = all_data['dim_positions']
    employees = all_data['employees']
    papi_scores = all_data['papi_scores']
    performance_yearly = all_data['performance_yearly']
    profiles_psych = all_data['profiles_psych']
    strengths = all_data['strengths']
    
    print("Fase 1: Mereplikasi SQL...")
    performance_sorted = performance_yearly.sort_values(by=['employee_id', 'year'], ascending=[True, False])
    latest_performance = performance_sorted.drop_duplicates(subset=['employee_id'], keep='first').copy()
    latest_performance['is_high_performer'] = latest_performance['rating'].apply(lambda x: 1 if x == 5 else 0)
    latest_performance = latest_performance[['employee_id', 'year', 'rating', 'is_high_performer']]
    competencies_valid_scores = competencies_yearly[competencies_yearly['score'].between(1, 5)].copy()
    competencies_filtered = competencies_valid_scores.merge(latest_performance[['employee_id', 'year']], on=['employee_id', 'year'], how='inner')
    competencies_wide = competencies_filtered.pivot_table(index='employee_id', columns='pillar_code', values='score').reset_index()
    competencies_wide.columns = [f"comp_{col}" if col != 'employee_id' else col for col in competencies_wide.columns]
    papi_wide = papi_scores.pivot_table(index='employee_id', columns='scale_code', values='score').reset_index()
    strengths_filtered = strengths[strengths['rank'] <= 5]
    strengths_wide = strengths_filtered.pivot_table(index='employee_id', columns='rank', values='theme', aggfunc='first').reset_index()
    strengths_wide.columns = [f"strength_{col}" if isinstance(col, int) else col for col in strengths_wide.columns]
    df_analysis = employees.merge(dim_grades, on='grade_id', how='left', suffixes=('', '_dim_grade'))
    df_analysis = df_analysis.merge(dim_education, on='education_id', how='left', suffixes=('', '_dim_edu'))
    df_analysis = df_analysis.merge(dim_majors, on='major_id', how='left', suffixes=('', '_dim_major'))
    df_analysis = df_analysis.merge(dim_positions, on='position_id', how='left', suffixes=('', '_dim_pos'))
    df_analysis = df_analysis.merge(dim_departments, on='department_id', how='left', suffixes=('', '_dim_dept'))
    df_analysis = df_analysis.merge(latest_performance, on='employee_id', how='left')
    df_analysis = df_analysis.merge(profiles_psych, on='employee_id', how='left')
    df_analysis = df_analysis.merge(competencies_wide, on='employee_id', how='left')
    df_analysis = df_analysis.merge(papi_wide, on='employee_id', how='left')
    df_analysis = df_analysis.merge(strengths_wide, on='employee_id', how='left')
    df_analysis.rename(columns={'name': 'grade', 'name_dim_edu': 'education', 'name_dim_major': 'major', 'name_dim_pos': 'position', 'name_dim_dept': 'department'}, inplace=True)

    # --- FASE 2: PEMBERSIHAN ---
    print("Fase 2: Membersihkan data...")
    df_cleaned = df_analysis[df_analysis['rating'].between(1, 5)].copy()
    cols_to_force_numeric = [
        'years_of_service_months', 'pauli', 'faxtor', 'iq', 'gtq', 'tiki',
        'comp_gdr', 'comp_cex', 'comp_ids', 'comp_qdd', 'comp_sto',
        'comp_sea', 'comp_vcu', 'comp_lie', 'comp_ftc', 'comp_csi',
        'Papi_N', 'Papi_G', 'Papi_A', 'Papi_L', 'Papi_P', 'Papi_I',
        'Papi_T', 'Papi_V', 'Papi_X', 'Papi_S', 'Papi_B', 'Papi_O',
        'Papi_R', 'Papi_D', 'Papi_C', 'Papi_Z', 'Papi_E', 'Papi_K', 'Papi_F', 'Papi_W'
    ]
    df_cleaned.rename(columns={
        'comp_gdr': 'comp_GDR', 'comp_cex': 'comp_CEX', 'comp_ids': 'comp_IDS', 'comp_qdd': 'comp_QDD',
        'comp_sto': 'comp_STO', 'comp_sea': 'comp_SEA', 'comp_vcu': 'comp_VCU', 'comp_lie': 'comp_LIE',
        'comp_ftc': 'comp_FTC', 'comp_csi': 'comp_CSI'
    }, inplace=True, errors='ignore')
    for col in cols_to_force_numeric:
        if col in df_cleaned.columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col not in ['rating', 'is_high_performer']:
            median_val = df_cleaned[col].median()
            df_cleaned[col].fillna(median_val, inplace=True)
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'employee_id':
            df_cleaned[col].fillna('Unknown', inplace=True)

    # --- FASE 3: MEMBUAT FITUR TGV (Skor 0-1) ---
    print("Fase 3: Membuat Fitur TGV (Skor 0-1)...")
    tv_cols = ['iq', 'gtq', 'tiki', 'Papi_N', 'Papi_I', 'pauli', 'Papi_A', 'Papi_L', 'Papi_P', 'Papi_T', 'Papi_E', 'Papi_C', 'Papi_D']
    df_cleaned.rename(columns={
        'Papi_N': 'papi_n', 'Papi_G': 'papi_g', 'Papi_A': 'papi_a', 'Papi_L': 'papi_l', 'Papi_P': 'papi_p',
        'Papi_I': 'papi_i', 'Papi_T': 'papi_t', 'Papi_V': 'papi_v', 'Papi_O': 'papi_o', 'Papi_B': 'papi_b',
        'Papi_S': 'papi_s', 'Papi_X': 'papi_x', 'Papi_C': 'papi_c', 'Papi_D': 'papi_d', 'Papi_R': 'papi_r',
        'Papi_Z': 'papi_z', 'Papi_E': 'papi_e', 'Papi_K': 'papi_k', 'Papi_F': 'papi_f', 'Papi_W': 'papi_w'
    }, inplace=True)
    tv_cols = [col.lower() for col in tv_cols] 

    valid_tv_cols = [col for col in tv_cols if col in df_cleaned.columns]
    for col in valid_tv_cols:
        epsilon = 1e-6
        min_val = df_cleaned[col].min()
        max_val = df_cleaned[col].max()
        df_cleaned[f'{col}_norm'] = (df_cleaned[col] - min_val) / (max_val - min_val + epsilon)

    tgv_components = {
        'tgv_cognitive': ['iq_norm', 'gtq_norm', 'tiki_norm', 'papi_n_norm', 'papi_i_norm'],
        'tgv_motivation': ['pauli_norm', 'papi_a_norm'],
        'tgv_leadership': ['papi_l_norm', 'papi_p_norm'],
        'tgv_adaptability': ['papi_t_norm', 'papi_e_norm'],
        'tgv_conscientious': ['papi_c_norm', 'papi_d_norm']
    }
    for tgv, components in tgv_components.items():
        valid_components = [c for c in components if c in df_cleaned.columns]
        if valid_components:
            df_cleaned[tgv] = df_cleaned[valid_components].mean(axis=1)
    
    # --- FASE 4: Logika Weighted Average ---
    print("Fase 4: Menghitung Weighted Average Match Rate...")
    
    # Tentukan benchmark_df
    if not benchmark_ids:
        benchmark_df = df_cleaned[df_cleaned['is_high_performer'] == 1] 
    else:
        benchmark_df = df_cleaned[df_cleaned['employee_id'].isin(benchmark_ids)]
        if benchmark_df.empty:
            benchmark_df = df_cleaned[df_cleaned['is_high_performer'] == 1]
            
    # --- 4a. Hitung TV Baselines
    benchmark_profile = benchmark_df[[
        'tgv_cognitive', 'tgv_motivation', 'tgv_leadership', 
        'tgv_adaptability', 'tgv_conscientious'
    ]].median()
    
    tv_norm_cols = [f'{col}_norm' for col in valid_tv_cols]
    tv_baselines = benchmark_df[tv_norm_cols].median()
    
    # --- 4b. Hitung TV Match Rate untuk semua karyawan
    for col_norm in tv_norm_cols:
        baseline_val = tv_baselines[col_norm]
        match_col_name = col_norm.replace('_norm', '_match_rate')
        if baseline_val > 0:
            df_cleaned[match_col_name] = (df_cleaned[col_norm] / baseline_val) * 100
        else:
            df_cleaned[match_col_name] = 0
        df_cleaned[match_col_name] = df_cleaned[match_col_name].clip(0, 100) 

    # --- 4c. Hitung TGV Match Rate
    for tgv, components in tgv_components.items():
        match_rate_cols = [c.replace('_norm', '_match_rate') for c in components if c in df_cleaned.columns]
        if match_rate_cols:
            df_cleaned[f'{tgv}_match_rate'] = df_cleaned[match_rate_cols].mean(axis=1)

    # --- 4d. Hitung Strength Match Rate 
    benchmark_strengths = ['Futuristic', 'Restorative', 'Self-Assurance', 'Intellection', 'Activator']
    strength_cols = ['strength_1', 'strength_2', 'strength_3', 'strength_4', 'strength_5']
    rule_2_strength_match = df_cleaned[strength_cols].isin(benchmark_strengths).any(axis=1)
    df_cleaned['strength_match_rate'] = rule_2_strength_match.astype(int) * 100
    
    # --- 4e. Hitung Final Weighted Average
    tgv_match_cols = [f'{tgv}_match_rate' for tgv in tgv_components.keys()] + ['strength_match_rate']
    df_cleaned[tgv_match_cols] = df_cleaned[tgv_match_cols].fillna(0)

    df_cleaned['final_match_rate'] = (
        (df_cleaned['tgv_cognitive_match_rate'] * 0.25) +
        (df_cleaned['tgv_motivation_match_rate'] * 0.25) +
        (df_cleaned['tgv_leadership_match_rate'] * 0.25) +
        (df_cleaned['strength_match_rate'] * 0.15) +
        (df_cleaned['tgv_adaptability_match_rate'] * 0.05) +
        (df_cleaned['tgv_conscientious_match_rate'] * 0.05)
    )

    # 6. Kembalikan hasil dan profil benchmark
    print("Simulasi selesai.")
    results_df = df_cleaned.sort_values(by='final_match_rate', ascending=False)
    
    return results_df, benchmark_profile

# --- 3. UI SIDEBAR ---
st.sidebar.title("Buat Lowongan Pekerjaan Baru")
employee_list = DB_DATA['employees']['employee_id'].tolist()

with st.sidebar.form(key='vacancy_form'):
    role_name = st.text_input("Nama Peran (Role Name)") 
    job_level = st.text_input("Level Pekerjaan (Job Level)") 
    role_purpose = st.text_area("Tujuan Peran (Role Purpose)") 
    
    selected_ids = st.multiselect(
        "Pilih Karyawan Benchmark",
        options=employee_list,
        max_selections=3
    ) 
    
    submitted = st.form_submit_button("Generate Profil & Temukan Talenta")


# --- 4. OUTPUT ---
if submitted:

    if not role_name or not job_level or not role_purpose or not selected_ids:
        st.sidebar.error("Harap isi semua 4 kolom sebelum menekan 'Generate'.")
    
    else:
        st.header(f"Profil Talenta untuk: {role_name}")
        
        # --- 4a. Output AI-Generated Job Profile 
        st.subheader("1. AI-Generated Job Profile")
    
        try:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            
            model = genai.GenerativeModel('gemini-2.0-flash-lite') 
            
            prompt = f"""
            Anda adalah asisten Perekrutan AI.
            Buatkan profil pekerjaan profesional untuk lowongan berikut:
            
            Nama Peran: {role_name}
            Level Pekerjaan: {job_level}
            Tujuan Peran: {role_purpose}
            
            Harap hasilkan output dalam format markdown yang rapi, yang mencakup:
            1.  **Deskripsi Pekerjaan** (Job Description)
            2.  **Persyaratan Kunci** (Job Requirements)
            3.  **Kompetensi Utama** (Key Competencies)
            """
            
            with st.spinner("Menghasilkan profil pekerjaan dengan AI..."):
                response = model.generate_content(prompt)
                st.markdown(response.text)
                
        except Exception as e:
            st.error(f"Gagal memanggil Google AI: {e}")
            st.error("Pastikan Anda sudah mengatur GOOGLE_API_KEY di .streamlit/secrets.toml")
        
        # --- 4b. Output Ranked Talent List 
        st.subheader("2. Peringkat Talenta (Ranked Talent List)")
        
        with st.spinner("Mensimulasikan query SQL dan menghitung match rate..."):
            results_df, benchmark_profile = run_dynamic_matching(selected_ids, DB_DATA)
        
        results_df_display = results_df.copy()
        results_df_display['final_match_rate'] = results_df['final_match_rate'].map('{:,.2f}%'.format)
        
        display_cols = ['employee_id', 'fullname', 'final_match_rate', 'rating', 'position', 'department']
        display_cols = [col for col in display_cols if col in results_df_display.columns]
        st.dataframe(results_df_display[display_cols])

        # --- 4c. Output Dashboard Visualization
        st.subheader("3. Visualisasi Dashboard")
        
        # Visual 1: Distribusi Match Rate (Histogram)
        st.write("**Distribusi Skor Kecocokan**")

        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(results_df['final_match_rate'], bins=20, edgecolor='black')
        ax_hist.set_xlabel('Skor Kecocokan Final (%)')
        ax_hist.set_ylabel('Jumlah Karyawan')
        st.pyplot(fig_hist)
        
        # Visual 2: Perbandingan Benchmark vs Kandidat (Radar Chart)
        st.write("**Perbandingan Profil TGV (Benchmark vs. Kandidat Teratas)**")
        
        tgv_cols = ['tgv_cognitive', 'tgv_motivation', 'tgv_leadership', 'tgv_adaptability', 'tgv_conscientious']
        
        if not results_df.empty:
            
            valid_tgv_cols = [col for col in tgv_cols if col in results_df.columns]
            
            if valid_tgv_cols: 

                top_candidate_profile = results_df.iloc[0][valid_tgv_cols]
                
                labels = valid_tgv_cols
                num_vars = len(labels)
                
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                angles += angles[:1] 

                def create_radar_chart(ax, angles, data, color, label):
                    data = np.nan_to_num(data) 
                    data = np.concatenate((data, data[:1])) 
                    ax.plot(angles, data, color=color, linewidth=2, label=label)
                    ax.fill(angles, data, color=color, alpha=0.25)

                fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

                data_0 = benchmark_profile[valid_tgv_cols].values
                create_radar_chart(ax_radar, angles, data_0, 'gray', 'Profil Benchmark')

                data_1 = top_candidate_profile.values
                create_radar_chart(ax_radar, angles, data_1, 'blue', f"Kandidat Teratas ({results_df.iloc[0]['employee_id']})")

                ax_radar.set_yticklabels([])
                ax_radar.set_xticks(angles[:-1])
                ax_radar.set_xticklabels(labels)
                ax_radar.set_title('Profil TGV: Benchmark vs Kandidat Teratas', size=16, fontweight='bold', y=1.1)
                ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                
                st.pyplot(fig_radar)
            else:
                st.warning("Kolom TGV tidak ditemukan untuk membuat Radar Chart.")
        else:
            st.warning("Tidak ada kandidat yang cocok ditemukan.")
    
else:

    st.info("Silakan isi formulir di sidebar kiri untuk memulai.")

