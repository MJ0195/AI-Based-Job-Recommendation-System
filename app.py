import streamlit as st
import pandas as pd

from recommender import recommend_jobs

st.set_page_config(
    page_title="AI Job Recommender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Overall background */
    .stApp { background-color: #f0f4f8; }

    /* Main title */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1a3c5e;
        text-align: center;
        padding: 10px 0 4px 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #5a7a9a;
        text-align: center;
        margin-bottom: 24px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1a3c5e;
        margin-bottom: 6px;
    }

    /* Job card */
    .job-card {
        background: white;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .job-card.top-match {
        border-left-color: #16a34a;
        background: #f0fdf4;
    }
    .job-title { font-size: 1.1rem; font-weight: 700; color: #1e293b; }
    .job-company { color: #475569; font-size: 0.9rem; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 4px 2px 0;
    }
    .badge-blue   { background:#dbeafe; color:#1d4ed8; }
    .badge-green  { background:#dcfce7; color:#15803d; }
    .badge-purple { background:#ede9fe; color:#6d28d9; }
    .badge-orange { background:#ffedd5; color:#c2410c; }
    .score-bar-label { font-size:0.8rem; color:#64748b; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #1a3c5e !important; }
    section[data-testid="stSidebar"] * { color: white !important; }
    section[data-testid="stSidebar"] .stSelectbox * { color: #0f172a !important; }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[role="button"],
    section[data-testid="stSidebar"] .stSelectbox div[role="button"] *,
    section[data-testid="stSidebar"] .stSelectbox div[role="button"] span,
    section[data-testid="stSidebar"] .stSelectbox div[role="button"] div,
    section[data-testid="stSidebar"] .stSelectbox div[role="button"] input,
    section[data-testid="stSidebar"] .stSelectbox div[role="button"] span > div {
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[role="button"] {
        background-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[role="listbox"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Button */
    div.stButton > button {
        background: linear-gradient(135deg, #2563eb, #1e40af);
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 0.55rem 2rem;
        font-size: 1rem;
        border: none;
        width: 100%;
        transition: 0.2s;
    }
    div.stButton > button:hover { opacity: 0.88; }

    /* Divider */
    hr { border: 1px solid #e2e8f0; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

SKILL_SUGGESTIONS = [
    "Python", "SQL", "Machine Learning", "Data Analysis", "Java",
    "JavaScript", "React", "Node.js", "AWS", "Azure", "Docker",
    "Kubernetes", "TensorFlow", "Pandas", "Excel", "Tableau",
    "Communication", "Project Management", "Leadership", "Agile",
    "Financial Modeling", "Risk Analysis", "Marketing", "SEO",
    "Content Writing", "Google Ads", "Nursing", "Patient Care",
    "Medical Research", "Pharmaceuticals", "Production Planning",
    "Supply Chain", "Accounting", "Customer Service"
]

EXPERIENCE_OPTIONS = ["Any Level", "Entry Level", "Mid Level", "Senior Level"]
INDUSTRY_OPTIONS   = ["Any Industry", "Software", "Healthcare", "Finance",
                      "Marketing", "Manufacturing", "Retail", "Education"]
LOCATION_OPTIONS   = ["Any Location", "Sydney", "San Francisco", "New York", 
                      "Berlin", "London", "Bangalore", "Toronto"]

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown(
        '<div style="color:#ffffff; font-size:1rem; font-weight:700; margin-bottom:6px;">🎓 Experience Level</div>',
        unsafe_allow_html=True
    )
    experience_filter = st.selectbox(
        "",
        options=EXPERIENCE_OPTIONS,
        index=0,
        label_visibility="collapsed",
        help="Filter recommendations by your experience level"
    )

    st.markdown(
        '<div style="color:#ffffff; font-size:1rem; font-weight:700; margin-bottom:6px;">🏭 Preferred Industry</div>',
        unsafe_allow_html=True
    )
    industry_filter = st.selectbox(
        "",
        options=INDUSTRY_OPTIONS,
        index=0,
        label_visibility="collapsed",
        help="Optionally focus on a specific industry"
    )

    st.markdown(
        '<div style="color:#ffffff; font-size:1rem; font-weight:700; margin-bottom:6px;">📍 Preferred Location</div>',
        unsafe_allow_html=True
    )
    location_filter = st.selectbox(
        "",
        options=LOCATION_OPTIONS,
        index=0,
        label_visibility="collapsed",
        help="Optionally focus on a specific location"
    )

    top_n = st.slider(
        "📋 Number of Results",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )

    st.markdown("---")
    alpha = 0.6
    beta  = 0.4

    st.markdown("#### 💡 About")
    st.info(
        "This system uses a **Hybrid Recommendation Engine**:\n\n"
        "• **TF-IDF + Cosine Similarity** for skill matching\n"
        "• **Random Forest** ML model for relevance prediction\n"
        "• Both scores are combined into a final **Hybrid Score**"
    )


st.markdown('<div class="main-title"> AI-Powered Job Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Find jobs that match your skills using Machine Learning & Cosine Similarity</div>', unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="section-header">✍️ Enter Your Skills</div>', unsafe_allow_html=True)
    user_skills_text = st.text_area(
        label="Skills (comma-separated)",
        placeholder="e.g.  Python, SQL, Machine Learning, Data Analysis",
        height=110,
        label_visibility="collapsed"
    )

with col_right:
    st.markdown('<div class="section-header">🔖 Quick Skill Picker</div>', unsafe_allow_html=True)
    selected_suggestions = st.multiselect(
        label="Select skills to add",
        options=SKILL_SUGGESTIONS,
        default=[],
        label_visibility="collapsed",
        placeholder="Choose from common skills..."
    )
    if selected_suggestions:
        st.caption("Selected skills will be merged with your text input above.")

# Merge typed text + multi-select picks
all_skills = user_skills_text.strip()
if selected_suggestions:
    extra = ", ".join(selected_suggestions)
    all_skills = (all_skills.rstrip(", ") + ", " + extra).strip(", ") if all_skills else extra

# Display the merged skill string
if all_skills:
    st.info(f"📌 **Skills to match:** {all_skills}")

st.markdown("")

# Centered button
btn_col1, btn_col2, btn_col3 = st.columns([1.5, 2, 1.5])
with btn_col2:
    search_clicked = st.button("🔍 Get Recommendations", use_container_width=True)

st.markdown("---")

if search_clicked:
    if not all_skills:
        st.warning("⚠️  Please enter at least one skill before searching.")
        st.stop()

    # Resolve filter values
    exp_val = "" if experience_filter == "Any Level"    else experience_filter
    ind_val = "" if industry_filter   == "Any Industry" else industry_filter
    loc_val = "" if location_filter   == "Any Location" else location_filter

    with st.spinner("🤖 Analysing skills and finding the best job matches..."):
        try:
            results = recommend_jobs(
                user_skills=all_skills,
                industry_preference=ind_val,
                experience_level=exp_val,
                location_preference=loc_val,
                top_n=top_n,
                alpha=alpha,
                beta=beta
            )
        except FileNotFoundError:
            st.error(
                "⚠️  Model files not found! Please run the setup steps first:\n\n"
                "```\npython data_preprocessing.py\npython model.py\n```"
            )
            st.stop()

    if results.empty:
        st.warning("😕 No matching jobs found. Try different skills or remove filters.")
        st.stop()

    # ---- Summary banner ----
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🎯 Jobs Found",       len(results))
    col_b.metric("⭐ Top Match Score",  f"{results['Hybrid_Score'].iloc[0]:.1%}")
    col_c.metric("📊 Avg Similarity",   f"{results['Similarity_Score'].mean():.1%}")

    # Display applied filters
    filters_applied = []
    if exp_val: filters_applied.append(f"Experience: {exp_val}")
    if ind_val: filters_applied.append(f"Industry: {ind_val}")
    if loc_val: filters_applied.append(f"Location: {loc_val}")
    if filters_applied:
        st.info(f"🔍 **Applied Filters:** {', '.join(filters_applied)}")
    else:
        st.info("🔍 **Applied Filters:** None (showing all jobs)")

    st.markdown("---")

    # ---- Toggle between Card view and Table view ----
    view_mode = st.radio(
        "Display as:", ["🃏 Cards", "📋 Table"],
        horizontal=True, label_visibility="collapsed"
    )

    st.markdown("")

    if view_mode == "🃏 Cards":
        st.markdown(f"### 🏆 Top {len(results)} Job Recommendations")

        for rank, row in results.iterrows():
            is_top = rank <= 3
            card_class = "job-card top-match" if is_top else "job-card"
            top_badge  = "🥇 Top Match  " if is_top else ""

            sim_pct    = int(row["Similarity_Score"] * 100)
            hybrid_pct = int(row["Hybrid_Score"]     * 100)

            salary_str = f"${int(row['Salary']):,}" if pd.notna(row["Salary"]) else "N/A"

            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div class="job-title">#{rank}  {top_badge}{row['Job Title']}</div>
                        <div class="job-company">🏢 {row['Company']}  &nbsp;|&nbsp;  📍 {row['Location']}</div>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:1.3rem; font-weight:800; color:#2563eb;">{hybrid_pct}%</span><br>
                        <span style="font-size:0.75rem; color:#94a3b8;">Hybrid Score</span>
                    </div>
                </div>
                <div style="margin:10px 0 6px 0;">
                    <span class="badge badge-blue">💰 {salary_str}</span>
                    <span class="badge badge-purple">🎓 {row['Experience Level']}</span>
                    <span class="badge badge-orange">🏭 {row['Industry']}</span>
                </div>
                <div style="font-size:0.85rem; color:#475569; margin-top:6px;">
                    🛠️ <b>Skills:</b> {row['Required Skills']}
                </div>
                <div style="margin-top:8px; font-size:0.8rem; color:#64748b;">
                    Skill Similarity: <b>{sim_pct}%</b> &nbsp;|&nbsp; Hybrid Score: <b>{hybrid_pct}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown(f"### 📋 Top {len(results)} Job Recommendations")

        display_df = results.copy()
        display_df["Salary"]           = display_df["Salary"].apply(lambda x: f"${int(x):,}" if pd.notna(x) else "N/A")
        display_df["Similarity_Score"] = display_df["Similarity_Score"].apply(lambda x: f"{x:.1%}")
        display_df["Hybrid_Score"]     = display_df["Hybrid_Score"].apply(lambda x: f"{x:.1%}")
        display_df.index.name = "Rank"

        st.dataframe(
            display_df,
            use_container_width=True,
            height=420
        )

        # Download button
        csv_data = results.to_csv(index_label="Rank")
        st.download_button(
            label="⬇️  Download Results as CSV",
            data=csv_data,
            file_name="job_recommendations.csv",
            mime="text/csv"
        )

    st.success(f"✅ Done! Showing top {len(results)} jobs for skills: **{all_skills}**")


else:
    # Show a welcome / how-to-use section when nothing is searched
    st.markdown("### 🚀 How to Use")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **Step 1 — Enter Skills**
        Type your skills in the text box (comma-separated), or pick from the Quick Skill Picker.
        """)
    with c2:
        st.markdown("""
        **Step 2 — Set Filters** _(optional)_
        Use the sidebar to filter by experience level or industry.
        """)
    with c3:
        st.markdown("""
        **Step 3 — Get Results**
        Click **Get Recommendations** to see your personalised job matches ranked by a Hybrid Score.
        """)

st.markdown("---")
st.caption("🤖 Powered by TF-IDF Cosine Similarity + Random Forest · Built with Streamlit")