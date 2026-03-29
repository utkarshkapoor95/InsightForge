# ============================================
# 📦 LIBRARIES
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

# ============================================
# 🔐 LOAD FREE GROQ API KEY
# ============================================
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================
# 🎨 PAGE SETUP
# ============================================
st.set_page_config(page_title="InsightForge", page_icon="🚀", layout="wide")

# ============================================
# 🔐 LOGIN
# ============================================
def login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if username == "admin" and password == "1234":
                st.session_state["login"] = True
                st.rerun()
            else:
                st.error("❌ Wrong credentials")

# ============================================
# 📊 MAIN APP
# ============================================
def app():
    st.title("🚀 InsightForge")
    st.caption("Upload CSV → Clean → Visualize → Train ML → Get FREE AI Insights")
    st.divider()

    # ---- FILE UPLOAD ----
    file = st.file_uploader("📂 Upload your CSV", type=["csv"])
    if file is None:
        st.info("👆 Upload a CSV file to get started")
        return

    df = pd.read_csv(file)

    # ============================================
    # TABS
    # ============================================
    t1, t2, t3, t4, t5 = st.tabs([
        "📄 Overview", "🧹 Clean Data",
        "📊 Visualize", "🤖 ML Model", "🧠 AI Insight"
    ])

    # ============================================
    # TAB 1 — DATA OVERVIEW
    # ============================================
    with t1:
        st.subheader("Dataset Overview")

        # 4 quick metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        c4.metric("Duplicates", int(df.duplicated().sum()))

        st.markdown("#### Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("#### Column Summary")
        summary = pd.DataFrame({
            "Type": df.dtypes,
            "Nulls": df.isnull().sum(),
            "Null %": (df.isnull().sum() / len(df) * 100).round(1),
            "Unique": df.nunique()
        })
        st.dataframe(summary, use_container_width=True)

        st.markdown("#### Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    # ============================================
    # TAB 2 — AUTO CLEAN
    # ============================================
    with t2:
        st.subheader("🧹 Auto Data Cleaning")

        if st.button("Run Auto Clean", use_container_width=True):
            cleaned = df.copy()
            log = []

            # Remove duplicates
            dupes = cleaned.duplicated().sum()
            if dupes:
                cleaned.drop_duplicates(inplace=True)
                log.append(f"✅ Removed {dupes} duplicate rows")

            # Fill numeric nulls with median
            for col in cleaned.select_dtypes(include=np.number).columns:
                n = cleaned[col].isnull().sum()
                if n:
                    cleaned[col].fillna(cleaned[col].median(), inplace=True)
                    log.append(f"✅ Filled {n} nulls in '{col}' with median")

            # Fill text nulls with mode
            for col in cleaned.select_dtypes(include='object').columns:
                n = cleaned[col].isnull().sum()
                if n:
                    cleaned[col].fillna(cleaned[col].mode()[0], inplace=True)
                    log.append(f"✅ Filled {n} nulls in '{col}' with mode")

            if not log:
                log.append("✅ Data is already clean — no issues found!")

            st.success("Cleaning complete!")
            for item in log:
                st.write(item)

            st.dataframe(cleaned.head(), use_container_width=True)
            st.session_state["clean_df"] = cleaned

    # Use cleaned if available
    data = st.session_state.get("clean_df", df)

    # ============================================
    # TAB 3 — VISUALIZATIONS
    # ============================================
    with t3:
        st.subheader("📊 Visualizations")
        num_cols = data.select_dtypes(include=np.number).columns.tolist()

        if not num_cols:
            st.warning("No numeric columns found")
        else:
            col1, col2 = st.columns(2)

            # Chart
            with col1:
                st.markdown("#### Column Chart")
                chosen = st.selectbox("Column", num_cols)
                chart = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Line"])
                fig, ax = plt.subplots()
                vals = data[chosen].dropna()
                if chart == "Histogram":
                    ax.hist(vals, bins=20, color="steelblue", edgecolor="white")
                elif chart == "Box Plot":
                    ax.boxplot(vals)
                else:
                    ax.plot(vals.head(100).values, color="steelblue")
                ax.set_title(f"{chart} — {chosen}")
                st.pyplot(fig)

            # Heatmap
            with col2:
                st.markdown("#### Correlation Heatmap")
                fig2, ax2 = plt.subplots()
                sns.heatmap(data[num_cols].corr(), annot=True,
                            cmap="coolwarm", fmt=".2f", ax=ax2)
                st.pyplot(fig2)

            # Category charts
            cat_cols = data.select_dtypes(include='object').columns.tolist()
            if cat_cols:
                st.markdown("#### Category Analysis")
                cat = st.selectbox("Categorical Column", cat_cols)
                fig3, ax3 = plt.subplots()
                data[cat].value_counts().head(10).plot(kind="barh", ax=ax3, color="steelblue")
                ax3.invert_yaxis()
                ax3.set_title(f"Top values in '{cat}'")
                st.pyplot(fig3)

    # ============================================
    # TAB 4 — ML MODEL
    # ============================================
    with t4:
        st.subheader("🤖 Machine Learning")
        target = st.selectbox("🎯 Target Column", data.columns)
        test_pct = st.slider("Test Size %", 10, 40, 20)

        if st.button("🚀 Train Model", use_container_width=True):
            X = data.drop(columns=[target]).select_dtypes(include=np.number)
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_pct/100, random_state=42
            )

            # Auto-detect problem type
            is_clf = y.nunique() <= 10

            if is_clf:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
                st.success(f"✅ Accuracy: {score:.2%}")
                st.info("Accuracy = % of correct predictions")
                metric_name = "Accuracy"
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                score = r2_score(y_test, model.predict(X_test))
                st.success(f"✅ R² Score: {score:.4f}")
                st.info("R² close to 1.0 = strong model")
                metric_name = "R² Score"

            # Feature importance chart
            st.markdown("#### Feature Importance")
            imp = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig4, ax4 = plt.subplots()
            ax4.barh(imp["Feature"][:8], imp["Importance"][:8], color="steelblue")
            ax4.invert_yaxis()
            ax4.set_title("Top Features by Importance")
            st.pyplot(fig4)
            st.dataframe(imp, use_container_width=True)

            # Save for AI tab
            st.session_state.update({
                "score": score,
                "metric": metric_name,
                "target": target,
                "columns": list(data.columns),
                "features": imp.to_string(),
                "problem": "Classification" if is_clf else "Regression"
            })
            st.success("✅ Done! Go to AI Insight tab.")

    # ============================================
    # TAB 5 — AI INSIGHT (FREE GROQ)
    # ============================================
    with t5:
        st.subheader("🧠 AI-Powered Insights")
        st.caption("Powered by Groq (100% Free)")

        if "score" not in st.session_state:
            st.warning("⚠️ Train a model first in the ML Model tab")
            return

        # Show model summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Problem Type", st.session_state["problem"])
        c2.metric(st.session_state["metric"], f"{st.session_state['score']:.2%}")
        c3.metric("Target", st.session_state["target"])

        insight_type = st.selectbox("What do you want to know?", [
            "📊 Full Model Analysis",
            "💼 Business Recommendations",
            "🔍 Key Feature Explanation",
            "⚠️ Model Risks & Limitations",
            "📈 How to Improve This Model"
        ])

        if st.button("🧠 Generate Insight", use_container_width=True):
            prompt = f"""
Analyze this ML project:

Dataset columns: {st.session_state['columns']}
Target variable: {st.session_state['target']}
Problem type: {st.session_state['problem']}
Model score ({st.session_state['metric']}): {st.session_state['score']:.4f}
Feature importances:
{st.session_state['features']}

Requested: {insight_type}

Give structured, business-friendly insights with:
- Model performance in plain English
- What key features mean for the business
- Specific actionable recommendations
- Risks and next steps
Keep it practical and clear.
            """

            with st.spinner("🧠 AI is thinking..."):
                try:
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",   # Free Groq model
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000
                    )
                    result = response.choices[0].message.content
                    st.success("✅ Insight Ready!")
                    st.markdown(result)
                    st.session_state["report"] = result

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Check your GROQ_API_KEY in .env file")

        # Download
        if "report" in st.session_state:
            st.divider()
            report = f"""AI ML ANALYZER REPORT
Target: {st.session_state['target']}
Score: {st.session_state['score']:.4f}
---
{st.session_state['report']}"""
            st.download_button(
                "⬇️ Download Report",
                report,
                file_name="report.txt",
                use_container_width=True
            )

# ============================================
# ▶ RUN
# ============================================
if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    login()
else:
    app()