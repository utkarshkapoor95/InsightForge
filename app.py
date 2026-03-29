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
from io import StringIO
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

# ============================================
# 🔐 GROQ CLIENT — Works on Cloud + Locally
# ============================================
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    load_dotenv()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================
# 🎨 PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="InsightForge",
    page_icon="🚀",
    layout="wide"
)

# ============================================
# 📦 SAMPLE DATA BUILT INTO APP
# ============================================
SAMPLE_DATA = """ProductID,ProductName,Category,StockLevel,DailyDemand,LeadTime,StorageCost,SupplierReliability,Discount,AdjustedDemand,Reorder
P001,Cotton Shirt,Apparel,150,45,7,2.5,0.92,0.1,50,1
P002,Denim Jeans,Apparel,80,30,10,3.2,0.88,0.05,32,1
P003,Formal Trousers,Apparel,200,20,5,2.8,0.95,0.0,20,0
P004,Winter Jacket,Apparel,60,15,14,5.5,0.85,0.15,17,1
P005,Sports Shoes,Footwear,120,35,8,4.0,0.90,0.1,38,1
P006,Casual Sneakers,Footwear,90,28,9,3.8,0.87,0.05,29,1
P007,Leather Boots,Footwear,45,12,12,6.2,0.82,0.0,12,1
P008,Sandals,Footwear,180,50,5,1.5,0.93,0.2,60,0
P009,Cotton Kurti,Ethnic,220,60,6,2.2,0.94,0.1,66,0
P010,Saree,Ethnic,100,25,8,3.5,0.89,0.05,26,0
P011,Lehenga,Ethnic,40,8,15,8.0,0.80,0.0,8,1
P012,Dupatta,Ethnic,300,80,4,1.0,0.96,0.15,92,0
P013,T-Shirt,Casuals,250,70,5,1.8,0.95,0.1,77,0
P014,Shorts,Casuals,160,40,6,1.5,0.92,0.05,42,0
P015,Polo Shirt,Casuals,110,33,7,2.0,0.91,0.0,33,0
P016,Hoodie,Casuals,75,22,10,3.0,0.86,0.1,24,1
P017,Track Pants,Sportswear,130,38,7,2.3,0.90,0.05,40,0
P018,Sports Jersey,Sportswear,95,30,8,2.8,0.88,0.1,33,1
P019,Gym Shorts,Sportswear,170,45,5,1.6,0.93,0.0,45,0
P020,Yoga Pants,Sportswear,85,27,9,2.5,0.87,0.15,31,1
P021,Blazer,Formal,55,10,12,7.0,0.83,0.0,10,1
P022,Dress Shirt,Formal,140,35,7,3.5,0.91,0.05,37,0
P023,Tie,Formal,200,15,5,0.8,0.95,0.0,15,0
P024,Formal Shoes,Footwear,65,18,11,5.5,0.84,0.0,18,1
P025,Belt,Accessories,280,55,4,0.9,0.96,0.1,60,0
P026,Wallet,Accessories,190,42,5,1.2,0.94,0.05,44,0
P027,Handbag,Accessories,70,20,10,4.5,0.86,0.0,20,1
P028,Sunglasses,Accessories,155,38,6,2.0,0.92,0.15,44,0
P029,Watch,Accessories,50,12,14,8.5,0.81,0.0,12,1
P030,Cap,Accessories,230,65,4,0.7,0.97,0.1,71,0
P031,Scarf,Accessories,175,30,6,1.1,0.93,0.05,31,0
P032,Gloves,Accessories,90,18,9,1.8,0.88,0.0,18,1
P033,Socks Pack,Innerwear,320,90,3,0.5,0.98,0.1,99,0
P034,Undershirt,Innerwear,260,70,4,0.8,0.96,0.05,73,0
P035,Thermal Set,Innerwear,80,20,10,2.5,0.87,0.0,20,1
P036,Sports Bra,Innerwear,115,32,7,1.5,0.91,0.1,35,0
P037,Nightwear Set,Innerwear,95,25,8,2.0,0.89,0.05,26,1
P038,Raincoat,Seasonal,60,15,12,4.0,0.84,0.2,18,1
P039,Sweater,Seasonal,105,28,9,3.2,0.88,0.1,31,1
P040,Shawl,Seasonal,140,35,7,2.2,0.90,0.05,37,0"""


# ============================================
# 🔐 LOGIN
# ============================================
def login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 InsightForge Login")
        st.caption("AI-Powered Data Analytics Platform")
        st.divider()
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            if username == "admin" and password == "1234":
                st.session_state["login"] = True
                st.rerun()
            else:
                st.error("❌ Wrong credentials — try admin / 1234")


# ============================================
# 📊 MAIN APP
# ============================================
def app():
    st.title("🚀 InsightForge")
    st.caption("Upload CSV → Clean → Visualize → Train ML → Get FREE AI Insights")
    st.divider()

    # ---- FILE UPLOAD + SAMPLE DATA BUTTON ----
    col1, col2 = st.columns([3, 1])

    with col1:
        file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

    with col2:
        st.markdown("#### 🎯 No file? Try:")
        if st.button("Load Sample\nInventory Data", use_container_width=True):
            st.session_state["sample_loaded"] = True
            st.session_state.pop("clean_df", None)

    # ---- DECIDE WHICH DATA TO USE ----
    if file is not None:
        df = pd.read_csv(file)
        st.session_state["sample_loaded"] = False
    elif st.session_state.get("sample_loaded"):
        df = pd.read_csv(StringIO(SAMPLE_DATA))
        st.success("✅ Sample retail inventory dataset loaded — 40 products, 11 features")
    else:
        st.info("👆 Upload a CSV file OR click 'Load Sample Inventory Data' to try a demo")
        return

    # ============================================
    # TABS
    # ============================================
    t1, t2, t3, t4, t5 = st.tabs([
        "📄 Overview",
        "🧹 Clean Data",
        "📊 Visualize",
        "🤖 ML Model",
        "🧠 AI Insight"
    ])

    # ============================================
    # TAB 1 — DATA OVERVIEW
    # ============================================
    with t1:
        st.subheader("Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        c4.metric("Duplicates", int(df.duplicated().sum()))

        st.markdown("#### 👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("#### 📋 Column Info")
        info = pd.DataFrame({
            "Type": df.dtypes,
            "Nulls": df.isnull().sum(),
            "Null %": (df.isnull().sum() / len(df) * 100).round(1),
            "Unique": df.nunique()
        })
        st.dataframe(info, use_container_width=True)

        st.markdown("#### 📊 Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        st.download_button(
            "⬇️ Download This Dataset",
            df.to_csv(index=False),
            file_name="dataset.csv",
            mime="text/csv"
        )

    # ============================================
    # TAB 2 — AUTO CLEAN
    # ============================================
    with t2:
        st.subheader("🧹 Auto Data Cleaning")
        st.info("Automatically detects and fixes missing values and duplicates")

        if st.button("▶ Run Auto Clean", use_container_width=True):
            cleaned = df.copy()
            log = []

            dupes = cleaned.duplicated().sum()
            if dupes:
                cleaned.drop_duplicates(inplace=True)
                log.append(f"✅ Removed {dupes} duplicate rows")

            for col in cleaned.select_dtypes(include=np.number).columns:
                n = cleaned[col].isnull().sum()
                if n:
                    cleaned[col].fillna(cleaned[col].median(), inplace=True)
                    log.append(f"✅ Filled {n} nulls in '{col}' with median")

            for col in cleaned.select_dtypes(include='object').columns:
                n = cleaned[col].isnull().sum()
                if n:
                    cleaned[col].fillna(cleaned[col].mode()[0], inplace=True)
                    log.append(f"✅ Filled {n} nulls in '{col}' with mode")

            if not log:
                log.append("✅ Data is already clean — no issues found!")

            c1, c2 = st.columns(2)
            c1.metric("Before", f"{df.shape[0]} rows")
            c2.metric("After", f"{cleaned.shape[0]} rows")

            for item in log:
                st.write(item)

            st.dataframe(cleaned.head(), use_container_width=True)
            st.session_state["clean_df"] = cleaned

            st.download_button(
                "⬇️ Download Cleaned Data",
                cleaned.to_csv(index=False),
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

    # Use cleaned data if available
    data = st.session_state.get("clean_df", df)

    # ============================================
    # TAB 3 — VISUALIZATIONS
    # ============================================
    with t3:
        st.subheader("📊 Visual Analysis")
        num_cols = data.select_dtypes(include=np.number).columns.tolist()

        if not num_cols:
            st.warning("No numeric columns found")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Column Chart")
                chosen = st.selectbox("Select Column", num_cols)
                chart = st.selectbox("Chart Type", ["Histogram", "Box Plot", "Line"])
                fig, ax = plt.subplots(figsize=(7, 4))
                vals = data[chosen].dropna()
                if chart == "Histogram":
                    ax.hist(vals, bins=20, color="#4C72B0", edgecolor="white")
                elif chart == "Box Plot":
                    ax.boxplot(vals, patch_artist=True,
                               boxprops=dict(facecolor="#4C72B0"))
                else:
                    ax.plot(vals.head(100).values, color="#4C72B0")
                ax.set_title(f"{chart} — {chosen}")
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                st.markdown("#### Correlation Heatmap")
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                sns.heatmap(data[num_cols].corr(), annot=True,
                            cmap="coolwarm", fmt=".2f", ax=ax2,
                            annot_kws={"size": 7})
                ax2.set_title("Feature Correlation Matrix")
                plt.tight_layout()
                st.pyplot(fig2)

            cat_cols = data.select_dtypes(include='object').columns.tolist()
            if cat_cols:
                st.markdown("#### Category Breakdown")
                cat = st.selectbox("Categorical Column", cat_cols)
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                vc = data[cat].value_counts().head(10)
                ax3.barh(vc.index.astype(str), vc.values, color="#4C72B0")
                ax3.invert_yaxis()
                ax3.set_title(f"Top values in '{cat}'")
                plt.tight_layout()
                st.pyplot(fig3)

            st.markdown("#### 🔍 Outlier Detection")
            outlier_col = st.selectbox("Check outliers in", num_cols, key="outlier")
            Q1 = data[outlier_col].quantile(0.25)
            Q3 = data[outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[
                (data[outlier_col] < Q1 - 1.5 * IQR) |
                (data[outlier_col] > Q3 + 1.5 * IQR)
            ]
            st.info(f"Found **{len(outliers)} outliers** in '{outlier_col}' "
                    f"({len(outliers)/len(data)*100:.1f}% of data)")
            if len(outliers) > 0:
                st.dataframe(outliers, use_container_width=True)

    # ============================================
    # TAB 4 — ML MODEL
    # ============================================
    with t4:
        st.subheader("🤖 Machine Learning Model")

        target = st.selectbox("🎯 Select Target Column", data.columns)
        test_pct = st.slider("Test Size %", 10, 40, 20)

        if st.button("🚀 Train Model", use_container_width=True):
            X = data.drop(columns=[target]).select_dtypes(include=np.number)
            y = data[target]

            if X.empty:
                st.error("No numeric feature columns found")
                return

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_pct / 100, random_state=42
            )

            is_clf = y.nunique() <= 10

            with st.spinner("Training model..."):
                if is_clf:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = accuracy_score(y_test, preds)
                    cv = cross_val_score(model, X, y, cv=5).mean()
                    metric_name = "Accuracy"

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Problem Type", "Classification")
                    c2.metric("Accuracy", f"{score:.2%}")
                    c3.metric("CV Score", f"{cv:.2%}")

                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_test, preds)
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    st.pyplot(fig_cm)

                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = r2_score(y_test, preds)
                    cv = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
                    metric_name = "R² Score"

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Problem Type", "Regression")
                    c2.metric("R² Score", f"{score:.4f}")
                    c3.metric("CV Score", f"{cv:.4f}")

                    st.markdown("#### Actual vs Predicted")
                    fig_ap, ax_ap = plt.subplots(figsize=(7, 4))
                    ax_ap.scatter(y_test, preds, alpha=0.6, color="#4C72B0")
                    ax_ap.plot([y_test.min(), y_test.max()],
                               [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax_ap.set_xlabel("Actual")
                    ax_ap.set_ylabel("Predicted")
                    ax_ap.set_title("Actual vs Predicted")
                    st.pyplot(fig_ap)

            st.markdown("#### 📊 Feature Importance")
            imp = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig_i, ax_i = plt.subplots(figsize=(8, 4))
            ax_i.barh(imp["Feature"][:10], imp["Importance"][:10], color="#4C72B0")
            ax_i.invert_yaxis()
            ax_i.set_title("Top Features by Importance")
            plt.tight_layout()
            st.pyplot(fig_i)
            st.dataframe(imp, use_container_width=True)

            st.session_state.update({
                "score": score,
                "cv": cv,
                "metric": metric_name,
                "target": target,
                "columns": list(data.columns),
                "features": imp.to_string(),
                "problem": "Classification" if is_clf else "Regression",
                "model_name": type(model).__name__,
                "rows": len(data),
                "cols": len(data.columns)
            })
            st.success("✅ Model trained! Go to AI Insight tab.")

    # ============================================
    # TAB 5 — AI INSIGHTS
    # ============================================
    with t5:
        st.subheader("🧠 AI-Powered Business Insights")
        st.caption("Powered by LLaMA3 via Groq — 100% Free")

        if "score" not in st.session_state:
            st.warning("⚠️ Please train a model first in the ML Model tab")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", st.session_state["model_name"])
        c2.metric(st.session_state["metric"], f"{st.session_state['score']:.2%}")
        c3.metric("CV Score", f"{st.session_state['cv']:.2%}")
        c4.metric("Problem", st.session_state["problem"])

        insight_type = st.selectbox("What do you want to know?", [
            "📊 Full Model Performance Analysis",
            "💼 Business Recommendations",
            "🔍 Key Feature Deep Dive",
            "⚠️ Model Risks & Limitations",
            "📈 How to Improve This Model"
        ])

        if st.button("🧠 Generate AI Insight", use_container_width=True):
            prompt = f"""
You are a Senior Data Scientist and Business Analyst.

Dataset: {st.session_state['rows']} rows, {st.session_state['cols']} columns
Columns: {st.session_state['columns']}
Target: {st.session_state['target']}
Problem type: {st.session_state['problem']}
Model: {st.session_state['model_name']}
{st.session_state['metric']}: {st.session_state['score']:.4f}
Cross-validation score: {st.session_state['cv']:.4f}

Feature Importances:
{st.session_state['features']}

Requested: {insight_type}

Give structured, business-friendly insights with:
1. Model performance in plain English
2. What key features mean for the business
3. Specific actionable recommendations with numbers
4. Risks and limitations
5. Clear next steps

Write for a business manager, not a data scientist.
            """

            with st.spinner("🧠 AI is analyzing your data..."):
                try:
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1200
                    )
                    result = response.choices[0].message.content
                    st.success("✅ Insight Generated!")
                    st.divider()
                    st.markdown(result)
                    st.session_state["report"] = result

                except Exception as e:
                    st.error(f"AI Error: {e}")

        if "report" in st.session_state:
            st.divider()
            full_report = f"""INSIGHTFORGE — AI ANALYSIS REPORT
===================================
Target: {st.session_state['target']}
Model: {st.session_state['model_name']}
{st.session_state['metric']}: {st.session_state['score']:.4f}
CV Score: {st.session_state['cv']:.4f}
Problem Type: {st.session_state['problem']}

AI ANALYSIS
-----------
{st.session_state['report']}

Generated by InsightForge | insightforge1234.streamlit.app
            """
            st.download_button(
                "⬇️ Download Report",
                full_report,
                file_name="InsightForge_Report.txt",
                use_container_width=True
            )


# ============================================
# ▶ RUN APP
# ============================================
if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    login()
else:
    app()
