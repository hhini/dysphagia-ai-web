import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px

# ================= 1. 页面配置 =================
st.set_page_config(
    page_title="Dysphagia AI (吞咽障碍智能预测)",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 2. HTML 文本定义 (无缩进) =================

HTML_ANALYSIS_REPORT = """
<div class="css-card">
    <h3 style="color:#1e3a8a;">📊 深度模型分析报告</h3>
    <h4>1. 综合性能指标</h4>
    <ul>
        <li><strong>区分能力 (AUC)：</strong> 两个模型的 AUC 值均表现优异，表明它们对“患病”和“不患病”人群有极强的区分能力。</li>
        <li><strong>准确率与稳定性：</strong> 随机森林模型引入了“中药服用史”作为第10个特征，在处理复杂交互关系上可能略优于逻辑回归。</li>
        <li><strong>临床应用：</strong> 逻辑回归仅需9个核心特征，计算简便，适合快速筛查；随机森林增加了用药史维度，适合更精细的评估。</li>
    </ul>
    <h4>2. 模型特性对比</h4>
    <table style="width:100%; border-collapse: collapse; margin-top: 10px;">
      <tr style="background-color: #f2f2f2;">
        <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">维度</th>
        <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Logistic Regression</th>
        <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Random Forest</th>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd;"><strong>特征数量</strong></td>
        <td style="padding: 8px; border: 1px solid #ddd;"><strong>9项</strong> (极简核心指标)</td>
        <td style="padding: 8px; border: 1px solid #ddd;"><strong>10项</strong> (增加中药服用史)</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd;"><strong>解释性</strong></td>
        <td style="padding: 8px; border: 1px solid #ddd;">高 (线性关系清晰)</td>
        <td style="padding: 8px; border: 1px solid #ddd;">中 (非线性交互强)</td>
      </tr>
    </table>
</div>
"""

HTML_ABOUT_SYSTEM = """
<div class="css-card">
    <h2 style="color: #1e3a8a;">🏥 关于本系统 (About)</h2>
    <p>本系统基于最新的临床数据训练，旨在辅助医护人员快速评估老年吞咽障碍风险。</p>
    <h4>🛠️ 模型配置</h4>
    <ul>
        <li><strong>Logistic Regression (逻辑回归)：</strong> 使用 9 项核心临床指标（如BMI、牙齿、认知状态等）。</li>
        <li><strong>Random Forest (随机森林)：</strong> 在核心指标基础上增加了“中药及中成药服用史”，共 10 项特征。</li>
    </ul>
    <h4>⚠️ 免责声明 (Disclaimer)</h4>
    <p style="color: #666;">
        本系统的预测结果仅供参考，<strong>不能替代专业医生的临床诊断</strong>。
        吞咽障碍的最终确诊需要结合临床查体、影像学检查（如 VFSS 或 FEES）由专业医疗团队做出。
    </p>
    <hr>
    <p style="text-align: center; color: #888;">© 2026 Dysphagia AI Research Group</p>
</div>
"""

# ================= 3. CSS 样式 (适配 Radio Button) =================
st.markdown("""
<style>
    /* 全局设置 */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }
    h1, h2, h3, h4, h5, h6, p, li, span, label, div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    
    /* 输入框 (Number Input) */
    .stNumberInput input {
        background-color: #ffffff !important; 
        color: #000000 !important;
        border: 1px solid #ccc !important;
    }
    
    /* 单选按钮 (Radio) 文字颜色 */
    div[role="radiogroup"] label p {
        color: #000000 !important;
        font-weight: 500;
    }

    /* 下拉框 (Selectbox) */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #ccc !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid #eee !important;
    }
    li[role="option"], div[role="option"] {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    li[role="option"]:hover, div[role="option"]:hover {
        background-color: #e9ecef !important;
        color: #000000 !important;
    }
    div[data-testid="stSelectbox"] div[class*="singleValue"] {
        color: #000000 !important;
    }

    /* 卡片与按钮 */
    .css-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    .stButton>button {
        background: #4361ee;
        color: white !important;
        border-radius: 8px;
        height: 48px;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background: #3a56d4; }
</style>
""", unsafe_allow_html=True)

# ================= 4. 特征定义 =================

FEATURES_LR = [
    'chewing', 'number_of_teeth', 'choking', 'eating', 
    'age', 'weight', 'frail', 'BMI', 'MMSE'
]

FEATURES_RF = [
    'chewing', 'number_of_teeth', 'choking', 'eating', 
    'age', 'weight', 'frail', 'BMI', 'MMSE', 
    'zhongyaojizhongchengyao'
]

# ================= 5. 工具函数 =================

def manual_standardization(df):
    """仅对连续变量进行标准化"""
    df_scaled = df.copy()
    stats_config = {
        'number_of_teeth': {'mean': 18.0,  'std': 9.299115},
        'weight':          {'mean': 60.0,  'std': 9.572267},
        'BMI':             {'mean': 23.0,  'std': 3.310996},
        'age':             {'mean': 75.0,  'std': 7.154127}
    }
    for col, stats in stats_config.items():
        if col in df_scaled.columns:
            df_scaled[col] = (df_scaled[col] - stats['mean']) / stats['std']
    return df_scaled

@st.cache_resource
def load_models():
    models = {}
    try:
        models['Logistic Regression'] = joblib.load("logistic_model.pkl")
    except:
        models['Logistic Regression'] = None
    try:
        models['Random Forest'] = joblib.load("random_forest_model.pkl")
    except:
        models['Random Forest'] = None
    return models

models = load_models()

# ================= 6. 主界面 (Banner 在顶部) =================

try:
    st.image("assets/banner.png", use_container_width=True)
except:
    # 如果找不到图片，显示一个带背景色的标题块
    st.markdown("""<div style="background: linear-gradient(90deg, #1e3a8a 0%, #4361ee 100%); padding: 30px; border-radius: 12px; color: white; text-align: center; margin-bottom: 25px;"><h1>Dysphagia Prediction System</h1></div>""", unsafe_allow_html=True)

# ================= 7. 侧边栏输入 (优化交互) =================
with st.sidebar:
    try:
        st.image("assets/logo.png", width=180)
    except:
        st.markdown("## 🏥 AI Med Assist")
    
    st.markdown("---")
    
    selected_model_name = st.selectbox(
        "🛠️ Select Model (选择模型)", 
        ["Logistic Regression", "Random Forest"],
        index=1
    )
    is_rf = selected_model_name == "Random Forest"
    
    with st.form("main_form"):
        # --- 1. 身体测量 (使用 Number Input 带加减号) ---
        st.markdown("### 1. Measurements (身体测量)")
        col1, col2 = st.columns(2)
        # step=1 确保出现加减按钮
        age = col1.number_input("Age (年龄)", min_value=20, max_value=110, value=75, step=1)
        hight = col2.number_input("Height (cm)", min_value=100, max_value=220, value=160, step=1)
        
        col3, col4 = st.columns(2)
        weight = col3.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=60.0, step=0.5)
        
        # 自动计算 BMI
        bmi_val = weight / ((hight / 100) ** 2)
        BMI = bmi_val
        col4.markdown(f"<div style='padding-top:35px; color:#4361ee; font-weight:bold;'>BMI: {bmi_val:.1f}</div>", unsafe_allow_html=True)

        # --- 2. 核心临床特征 (使用 Radio Button 增强可视性) ---
        st.markdown("---")
        st.markdown("### 2. Clinical Status (临床状态)")
        
        # 牙齿 - 使用 Number Input 方便加减
        number_of_teeth = st.number_input("Number of Teeth (牙齿数量)", min_value=0, max_value=32, value=20, step=1)
        
        st.markdown("---")
        # 咀嚼 - 使用 Radio Button (水平排列)，用户一眼就能看到选了 Yes 还是 No
        chewing = st.radio(
            "Chewing Difficulty (咀嚼障碍)", 
            [0, 1], 
            format_func=lambda x: "无 (No)" if x==0 else "有 (Yes)",
            horizontal=True
        )
        
        # 呛咳 - Radio Button
        choking = st.radio(
            "Choking History (呛咳史)", 
            [0, 1], 
            format_func=lambda x: "无 (No)" if x==0 else "有 (Yes)",
            horizontal=True
        )
        
        # 进食 - Selectbox (选项较多，Radio太占地，但Selectbox已修复可见性)
        c_a3, c_a4 = st.columns(2)
        eat_map = {0: "良好 (Good)", 1: "一般 (Fair)", 2: "差 (Poor)"}
        eating = c_a3.selectbox("Eating (进食情况)", [0, 1, 2], format_func=lambda x: eat_map[x])
        
        frail_map = {0: "无 (None)", 1: "衰弱前期 (Pre)", 2: "衰弱 (Frail)"}
        frail = c_a4.selectbox("Frailty (衰弱状态)", [0, 1, 2], format_func=lambda x: frail_map[x])
        
        # 认知
        mmse_map = {0:"正常", 1:"轻度障碍", 2:"中度障碍", 3:"重度障碍"}
        MMSE = st.selectbox("MMSE (认知功能)", [0, 1, 2, 3], format_func=lambda x: mmse_map[x])

        # --- 3. 随机森林专属特征 ---
        zhongyaojizhongchengyao = 0
        if is_rf:
            st.markdown("---")
            st.markdown("### 3. Medication (用药)")
            # 中药 - Radio Button
            zhongyaojizhongchengyao = st.radio(
                "TCM Usage (中药/中成药)", 
                [0, 1], 
                format_func=lambda x: "无 (No)" if x==0 else "有 (Yes)",
                horizontal=True,
                help="是否正在服用中药或中成药"
            )

        st.markdown("---")
        submit_btn = st.form_submit_button("🚀 Run Prediction")

# ================= 8. 主内容区 (Tabs) =================

tab_diagnosis, tab_explain, tab_about = st.tabs(["🩺 AI Diagnosis", "📊 Analysis", "ℹ️ About"])

# ------ 1. 诊断 ------
with tab_diagnosis:
    if submit_btn:
        model = models[selected_model_name]
        
        if model is None:
            st.error(f"❌ Error: Model file for {selected_model_name} not found.")
        else:
            full_data = {
                'chewing': chewing, 
                'number_of_teeth': number_of_teeth, 
                'choking': choking, 
                'eating': eating, 
                'age': age, 
                'weight': weight, 
                'frail': frail, 
                'BMI': BMI, 
                'MMSE': MMSE,
                'zhongyaojizhongchengyao': zhongyaojizhongchengyao
            }
            raw_df = pd.DataFrame([full_data])
            
            try:
                if not is_rf:
                    input_df = raw_df.reindex(columns=FEATURES_LR)
                    final_input = manual_standardization(input_df)
                else:
                    input_df = raw_df.reindex(columns=FEATURES_RF)
                    final_input = input_df
                
                prediction = model.predict(final_input)[0]
                if hasattr(model, 'predict_proba'):
                    prob_pos = model.predict_proba(final_input)[0][1]
                else:
                    prob_pos = float(prediction)
                
                st.markdown(f"### Diagnosis Result: {selected_model_name}")
                col_res1, col_res2 = st.columns([1, 1.5])
                with col_res1:
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob_pos * 100,
                        number = {'suffix': "%", 'font': {'color': "#000000"}},
                        title = {'text': "Dysphagia Risk", 'font': {'color': "#000000"}},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#ef233c" if prob_pos > 0.5 else "#2a9d8f"}
                        }
                    ))
                    fig.update_layout(height=280, margin=dict(t=30,b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_res2:
                    if prob_pos > 0.5:
                        st.markdown(f"""
<div class="css-card" style="border-left: 6px solid #ef233c; background-color: #fff5f5;">
    <h2 style="color: #ef233c !important; margin-top:0;">⚠️ High Risk Detected</h2>
    <p style="font-size: 1.1em;">Probability: <strong>{prob_pos*100:.1f}%</strong></p>
    <hr>
    <p><strong>🚨 建议与干预：</strong></p>
    <ul style="line-height: 1.6;">
        <li><strong>转诊：</strong> 建议咨询言语治疗师(SLP)或进行VFSS检查。</li>
        <li><strong>饮食：</strong> 避免干硬食物，考虑使用增稠剂。</li>
        <li><strong>姿势：</strong> 尝试低头吞咽 (Chin Tuck)。</li>
    </ul>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
<div class="css-card" style="border-left: 6px solid #2a9d8f; background-color: #f0fdf4;">
    <h2 style="color: #2a9d8f !important; margin-top:0;">✅ Low Risk</h2>
    <p style="font-size: 1.1em;">Probability: <strong>{prob_pos*100:.1f}%</strong></p>
    <hr>
    <p><strong>💡 维持建议：</strong></p>
    <ul style="line-height: 1.6;">
        <li><strong>监测：</strong> 每年定期进行吞咽功能筛查。</li>
        <li><strong>习惯：</strong> 细嚼慢咽，保持良好口腔卫生。</li>
        <li><strong>营养：</strong> 保证蛋白质摄入，维持肌肉力量。</li>
    </ul>
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Analysis Error: {e}")
                st.write("Input columns:", final_input.columns.tolist())
    else:
        st.info("👈 请在左侧输入数据并点击 'Run Prediction'")

# ------ 2. 分析 ------
with tab_explain:
    st.markdown("### 🔍 Feature Importance")
    model = models[selected_model_name]

    if model:
        try:
            if not is_rf:
                importances = model.coef_[0] if hasattr(model, 'coef_') else model.named_steps['clf'].coef_[0]
                feature_names = FEATURES_LR
                color_scale = 'RdBu_r'
            else:
                importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.named_steps['clf'].feature_importances_
                feature_names = FEATURES_RF
                color_scale = 'Viridis'

            if len(importances) == len(feature_names):
                df_imp = pd.DataFrame({'Feature': feature_names, 'Value': importances})
                df_imp['AbsValue'] = df_imp['Value'].abs()
                df_imp = df_imp.sort_values(by='AbsValue', ascending=True)

                fig_bar = px.bar(df_imp, x='Value', y='Feature', orientation='h',
                                 title=f"Feature Contribution ({selected_model_name})",
                                 color='Value', color_continuous_scale=color_scale)
                fig_bar.update_layout(font=dict(color="black"), plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning(f"Feature count mismatch: Model({len(importances)}) vs List({len(feature_names)})")
        except Exception as e:
            st.error(f"Plot Error: {e}")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix**")
        img_name = "Test_CM_Logistic.png" if not is_rf else "Test_CM_RandomForest.png"
        try:
            st.image(f"assets/{img_name}", use_container_width=True)
        except:
            st.warning("Missing Image")
    with c2:
        st.markdown("**ROC Curve**")
        try:
            st.image("assets/Test_ROC_Comparison.png", use_container_width=True)
        except:
            st.warning("Missing Image")
            
    st.markdown("**Metrics Comparison**")
    try:
        st.image("assets/Test_Metrics_Comparison.png", use_container_width=True)
    except:
        st.warning("Missing Image")

    st.markdown(HTML_ANALYSIS_REPORT, unsafe_allow_html=True)

# ------ 3. 关于 ------
with tab_about:
    st.markdown(HTML_ABOUT_SYSTEM, unsafe_allow_html=True)