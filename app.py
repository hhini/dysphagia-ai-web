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

# ================= 2. 终极 CSS 修复 (下拉框 + 字体) =================
st.markdown("""
<style>
    /* 1. 强制全局白底黑字 */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #f8f9fa !important;
        color: #000000 !important;
    }
    
    /* 2. 侧边栏 */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }
    
    /* 3. 强制黑色文本 */
    h1, h2, h3, h4, h5, h6, p, li, span, label, div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    
    /* 4. 输入框样式 */
    .stTextInput input, .stNumberInput input {
        background-color: #ffffff !important; 
        color: #000000 !important;
        border: 1px solid #ccc !important;
    }
    
    /* 5. 关键修复：下拉菜单 (Selectbox) */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-color: #ccc !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #ffffff !important;
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

    /* 6. 卡片样式 */
    .css-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    /* 7. 按钮样式 */
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

# ================= 3. 特征定义 =================

FEATURES_LR = [
    'chewing', 'choking', 'eating', 'number_of_teeth', 'number_of_hospitalizations', 
    'hight', 'age', 'BMI', 'weight', 'MMSE', 'frail', 'number_of_drug_types', 
    'kangningyao', 'SSRS', 'dry_mouth', 'occupation', 'education'
]

FEATURES_RF = [
    'chewing', 'choking', 'eating', 'number_of_teeth', 'number_of_hospitalizations', 
    'hight', 'age', 'BMI', 'weight', 'MMSE', 'frail', 'number_of_drug_types', 
    'kangningyao', 'SSRS', 'dry_mouth', 'occupation', 'education', 'CVD', 
    'number_of_diseases', 'zhongyaojizhongchengyao', 'gum', 'MNA_SF', 
    'monthly_income', 'jiangyayao', 'drink', 'zhenjingcuimianyao', 'caregiver', 
    'residence', 'hospitalization', 'dentures', 'total_drugs', 'exercise'
]

# ================= 4. 工具函数 =================

def manual_standardization(df):
    df_scaled = df.copy()
    stats_config = {
        'number_of_teeth': {'mean': 18.0,  'std': 9.299115},
        'weight':          {'mean': 60.0,  'std': 9.572267},
        'BMI':             {'mean': 23.0,  'std': 3.310996},
        'age':             {'mean': 75.0,  'std': 7.154127},
        'hight':           {'mean': 160.0, 'std': 8.207334}
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

# ================= 5. 侧边栏 =================
with st.sidebar:
    try:
        st.image("assets/logo.png", width=180)
    except:
        st.markdown("## 🏥 AI Med Assist")
    
    st.markdown("---")
    
    selected_model_name = st.selectbox(
        "🛠️ Select Model (选择模型)", 
        ["Logistic Regression", "Random Forest"],
        index=0
    )
    is_rf = selected_model_name == "Random Forest"
    
    with st.form("main_form"):
        st.markdown("### 1. Demographics & Body")
        col1, col2 = st.columns(2)
        age = col1.number_input("Age (年龄)", 40, 110, 75)
        hight = col2.number_input("Height (cm)", 100, 200, 160)
        col3, col4 = st.columns(2)
        weight = col3.number_input("Weight (kg)", 30, 120, 60)
        bmi_val = weight / ((hight / 100) ** 2)
        BMI = bmi_val
        col4.markdown(f"<div style='padding-top:25px;'><b>BMI: {bmi_val:.1f}</b></div>", unsafe_allow_html=True)
        
        edu_map = {0:"文盲", 1:"小学", 2:"初中", 3:"高中", 4:"大专+"}
        education = st.selectbox("Education", [0,1,2,3,4], format_func=lambda x: f"{x}: {edu_map[x]}", index=2)
        
        occ_map = {0:"农民", 1:"工人", 2:"其他/脑力"}
        occupation = st.selectbox("Occupation", [0,1,2], format_func=lambda x: occ_map[x])

        st.markdown("---")
        st.markdown("### 2. Clinical & Oral")
        number_of_teeth = st.slider("Teeth Count (牙齿数)", 0, 32, 20)
        
        c_a1, c_a2 = st.columns(2)
        chewing = c_a1.selectbox("Chewing Difficulty", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        choking = c_a2.selectbox("Choking History", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        
        c_a3, c_a4 = st.columns(2)
        eating = c_a3.selectbox("Eating Ability", [0, 1, 2], help="0:独立, 1:需协助, 2:依赖")
        frail = c_a4.selectbox("Frailty Status", [0, 1, 2], help="0:健壮, 1:衰弱前期, 2:衰弱")
        
        c_a5, c_a6 = st.columns(2)
        MMSE = c_a5.selectbox("MMSE Level", [0, 1, 2])
        hosp_map = {0:"0次", 1:"1-2次", 2:"3-4次", 3:"5次+"}
        number_of_hospitalizations = c_a6.selectbox("Hosp. Freq", [0, 1, 2, 3], format_func=lambda x: hosp_map[x])

        st.markdown("---")
        st.markdown("### 3. Medications & Social")
        c_b1, c_b2 = st.columns(2)
        dry_mouth = c_b1.selectbox("Dry Mouth (口干)", [0, 1])
        kangningyao = c_b2.selectbox("Antipsychotics", [0, 1])
        c_b3, c_b4 = st.columns(2)
        number_of_drug_types = c_b3.selectbox("Drug Types", [0,1,2,3,4,5])
        SSRS = c_b4.selectbox("Social Support (SSRS)", [0, 1, 2])

        # 初始化 RF 变量
        total_drugs=0; number_of_diseases=0; zhenjingcuimianyao=0; jiangyayao=0; 
        zhongyaojizhongchengyao=0; CVD=0; hospitalization=0; gum=0; dentures=0;
        MNA_SF=0; monthly_income=0; drink=0; exercise=0; caregiver=0; residence=0

        if is_rf:
            st.markdown("---")
            st.markdown("### 4. RF Detailed Survey")
            with st.expander("📋 Expand for Details", expanded=True):
                c_rf1, c_rf2 = st.columns(2)
                total_drugs = c_rf1.selectbox("Total Drugs", [0,1,2,3,4,5])
                number_of_diseases = c_rf2.slider("Diseases Count", 0, 6, 1)
                
                c_rf3, c_rf4, c_rf5 = st.columns(3)
                zhenjingcuimianyao = c_rf3.selectbox("Sedatives", [0, 1])
                jiangyayao = c_rf4.selectbox("Anti-HTN", [0, 1])
                zhongyaojizhongchengyao = c_rf5.selectbox("TCM", [0, 1])
                
                c_rf6, c_rf7, c_rf8 = st.columns(3)
                CVD = c_rf6.selectbox("CVD History", [0, 1])
                hospitalization = c_rf7.selectbox("In-Patient?", [0, 1])
                gum = c_rf8.selectbox("Gum Issues", [0, 1])
                
                c_rf9, c_rf10 = st.columns(2)
                dentures = c_rf9.selectbox("Dentures", [0, 1])
                MNA_SF = c_rf10.selectbox("MNA-SF", [0, 1, 2])
                
                c_rf11, c_rf12 = st.columns(2)
                monthly_income = c_rf11.selectbox("Income", [0, 1, 2, 3, 4])
                drink = c_rf12.selectbox("Drink", [0, 1, 2, 3, 10])
                
                c_rf13, c_rf14, c_rf15 = st.columns(3)
                exercise = c_rf13.selectbox("Exercise", [0, 1, 2, 3, 4])
                caregiver = c_rf14.selectbox("Caregiver", [0, 1, 2, 4])
                residence = c_rf15.selectbox("Residence", [0, 1, 2, 3, 4])

        submit_btn = st.form_submit_button("🚀 Run Prediction")

# ================= 6. 主逻辑 =================

try:
    st.image("assets/banner.png", use_container_width=True)
except:
    st.markdown("""<div style="background: linear-gradient(90deg, #1e3a8a 0%, #4361ee 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;"><h1>Dysphagia Prediction System</h1></div>""", unsafe_allow_html=True)

tab_diagnosis, tab_explain, tab_about = st.tabs(["🩺 AI Diagnosis (智能诊断)", "📊 Model Analysis (模型分析)", "ℹ️ About (关于)"])

# ------ 1. 诊断与建议 ------
with tab_diagnosis:
    if submit_btn:
        model = models[selected_model_name]
        
        if model is None:
            st.error(f"❌ Error: Model file for {selected_model_name} not found.")
        else:
            full_data = {
                'chewing': chewing, 'choking': choking, 'eating': eating, 
                'number_of_teeth': number_of_teeth, 'number_of_hospitalizations': number_of_hospitalizations,
                'hight': hight, 'age': age, 'BMI': BMI, 'weight': weight, 
                'MMSE': MMSE, 'frail': frail, 'number_of_drug_types': number_of_drug_types,
                'kangningyao': kangningyao, 'SSRS': SSRS, 'dry_mouth': dry_mouth, 
                'occupation': occupation, 'education': education,
                'CVD': CVD, 'number_of_diseases': number_of_diseases,
                'zhongyaojizhongchengyao': zhongyaojizhongchengyao, 'gum': gum, 
                'MNA_SF': MNA_SF, 'monthly_income': monthly_income, 'jiangyayao': jiangyayao, 
                'drink': drink, 'zhenjingcuimianyao': zhenjingcuimianyao, 
                'caregiver': caregiver, 'residence': residence, 'hospitalization': hospitalization, 
                'dentures': dentures, 'total_drugs': total_drugs, 'exercise': exercise
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
                
                # --- 结果展示区 ---
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
                    # ⚠️ 这里也是 HTML 字符串，必须顶格写
                    if prob_pos > 0.5:
                        st.markdown(f"""
<div class="css-card" style="border-left: 6px solid #ef233c; background-color: #fff5f5;">
    <h2 style="color: #ef233c !important; margin-top:0;">⚠️ High Risk Detected</h2>
    <p style="font-size: 1.1em;">Probability: <strong>{prob_pos*100:.1f}%</strong></p>
    <hr>
    <p><strong>🚨 临床建议与干预措施：</strong></p>
    <ul style="line-height: 1.6;">
        <li><strong>立即转诊：</strong> 建议咨询言语语言治疗师 (SLP) 进行吞咽造影检查 (VFSS)。</li>
        <li><strong>饮食调整：</strong> 
            <ul>
                <li>避免干硬、易碎食物（如坚果、饼干）。</li>
                <li>考虑使用增稠剂调整液体粘稠度，防止误吸。</li>
            </ul>
        </li>
        <li><strong>代偿性体位：</strong> 尝试“低头吞咽 (Chin Tuck)”姿势，保护气道。</li>
        <li><strong>口腔护理：</strong> 强化口腔清洁，减少吸入性肺炎风险。</li>
    </ul>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
<div class="css-card" style="border-left: 6px solid #2a9d8f; background-color: #f0fdf4;">
    <h2 style="color: #2a9d8f !important; margin-top:0;">✅ Low Risk</h2>
    <p style="font-size: 1.1em;">Probability: <strong>{prob_pos*100:.1f}%</strong></p>
    <hr>
    <p><strong>💡 健康维持建议：</strong></p>
    <ul style="line-height: 1.6;">
        <li><strong>定期监测：</strong> 每年进行一次简单的吞咽筛查，尤其是高龄老人。</li>
        <li><strong>健康饮食：</strong> 保持均衡饮食，多摄入富含蛋白质的食物以维持肌肉力量。</li>
        <li><strong>良好的进食习惯：</strong> 细嚼慢咽，避免进食时大声说话或分心。</li>
        <li><strong>牙齿保健：</strong> 定期看牙医，保持咀嚼功能完好。</li>
    </ul>
</div>
""", unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Analysis Error: {e}")
    else:
        st.info("👈 请在左侧输入患者数据并点击 'Run Prediction'")

# ------ 2. 模型分析 ------
with tab_explain:
    st.markdown("### 🔍 Model Feature Importance")
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
                df_imp = df_imp.sort_values(by='AbsValue', ascending=True).tail(17)

                fig_bar = px.bar(df_imp, x='Value', y='Feature', orientation='h',
                                 title=f"Top 17 Influential Factors ({selected_model_name})",
                                 color='Value', color_continuous_scale=color_scale)
                fig_bar.update_layout(font=dict(color="black"), plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error(f"Plot Error: {e}")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix (混淆矩阵)**")
        img_name = "Test_CM_Logistic.png" if not is_rf else "Test_CM_RandomForest.png"
        try:
            st.image(f"assets/{img_name}", use_container_width=True)
        except:
            st.warning("Missing Image")
    with c2:
        st.markdown("**ROC Curve Comparison (ROC对比)**")
        try:
            st.image("assets/Test_ROC_Comparison.png", use_container_width=True)
        except:
            st.warning("Missing Image")
            
    st.markdown("**Metrics Comparison (综合指标对比)**")
    try:
        st.image("assets/Test_Metrics_Comparison.png", use_container_width=True)
    except:
        st.warning("Missing Image")

    # 🔴 修复重点：这里的 HTML 字符串取消了所有缩进，顶格写
    st.markdown("""
<div class="css-card">
    <h3 style="color:#1e3a8a;">📊 深度模型分析报告</h3>
    <h4>1. 综合性能指标 (ROC & Metrics)</h4>
    <ul>
        <li><strong>曲线下面积 (AUC)：</strong> 两个模型的 AUC 值均超过了 0.92（逻辑回归 <strong>0.922</strong>，随机森林 <strong>0.923</strong>），这表明它们对“患病”和“不患病”人群的区分能力非常出色。</li>
        <li><strong>准确率 (Accuracy)：</strong> 随机森林（<strong>82.2%</strong>）略高于逻辑回归（81.5%）。</li>
        <li><strong>特异度 (Specificity)：</strong> 随机森林在识别“无障碍”人群方面表现更好（<strong>84.7%</strong> vs 83.3%），这意味着它误诊正常人为病人的概率更低。</li>
        <li><strong>召回率 (Recall)：</strong> 两个模型完全一致，均为 <strong>79.4%</strong>。这意味着它们在捕捉真正患有吞咽障碍的患者方面效果相同。</li>
    </ul>
    <h4>2. 混淆矩阵深度分析</h4>
    <ul>
        <li><strong>预测“No”的准确性（左上角）：</strong> 随机森林（<strong>84.72%</strong>）比逻辑回归（83.33%）表现更稳健，漏掉健康人的概率更小。</li>
        <li><strong>预测“Yes”的准确性（右下角）：</strong> 两个模型表现完全一致，均为 <strong>79.37%</strong>。这意味着对于真正的患者，两个模型的识别率是一样的。</li>
        <li><strong>误诊与漏诊率：</strong> 两个模型都有约 <strong>20.63%</strong> 的患者被错误地预测为“No”（假阴性/漏诊），这在临床筛查中是未来需要通过调整阈值进一步优化的重点。</li>
    </ul>
    <h4>3. 模型特性及应用建议</h4>
    <table style="width:100%; border-collapse: collapse; margin-top: 10px;">
      <tr style="background-color: #f2f2f2;">
        <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">特性</th>
        <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Logistic Regression (逻辑回归)</th>
        <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Random Forest (随机森林)</th>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd;"><strong>模型类型</strong></td>
        <td style="padding: 8px; border: 1px solid #ddd;">线性参数模型</td>
        <td style="padding: 8px; border: 1px solid #ddd;">非线性集成树模型</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd;"><strong>解释性</strong></td>
        <td style="padding: 8px; border: 1px solid #ddd;"><strong>极高</strong>。它能通过系数告诉你每个特征对风险的具体贡献量。</td>
        <td style="padding: 8px; border: 1px solid #ddd;">中等。能看出特征重要性，但很难直观解释交互作用。</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd;"><strong>最佳用途</strong></td>
        <td style="padding: 8px; border: 1px solid #ddd;">适合临床医生直观理解因素，生成评分表。</td>
        <td style="padding: 8px; border: 1px solid #ddd;">适合追求极致精度的后台自动化系统。</td>
      </tr>
    </table>
</div>
""", unsafe_allow_html=True)

# ------ 3. 关于 ------
with tab_about:
    # 🔴 修复重点：这里的 HTML 字符串取消了所有缩进，顶格写
    st.markdown("""
<div class="css-card">
    <h2 style="color: #1e3a8a;">🏥 关于本系统 (About)</h2>
    <p>本系统是一个基于机器学习的老年吞咽障碍风险筛查工具，旨在辅助医护人员快速评估患者风险。</p>
    <h4>🛠️ 技术背景</h4>
    <ul>
        <li><strong>开发语言：</strong> Python (Streamlit, Scikit-learn, Plotly)</li>
        <li><strong>核心算法：</strong> 
            <ul>
                <li><strong>Logistic Regression：</strong> 经典统计学模型，提供高可解释性。</li>
                <li><strong>Random Forest：</strong> 集成学习模型，提供高精度预测。</li>
            </ul>
        </li>
        <li><strong>数据基础：</strong> 模型基于真实临床数据集训练，包含人口学、口腔状态、用药史等 32 个维度的特征。</li>
    </ul>
    <h4>⚠️ 免责声明 (Disclaimer)</h4>
    <p style="color: #666;">
        本系统的预测结果仅供参考，<strong>不能替代专业医生的临床诊断</strong>。
        吞咽障碍的最终确诊需要结合临床查体、影像学检查（如 VFSS 或 FEES）由专业医疗团队做出。
        如果您对自己或家人的吞咽功能有疑虑，请务必咨询医生。
    </p>
    <hr>
    <p style="text-align: center; color: #888;">© 2026 Dysphagia AI Research Group</p>
</div>
""", unsafe_allow_html=True)