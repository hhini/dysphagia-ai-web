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

# ================= 2. HTML 文本定义 (更新模型描述) =================

HTML_ANALYSIS_REPORT = """
<div class="css-card">
    <h3 style="color:#1e3a8a;">📊 深度模型分析报告</h3>
    <h4>1. 综合性能指标</h4>
    <ul>
        <li><strong>区分能力 (AUC)：</strong> 两个模型均经过临床数据验证，能有效区分吞咽障碍高风险与低风险人群。</li>
        <li><strong>模型差异：</strong> 随机森林模型纳入了身高、病史（脑血管病、抗凝药）等更多维度，适合全面评估；逻辑回归侧重于核心功能的快速筛查。</li>
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
        <td style="padding: 8px; border: 1px solid #ddd;"><strong>10项</strong> (包含基本身体指标与认知功能)</td>
        <td style="padding: 8px; border: 1px solid #ddd;"><strong>14项</strong> (增加疾病史、抗凝药、身高等详细指标)</td>
      </tr>
      <tr>
        <td style="padding: 8px; border: 1px solid #ddd;"><strong>适用场景</strong></td>
        <td style="padding: 8px; border: 1px solid #ddd;">快速筛查，关注核心功能</td>
        <td style="padding: 8px; border: 1px solid #ddd;">精细化评估，考虑多重共病影响</td>
      </tr>
    </table>
</div>
"""

HTML_ABOUT_SYSTEM = """
<div class="css-card">
    <h2 style="color: #1e3a8a;">🏥 关于本系统 (About)</h2>
    <p>本系统基于老年医学临床数据训练，旨在辅助医护人员评估吞咽障碍风险。</p>
    <h4>🛠️ 模型配置</h4>
    <ul>
        <li><strong>Logistic Regression (逻辑回归)：</strong> 使用 10 项核心指标（咀嚼、呛咳、牙齿、进食、年龄、体重、服药种类、MMSE、BMI、衰弱）。</li>
        <li><strong>Random Forest (随机森林)：</strong> 在逻辑回归基础上增加了“抗凝药使用”、“身高”、“脑血管病(CVD)”及“疾病种类数”，共 14 项特征。</li>
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

# ================= 3. CSS 样式 (保持不变) =================
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

# ================= 4. 特征定义 (严格按照提供的顺序) =================

# 逻辑回归 (10个特征)
FEATURES_LR = [
    'chewing',               # 1. 咀嚼障碍
    'choking',               # 2. 呛咳史
    'number_of_teeth',       # 3. 牙齿数量
    'eating',                # 4. 进食情况
    'age',                   # 5. 年龄
    'weight',                # 6. 体重
    'number_of_drug_types',  # 7. 药物种类数
    'MMSE',                  # 8. 认知功能
    'BMI',                   # 9. BMI
    'frail'                  # 10. 衰弱状态
]

# 随机森林 (14个特征)
FEATURES_RF = [
    'chewing',               # 1
    'choking',               # 2
    'number_of_teeth',       # 3
    'eating',                # 4
    'age',                   # 5
    'weight',                # 6
    'number_of_drug_types',  # 7
    'MMSE',                  # 8
    'BMI',                   # 9
    'frail',                 # 10
    'kangningyao',           # 11. 抗凝药
    'hight',                 # 12. 身高 (注意变量名是 hight)
    'CVD',                   # 13. 脑血管疾病
    'number_of_diseases'     # 14. 疾病种类数
]

# ================= 5. 工具函数 =================

def manual_standardization(df):
    """仅对逻辑回归中已知的连续变量进行标准化"""
    # 注意：如果 number_of_drug_types 等新变量需要标准化，请在此处添加对应的 mean/std
    df_scaled = df.copy()
    stats_config = {
        'number_of_teeth': {'mean': 18.0,  'std': 9.299115},
        'weight':          {'mean': 60.0,  'std': 9.572267},
        'BMI':             {'mean': 23.0,  'std': 3.310996},
        'age':             {'mean': 75.0,  'std': 7.154127}
        # 如果需要对 number_of_drug_types 进行标准化，请取消注释并填入数值
        # 'number_of_drug_types': {'mean': X.X, 'std': Y.Y},
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

# ================= 6. 主界面 =================

try:
    st.image("assets/banner.png", use_container_width=True)
except:
    st.markdown("""<div style="background: linear-gradient(90deg, #1e3a8a 0%, #4361ee 100%); padding: 30px; border-radius: 12px; color: white; text-align: center; margin-bottom: 25px;"><h1>Dysphagia Prediction System</h1></div>""", unsafe_allow_html=True)

# ================= 7. 侧边栏输入 (更新控件) =================
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
        # --- 1. 身体测量与基本信息 ---
        st.markdown("### 1. Basic Info (基本信息)")
        col1, col2 = st.columns(2)
        age = col1.number_input("Age (年龄)", min_value=20, max_value=120, value=75, step=1)
        # Height 即使LR不用，也需要用来计算BMI
        hight = col2.number_input("Height (cm)", min_value=100, max_value=220, value=160, step=1)
        
        col3, col4 = st.columns(2)
        weight = col3.number_input("Weight (kg)", min_value=30.0, max_value=150.0, value=60.0, step=0.5)
        
        # 自动计算 BMI
        bmi_val = weight / ((hight / 100) ** 2)
        BMI = bmi_val
        col4.markdown(f"<div style='padding-top:35px; color:#4361ee; font-weight:bold;'>BMI: {bmi_val:.1f}</div>", unsafe_allow_html=True)

        # --- 2. 核心症状 (咀嚼/呛咳/牙齿/进食) ---
        st.markdown("---")
        st.markdown("### 2. Oral & Feeding (口腔与进食)")
        
        # 咀嚼 (Chewing)
        chewing = st.radio(
            "1. Chewing Difficulty (咀嚼障碍)", 
            [0, 1], 
            format_func=lambda x: "0: 无 (No)" if x==0 else "1: 有 (Yes)",
            horizontal=True
        )

        # 呛咳 (Choking)
        choking = st.radio(
            "2. Choking History (呛咳史)", 
            [0, 1], 
            format_func=lambda x: "0: 无 (No)" if x==0 else "1: 有 (Yes)",
            horizontal=True
        )

        c_oral1, c_oral2 = st.columns(2)
        # 牙齿数量
        number_of_teeth = c_oral1.number_input("3. Teeth Count (牙齿数量)", min_value=0, max_value=32, value=20, step=1)
        
        # 进食情况
        eat_map = {0: "0: 良好", 1: "1: 一般", 2: "2: 差"}
        eating = c_oral2.selectbox("4. Eating Status (进食情况)", [0, 1, 2], format_func=lambda x: eat_map[x])

        # --- 3. 临床状态 (MMSE/衰弱/药物) ---
        st.markdown("---")
        st.markdown("### 3. Clinical Status (临床状态)")
        
        mmse_map = {0:"0: 正常", 1:"1: 轻度障碍", 2:"2: 中度障碍"} # 无重度(3)
        MMSE = st.selectbox("MMSE (认知功能)", [0, 1, 2], format_func=lambda x: mmse_map[x])

        frail_map = {0: "0: 无衰弱", 1: "1: 衰弱前期", 2: "2: 衰弱"}
        frail = st.selectbox("Frailty (衰弱状态)", [0, 1, 2], format_func=lambda x: frail_map[x])
        
        # 药物种类数 (LR 和 RF 都用)
        number_of_drug_types = st.number_input("Drugs Count (长期服用药物种类数)", min_value=0, max_value=20, value=3, step=1)

        # --- 4. 随机森林专属特征 (11-14) ---
        kangningyao = 0
        CVD = 0
        number_of_diseases = 0
        
        if is_rf:
            st.markdown("---")
            st.markdown("### 4. History (病史 - RF模型专用)")
            
            # 11. 抗凝药
            kangningyao = st.radio(
                "Anticoagulant Use (抗凝药)", 
                [0, 1], 
                format_func=lambda x: "0: 无 (No)" if x==0 else "1: 有 (Yes)",
                horizontal=True
            )
            
            # 13. 脑血管疾病
            CVD = st.radio(
                "CVD (脑血管疾病)", 
                [0, 1], 
                format_func=lambda x: "0: 无 (No)" if x==0 else "1: 有 (Yes)",
                horizontal=True
            )
            
            # 14. 疾病种类数
            number_of_diseases = st.number_input("Diseases Count (疾病种类数)", min_value=0, max_value=20, value=2, step=1)
            
            # 12. hight 已在上方输入

        st.markdown("---")
        submit_btn = st.form_submit_button("🚀 Run Prediction")

# ================= 8. 主内容区 (Tabs) =================

tab_diagnosis, tab_explain, tab_about = st.tabs(["🩺 AI Diagnosis", "📊 Analysis", "ℹ️ About"])
# ------ 1. 诊断 (修复版：自动识别 pipeline 键) ------
with tab_diagnosis:
    if submit_btn:
        # 1. 获取加载的对象
        loaded_object = models[selected_model_name]
        
        if loaded_object is None:
            st.error(f"❌ Error: Model file for {selected_model_name} not found.")
        else:
            # ================== 核心修复开始 ==================
            model = None
            # 检查加载的是不是字典
            if isinstance(loaded_object, dict):
                # 你的报错显示键名是 'pipeline'，所以把它放在第一个
                possible_keys = ['pipeline', 'model', 'classifier', 'clf', 'estimator']
                for key in possible_keys:
                    if key in loaded_object:
                        model = loaded_object[key]
                        st.success(f"✅ Successfully loaded model from key: '{key}'") # 提示用户加载成功
                        break
                
                # 如果还是找不到
                if model is None:
                    st.error(f"❌ Error: Could not find model in dictionary. Keys found: {list(loaded_object.keys())}")
                    st.stop()
            else:
                # 如果不是字典，直接使用
                model = loaded_object
            # ================== 核心修复结束 ==================

            # 2. 准备数据
            full_data = {
                'chewing': chewing, 
                'choking': choking,
                'number_of_teeth': number_of_teeth, 
                'eating': eating, 
                'age': age, 
                'weight': weight,
                'number_of_drug_types': number_of_drug_types,
                'MMSE': MMSE,
                'BMI': BMI, 
                'frail': frail, 
                'kangningyao': kangningyao,
                'hight': hight,
                'CVD': CVD,
                'number_of_diseases': number_of_diseases
            }
            raw_df = pd.DataFrame([full_data])
            
            try:
                # 3. 数据预处理
                if not is_rf:
                    # 逻辑回归：取前10个特征
                    input_df = raw_df.reindex(columns=FEATURES_LR)
                    # 注意：如果你的 'pipeline' 里已经包含了 StandardScaler，
                    # 这里的 manual_standardization 可能会导致二次标准化。
                    # 如果预测结果非常奇怪（比如全是0或1），请尝试注释掉下面这一行：
                    final_input = manual_standardization(input_df) 
                else:
                    # 随机森林：取14个特征
                    input_df = raw_df.reindex(columns=FEATURES_RF)
                    final_input = input_df
                
                # 4. 进行预测
                prediction = model.predict(final_input)[0]
                
                if hasattr(model, 'predict_proba'):
                    prob_pos = model.predict_proba(final_input)[0][1]
                else:
                    prob_pos = float(prediction)
                
                # 5. 显示结果
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
    <h2 style="color: #ef233c !important; margin-top:0;">⚠️ High Risk Detected (高风险)</h2>
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
    <h2 style="color: #2a9d8f !important; margin-top:0;">✅ Low Risk (低风险)</h2>
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
                st.write("Input Data Columns:", final_input.columns.tolist())
    else:
        st.info("👈 请在左侧输入数据并点击 'Run Prediction'")
# ------ 2. 分析 ------
# ------ 2. 分析 (修复版：解决 'dict' object has no attribute 'named_steps') ------
with tab_explain:
    st.markdown("### 🔍 Feature Importance")
    
    # 1. 获取加载的对象
    loaded_object = models[selected_model_name]
    
    # 2. 提取真正的模型 (关键修复步骤)
    model = None
    if loaded_object is not None:
        if isinstance(loaded_object, dict):
            # 优先查找 'pipeline'，因为你的报错显示键名是这个
            if 'pipeline' in loaded_object:
                model = loaded_object['pipeline']
            else:
                # 如果不是 pipeline，尝试找其他常见的键
                for key in ['model', 'clf', 'classifier', 'estimator']:
                    if key in loaded_object:
                        model = loaded_object[key]
                        break
        else:
            # 如果不是字典，说明它本身就是模型
            model = loaded_object

    # 3. 开始绘图
    if model:
        try:
            importances = None
            
            # --- A. 获取特征重要性数值 ---
            # 尝试从 Pipeline 中获取最后一步的分类器
            if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
                classifier = model.named_steps['clf']
            elif hasattr(model, 'steps'):
                # 如果没有名为 'clf' 的步骤，取最后一步
                classifier = model.steps[-1][1]
            else:
                # 如果不是 Pipeline，直接就是分类器
                classifier = model

            # 根据模型类型提取系数
            if not is_rf:
                # === 逻辑回归 (Logistic Regression) ===
                if hasattr(classifier, 'coef_'):
                    importances = classifier.coef_[0]
                else:
                    st.warning("⚠️ 无法从逻辑回归模型中提取系数 (coef_)")
                
                feature_names = FEATURES_LR # 10个特征
                color_scale = 'RdBu_r'
            else:
                # === 随机森林 (Random Forest) ===
                if hasattr(classifier, 'feature_importances_'):
                    importances = classifier.feature_importances_
                else:
                    st.warning("⚠️ 无法从随机森林模型中提取重要性 (feature_importances_)")
                
                feature_names = FEATURES_RF # 14个特征
                color_scale = 'Viridis'

            # --- B. 生成图表 ---
            if importances is not None:
                # 检查特征数量是否匹配
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
                    st.error(f"❌ 特征数量不匹配: 模型有 {len(importances)} 个系数，但定义的列表有 {len(feature_names)} 个。")
                    st.write("模型期望的特征数:", len(importances))
                    st.write("当前列表:", feature_names)

        except Exception as e:
            st.error(f"❌ 绘图错误: {e}")
            st.info("提示：可能是模型结构复杂，无法自动提取 'clf' 层。")
    else:
        st.warning("无法加载模型对象，请检查 .pkl 文件。")

    st.divider()
    
    # --- 图片显示部分 (保持不变) ---
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix**")
        img_name = "Test_CM_Logistic.png" if not is_rf else "Test_CM_RandomForest.png"
        try:
            st.image(f"assets/{img_name}", use_container_width=True)
        except:
            st.warning("Missing Image (assets folder)")
    with c2:
        st.markdown("**ROC Curve**")
        try:
            st.image("assets/Test_ROC_Comparison.png", use_container_width=True)
        except:
            st.warning("Missing Image (assets folder)")
            
    st.markdown("**Metrics Comparison**")
    try:
        st.image("assets/Test_Metrics_Comparison.png", use_container_width=True)
    except:
        st.warning("Missing Image")

    st.markdown(HTML_ANALYSIS_REPORT, unsafe_allow_html=True)
# ------ 3. 关于 ------
with tab_about:
    st.markdown(HTML_ABOUT_SYSTEM, unsafe_allow_html=True)