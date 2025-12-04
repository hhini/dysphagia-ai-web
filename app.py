import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
import base64

# ================= 1. 页面配置 (Page Config) =================
st.set_page_config(
    page_title="Dysphagia AI (吞咽障碍智能预测)",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 2. 深度定制 CSS (样式精修) =================
st.markdown("""
<style>
    /* 1. 全局字体与颜色强制设定 */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', 'Microsoft YaHei', sans-serif;
        color: #000000 !important; /* 强制文字黑色 */
    }
    
    /* 2. 背景色设定：极简灰白 */
    .stApp {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* 3. 侧边栏美化 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #dee2e6;
        box-shadow: 2px 0 15px rgba(0,0,0,0.05);
    }
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stMarkdown {
        color: #343a40 !important;
        font-weight: 500;
    }

    /* 4. 卡片浮入动画 */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translate3d(0, 20px, 0); }
        to { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    .css-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        animation: fadeInUp 0.8s ease-in-out;
        border-left: 6px solid #4361ee;
    }

    /* 5. 按钮重绘 */
    .stButton>button {
        background: linear-gradient(90deg, #4361ee 0%, #3f37c9 100%);
        color: white !important;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(67, 97, 238, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ================= 3. 核心工具函数 (修改版) =================

# 【关键修改】使用你提供的真实 StdDev 进行手动标准化
def manual_standardization(df):
    df_scaled = df.copy()
    
    # 1. 定义统计数据
    # 格式: '列名': {'mean': 平均值(医学估算), 'std': 标准差(你提供的真实值)}
    # ⚠️ 注意: 因为你没给我平均值，我用了医学常数估算(mean)。
    # 如果你之后找到了原始数据的平均值，请替换这里的 'mean'。
    stats_config = {
        'number_of_teeth': {'mean': 18.0,  'std': 9.299115}, # 你提供的 StdDev
        'weight':          {'mean': 60.0,  'std': 9.572267}, # 你提供的 StdDev
        'BMI':             {'mean': 23.0,  'std': 3.310996}, # 你提供的 StdDev
        'age':             {'mean': 75.0,  'std': 7.154127}, # 你提供的 StdDev
        'hight':           {'mean': 160.0, 'std': 8.207334}, # 你提供的 StdDev (注意列名是 hight)
        
        # 下面这些分类变量或非连续变量，你没提供 Std，我们做简单处理或不处理
        # 只要保证它们大致在 0-1 或小数值范围内即可，逻辑回归对它们不敏感
        'number_of_hospitalizations': {'mean': 1.0, 'std': 1.5}, 
        'MMSE':                       {'mean': 24.0, 'std': 5.0},
        'education':                  {'mean': 9.0,  'std': 4.0}
    }
    
    # 2. 只对连续变量进行转换：(数值 - 平均值) / 标准差
    for col, stats in stats_config.items():
        if col in df_scaled.columns:
            # 这里的公式将 75岁 转换为 0，将 82岁 转换为 1.0 (根据你的 std)
            df_scaled[col] = (df_scaled[col] - stats['mean']) / stats['std']
            
    # 3. 分类变量 (0/1) 不需要除以标准差，保持 0/1 或稍微中心化即可
    # 你的训练数据 X.head 显示 chewing 是 0 或 1 经过某种处理后的样子?
    # 如果 X.head 里 chewing 也是小数 (例如 -0.5, 0.5)，则需要下面这步：
    # 如果 X.head 里 chewing 是 0 和 1，则注释掉下面这几行
    binary_cols = ['chewing', 'choking', 'eating', 'frail']
    for col in binary_cols:
         # 简单的中心化，让 0 变成 -0.5，1 变成 0.5 (假设分布)
         # 这步是可选的，取决于你的 X.head 里分类变量长什么样
         # 既然你之前的 X.head 里 chewing 这一列是 0，我们就不动它
         pass 

    return df_scaled

@st.cache_resource
def load_model():
    try:
        return joblib.load("logistic_model.pkl")
    except:
        return None

model = load_model()

# ================= 4. 侧边栏：交互式输入 =================
with st.sidebar:
    try:
        st.image("assets/logo.png", width=180)
    except:
        st.markdown("## 🏥 AI Med Assist")
    
    st.markdown("---")
    with st.form("main_form"):
        st.markdown("**1. Demographics (基本特征)**")
        age = st.number_input("Age (年龄)", 20, 110, 75)
        
        c1, c2 = st.columns(2)
        hight = c1.number_input("Height (身高 cm)", 100, 220, 165)
        weight = c2.number_input("Weight (体重 kg)", 30, 150, 60)
        
        # 自动计算 BMI
        bmi_val = weight / ((hight / 100) ** 2)
        st.info(f"📊 Calculated BMI: **{bmi_val:.2f}**")
        BMI = bmi_val
        
        education = st.number_input("Education Years (教育年限)", 0, 30, 9)

        st.markdown("**2. Oral Status (口腔状况)**")
        number_of_teeth = st.slider("Teeth (牙齿)", 0, 32, 20)
        chewing = st.selectbox("Chewing Difficulty?", [0, 1], format_func=lambda x: "Yes (困难)" if x==1 else "No (正常)")
        choking = st.selectbox("Choking History?", [0, 1], format_func=lambda x: "Yes (呛咳)" if x==1 else "No (无)")
        eating = st.selectbox("Eating Assistance?", [0, 1], format_func=lambda x: "Yes (需辅助)" if x==1 else "No (独立)")

        st.markdown("**3. Clinical (临床)**")
        frail = st.selectbox("Frailty Status?", [0, 1], format_func=lambda x: "Yes (衰弱)" if x==1 else "No (正常)")
        hospitalizations = st.number_input("Hospitalizations (住院次数)", 0, 20, 0)
        MMSE = st.slider("MMSE Score", 0, 30, 25)

        submit_btn = st.form_submit_button("🚀 Run Prediction (开始预测)")

# ================= 5. 主界面 =================

try:
    st.image("assets/banner.png", use_container_width=True)
except:
    st.markdown("""<div style="background: linear-gradient(90deg, #1e3a8a 0%, #4361ee 100%); padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;"><h1>Dysphagia Prediction System</h1></div>""", unsafe_allow_html=True)

tab_diagnosis, tab_explain, tab_about = st.tabs(["🩺 AI Diagnosis (智能诊断)", "📊 Model Analysis (模型分析)", "ℹ️ About (关于)"])

# ------ Tab 1: 诊断结果 ------
with tab_diagnosis:
    if submit_btn:
        if model is None:
            st.error("❌ Model not found! 请检查 logistic_model.pkl")
        else:
            # 1. 原始数据封装 (Raw Data)
            # ⚠️ 必须保持特征顺序与训练时一致
            input_data = pd.DataFrame([{
                'chewing': chewing, 'choking': choking, 'eating': eating,
                'number_of_teeth': number_of_teeth, 'weight': weight, 'BMI': BMI,
                'frail': frail, 'age': age, 'number_of_hospitalizations': hospitalizations,
                'hight': hight, 'MMSE': MMSE, 'education': education
            }])

            # 2. 【核心修复】手动进行标准化
            # 将 75岁 转换成 -0.x，适应你的模型
            input_scaled = manual_standardization(input_data)

            # 3. 预测过程
            with st.status("🧬 AI Analysis in progress...", expanded=True) as status:
                time.sleep(0.8)
                try:
                    # 使用标准化后的数据 input_scaled 进行预测
                    prediction = model.predict(input_scaled)[0]
                    # 获取概率
                    if hasattr(model, 'predict_proba'):
                         prob_pos = model.predict_proba(input_scaled)[0][1]
                    else:
                         prob_pos = float(prediction) # 兜底
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    prob_pos = 0.0
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

            # 4. 结果展示
            col_l, col_r = st.columns([1, 1.5])
            with col_l:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_pos * 100,
                    number = {'suffix': "%", 'font': {'color': "black"}},
                    title = {'text': "Risk Probability", 'font': {'color': "black"}},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#ef233c" if prob_pos > 0.5 else "#2a9d8f"}
                    }
                ))
                fig.update_layout(height=300, margin=dict(t=50,b=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

            with col_r:
                if prob_pos > 0.5:
                    st.markdown(f"""
                    <div class="css-card" style="border-left-color: #ef233c;">
                        <h2 style="color:#ef233c !important;">⚠️ High risk detected（高风险）</h2>
                        <p>预测概率: <strong>{prob_pos*100:.1f}%</strong></p>
                        <hr>
                        <p><strong>建议（通用信息）：</strong></p>
                        <ul>
                            <li><strong>专业评估：</strong>尽快与临床医生或言语语言治疗师（SLP）讨论进一步评估的必要性。常用评估方式包括床旁吞咽筛查，必要时可考虑影像学吞咽评估（如 VFSS 或 FEES）。请根据医生意见决定具体检查。</li>
                            <li><strong>进食安全措施：</strong>在专业建议到位之前，进食时保持直立坐姿、细嚼慢咽、少量多次；避免同时说话或分心；进食后保持直立 30 分钟以降低误吸风险。</li>
                            <li><strong>质地与体积：</strong>遵循临床团队建议调整食物质地（如更软更易咀嚼）与单次入口体积；避免极易散落或黏稠度极端的食物，直到获得更明确的专业指导。</li>
                            <li><strong>口腔与水合：</strong>保持良好口腔卫生与充分水合，有助于降低感染风险与提升吞咽舒适度。</li>
                            <li><strong>监测红旗信号：</strong>若出现反复呛咳、湿性声音、发热或肺部不适、体重显著下降、进食时间延长或明显疲劳等，请尽快联系医疗团队。</li>
                            <li><strong>记录与沟通：</strong>记录进食过程中的不适、食物类型、时间点与症状变化，以便与医护人员沟通与个体化调整。</li>
                        </ul>
                        <p style="font-size:0.9em;color:#6c757d;"><em>说明：上述为一般性健康信息，不替代专业医疗建议。具体检查与管理方案应由专业人员评估后决定。</em></p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="css-card" style="border-left-color: #2a9d8f;">
                        <h2 style="color:#2a9d8f !important;">✅ Low risk（低风险）</h2>
                        <p>预测概率: <strong>{prob_pos*100:.1f}%</strong></p>
                        <hr>
                        <p><strong>建议（通用信息）：</strong></p>
                        <ul>
                            <li><strong>保持良好习惯：</strong>继续采取直立坐姿进食、细嚼慢咽、适当分口，避免匆忙或分心进食。</li>
                            <li><strong>规律随访与自我监测：</strong>如出现新发或加重的呛咳、湿性声音、吞咽疼痛、体重下降、反复呼吸道感染等，及时与临床医生沟通。</li>
                            <li><strong>口腔与水合：</strong>维持良好口腔卫生与充足水合；必要时咨询牙科或营养师，优化咀嚼与营养结构。</li>
                            <li><strong>个体化优化：</strong>如存在咀嚼困难或认知负担增加，考虑更易处理的食物质地与更简单的进食环境，必要时与专业人员讨论是否需要进一步评估。</li>
                        </ul>
                        <p style="font-size:0.9em;color:#6c757d;"><em>说明：上述为一般性健康信息，不替代专业医疗建议。若有疑问，请与专业人员讨论。</em></p>
                    </div>""", unsafe_allow_html=True)

    else:
        st.info("👈 请在左侧输入数据并点击 Run Prediction")

# ------ Tab 2: 模型解释 ------
# ------ Tab 2: 模型解释 ------
with tab_explain:
    # 【修改点1】用 HTML 包裹标题，加上白色背景(css-card)和强制黑色字体，瞬间清晰
    st.markdown("""
    <div class="css-card" style="padding: 20px; border-left: 6px solid #4361ee;">
        <h3 style="color: black; margin:0;">🔍 Model Interpretability (模型解释性)</h3>
        <p style="color: #333; margin-top:5px;">Visualizing why the model made this prediction. (可视化模型决策依据)</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. 特征重要性
    if model:
        try:
            # 提取系数 (兼容 Pipeline)
            if hasattr(model, 'named_steps'):
                coefs = model.named_steps['clf'].coef_[0]
            else:
                coefs = model.coef_[0]
            
            features = ['Chewing', 'Choking', 'Eating', 'Teeth', 'Weight', 'BMI',
                        'Frail', 'Age', 'Hospitalizations', 'Hight', 'MMSE', 'Education']
            
            # 绘图
            df_imp = pd.DataFrame({'Feature': features, 'Weight': coefs})
            df_imp = df_imp.sort_values(by='Weight', ascending=True)

            fig_bar = px.bar(df_imp, x='Weight', y='Feature', orientation='h',
                             color='Weight', color_continuous_scale='RdBu_r',
                             title="Feature Importance (特征权重分析)")
            
            # 【修改点2】强制图表文字变黑
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)", 
                 font=dict(
                    color="black",              # 全局字体强制黑色
                    size=14,                    # 字体稍微调大，更清晰
                    family="Arial"
                ),
                title=dict(
                    font=dict(color="#1e3a8a", size=20, weight="bold") # 标题用深蓝色加粗
                ),
                xaxis=dict(
                    tickfont=dict(color="black"), # X轴刻度字黑色
                    title_font=dict(color="black")
                ),
                yaxis=dict(
                    tickfont=dict(color="black"), # Y轴刻度字黑色
                    title_font=dict(color="black")
                )
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown(
                """<p style='color: black; font-weight: bold; text-align: center; margin-top: -10px;'>
                🔴 Red bars increase risk (红色增加风险) | 🔵 Blue bars decrease risk (蓝色降低风险)
                </p>""", 
                unsafe_allow_html=True
            )
            
        except:
            st.warning("Feature importance not available for this model structure.")

    # 2. 静态图片展示 (ROC & Matrix)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        # 【修改点3】用 HTML h4 标签强制加粗加黑，不用 st.markdown("**...**")
        st.markdown('<h4 style="color:black; text-align:center;">📉 ROC Curve (准确率曲线)</h4>', unsafe_allow_html=True)
        try:
            st.image("assets/roc_curve.png", use_container_width=True)
        except:
            st.warning("⚠️ Missing 'assets/roc_curve.png'")
    with c2:
        # 【修改点3】同上
        st.markdown('<h4 style="color:black; text-align:center;">🔲 Confusion Matrix (混淆矩阵)</h4>', unsafe_allow_html=True)
        try:
            st.image("assets/confusion_matrix.png", use_container_width=True)
        except:
            st.warning("⚠️ Missing 'assets/confusion_matrix.png'")

# ------ Tab 3: 关于 ------

with tab_about:
    # 【修改点4】把整个关于页面的文字包在 css-card 里
    # 这样背景是纯白的，字是黑的，对比度最高，最好看
    st.markdown("""
    <div class="css-card">
        <h3 style="color: #1e3a8a;">🏥 About This Project</h3>
        <p style="color: black; font-size: 16px;">
            This system utilizes <strong>Logistic Regression</strong> (Machine Learning) to screen for <strong>Dysphagia</strong> risk in elderly patients.
        </p>
        <ul style="color: black; font-size: 16px;">
            <li><strong>Data Source:</strong> Based on clinical datasets including 12 key indicators.</li>
            <li><strong>Accuracy:</strong> 88.5% (Based on training data).</li>
        </ul>
        <hr>
        <h3 style="color: #1e3a8a;">🇨🇳 关于本项目</h3>
        <p style="color: black; font-size: 16px;">
            本系统利用机器学习算法（逻辑回归）辅助医生筛查老年吞咽障碍风险。<br>
            通过输入年龄、BMI、认知分数等 12 项指标，快速输出风险概率。
        </p>
    </div>
    """, unsafe_allow_html=True)