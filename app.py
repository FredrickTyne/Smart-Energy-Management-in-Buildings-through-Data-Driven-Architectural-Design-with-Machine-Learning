import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ==========================================
# 1. 页面配置 (Page Configuration)
# ==========================================
st.set_page_config(
    page_title="Urban Design AI Assistant",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 以实现更现代的视觉效果
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1 {
        color: #2c3e50;
    }
    h2, h3 {
        color: #34495e;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 模型加载 (Model Loading)
# ==========================================
@st.cache_resource
def load_toolkit():
    """
    加载训练好的模型和标准化器。
    使用了缓存装饰器，避免每次交互都重新加载文件。
    """
    try:
        model = joblib.load('best_model_mlp.pkl')
        scaler_x = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        return model, scaler_x, scaler_y
    except FileNotFoundError:
        return None, None, None

model, scaler_x, scaler_y = load_toolkit()

# ==========================================
# 3. 侧边栏：参数输入 (Sidebar Inputs)
# ==========================================
with st.sidebar:
    st.title("🎛️ Design Parameters")
    st.markdown("Adjust the architectural parameters below to simulate performance.")
    st.markdown("---")

    # ⚠️【重要】：请根据您 df.columns 的实际特征名称和顺序修改这里
    # 这里的名字必须和您训练时的特征名字一模一样
    
    # 1. 密度与形态
    st.subheader("Morphology")
    FAR = st.slider('Floor Area Ratio (FAR)', 0.5, 8.0, 2.5, help="Building density")
    BuildingCov = st.slider('Building Coverage Ratio', 0.1, 0.9, 0.4)
    AveHeight = st.slider('Average Height (m)', 10.0, 150.0, 45.0)
    
    # 2. 开放度与朝向
    st.subheader("Openness & Climate")
    SVF = st.slider('Sky View Factor (SVF)', 0.1, 1.0, 0.5, help="Visibility of the sky")
    Orientation = st.selectbox('Street Orientation', [0, 45, 90, 135], index=2, help="0=N-S, 90=E-W")
    
    # 3. 绿化与反照率 (如果有这些特征的话，没有请删除)
    # GreenRatio = st.slider('Greenery Ratio', 0.0, 1.0, 0.3)
    # Albedo = st.slider('Surface Albedo', 0.1, 0.8, 0.3)

    st.markdown("---")
    predict_btn = st.button("🚀 Run Simulation")

# ==========================================
# 4. 主界面逻辑 (Main Interface)
# ==========================================

# 标题区域
st.title("🏙️ AI-Driven Urban Design Support System")
st.markdown("### Real-time Prediction of Thermal Comfort & Energy Efficiency")
st.markdown("This tool utilizes a **Multi-Layer Perceptron (MLP)** neural network to assist architects in early-stage decision making.")
st.divider()

# 检查模型是否加载成功
if model is None:
    st.error("❌ Model files not found! Please ensure 'best_model_mlp.pkl', 'scaler_X.pkl', and 'scaler_y.pkl' are in the same directory.")
    st.stop()

# 收集输入数据
# ⚠️【重要】：这里的 key (e.g., 'FAR') 必须和上面 slider 的变量名对应，且顺序必须与训练集一致！
input_data = {
    'FAR': FAR,
    'BuildingCov': BuildingCov,
    'AveHeight': AveHeight,
    'SVF': SVF,
    'Orientation': Orientation,
    # 'GreenRatio': GreenRatio, # 如果有的话
    # 'Albedo': Albedo          # 如果有的话
}

# 转换为 DataFrame
input_df = pd.DataFrame([input_data])

# ==========================================
# 5. 预测与结果展示 (Prediction & Visualization)
# ==========================================

if predict_btn:
    with st.spinner('Calculating physics...'):
        time.sleep(0.5) # 模拟一点计算延迟，增加交互感

        try:
            # 1. 数据标准化
            input_scaled = scaler_x.transform(input_df)
            
            # 2. 模型预测
            pred_scaled = model.predict(input_scaled)
            
            # 3. 逆标准化 (还原为真实物理量)
            pred_original = scaler_y.inverse_transform(pred_scaled)
            
            # 提取结果 (假设输出顺序是: [0]=UTCI, [1]=ATEC)
            # 如果您的输出顺序不一样，请在这里交换索引
            utci_val = pred_original[0][0]
            atec_val = pred_original[0][1]

            # --- 结果展示区 ---
            st.subheader("📊 Simulation Results")
            
            col1, col2 = st.columns(2)

            # 结果 1: 热舒适度 (UTCI)
            with col1:
                # 动态颜色判定
                if utci_val > 32:
                    status_color = "inverse" # 红色/强调
                    status_msg = "🔥 High Heat Stress"
                elif utci_val < 20:
                    status_color = "normal"
                    status_msg = "❄️ Cold Stress"
                else:
                    status_color = "normal" 
                    status_msg = "✅ Comfortable"
                
                st.metric(
                    label="🌡️ Thermal Comfort (aveUTCI)",
                    value=f"{utci_val:.2f} °C",
                    delta=status_msg,
                    delta_color=status_color
                )
                st.progress(min(max((utci_val + 20) / 70, 0.0), 1.0)) # 简单的进度条可视化

            # 结果 2: 能耗 (ATEC)
            with col2:
                # 动态逻辑
                if atec_val > 150: # 假设阈值，需根据您数据调整
                    energy_msg = "⚠️ High Consumption"
                    energy_color = "inverse"
                else:
                    energy_msg = "🌿 Energy Efficient"
                    energy_color = "normal"

                st.metric(
                    label="⚡ Energy Consumption (ATEC)",
                    value=f"{atec_val:.2f} kWh/m²",
                    delta=energy_msg,
                    delta_color=energy_color
                )
                st.progress(min(atec_val / 300, 1.0)) # 假设最大能耗300

            # --- 智能建议区 ---
            st.divider()
            st.subheader("💡 AI Design Analysis")
            
            # 这里可以写一些简单的基于规则的逻辑
            suggestions = []
            
            if utci_val > 30 and input_data['SVF'] < 0.3:
                suggestions.append(f"• The UTCI is high ({utci_val:.1f}°C). Considering the low Sky View Factor ({input_data['SVF']}), try **increasing street openness** to facilitate heat dissipation.")
            
            if utci_val > 30 and input_data['SVF'] > 0.7:
                suggestions.append(f"• High solar exposure detected (SVF {input_data['SVF']}). Consider **adding shading devices or trees** to reduce direct radiation.")
                
            if atec_val > 140 and input_data['FAR'] > 4.0:
                suggestions.append(f"• Energy consumption is high due to extreme density (FAR {input_data['FAR']}). Ensure sufficient spacing between buildings.")

            if not suggestions:
                st.info("The current design configuration seems balanced based on the model's training data.")
            else:
                for s in suggestions:
                    st.warning(s)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Hint: Check if the feature names in 'app.py' match exactly with 'scaler_X.pkl'.")

else:
    # 初始状态提示
    st.info("👈 Please adjust parameters in the sidebar and click **'Run Simulation'** to see results.")
    
    # 显示示例数据分布（可选，增加专业感）
    with st.expander("ℹ️ About the Model"):
        st.write("""
        This model was trained on a dataset of urban morphologies using MLP Regressor (R² ≈ 0.94).
        - **Inputs:** Geometric and density parameters.
        - **Outputs:** Microclimate comfort metrics and building energy use.
        """)