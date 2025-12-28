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
    st.markdown("Adjust parameters to simulate:")
    
    # 分组 1: 形态与密度 (Morphology)
    st.subheader("1. Morphology & Density")
    FAR = st.slider('FAR (Floor Area Ratio)', 0.0, 10.0, 2.5)
    BCR = st.slider('BCR (Building Coverage)', 0.0, 1.0, 0.4)
    OSR = st.slider('OSR (Open Space Ratio)', 0.0, 1.0, 0.3)
    AH = st.slider('AH (Ave Height)', 0.0, 100.0, 30.0)
    SD = st.slider('SD (Standard Deviation of Height)', 0.0, 50.0, 10.0)
    
    # 分组 2: 街道与朝向 (Street & Orientation)
    st.subheader("2. Street & Orientation")
    OR = st.slider('OR (Orientation)', 0.0, 180.0, 45.0, help="Street Orientation in degrees")
    SVF = st.slider('SVF (Sky View Factor)', 0.0, 1.0, 0.5)
    AS = st.slider('AS (Aspect Ratio)', 0.0, 5.0, 1.5)
    
    # 分组 3: 建筑表面与其他 (Facade & Others)
    st.subheader("3. Facade & Advanced Metrics")
    AAR = st.slider('AAR (Ave Aspect Ratio)', 0.0, 5.0, 1.0)
    xAAR = st.slider('xAAR (Aspect Ratio X)', 0.0, 5.0, 1.0)
    yAAR = st.slider('yAAR (Aspect Ratio Y)', 0.0, 5.0, 1.0)
    BESA = st.slider('BESA (Building Energy Surface)', 0.0, 5000.0, 1000.0)
    SF = st.slider('SF (Shape Factor)', 0.0, 1.0, 0.5)
    APR = st.slider('APR (Area Perimeter Ratio)', 0.0, 50.0, 10.0)

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
# ⚠️【重要】：顺序必须严格对应 X_train 的列顺序！
input_data = {
    'FAR': FAR,
    'BCR': BCR,
    'OSR': OSR,
    'AS': AS,
    'AH': AH,
    'OR': OR,
    'SD': SD,
    'SVF': SVF,
    'AAR': AAR,
    'xAAR': xAAR,
    'yAAR': yAAR,
    'BESA': BESA,
    'SF': SF,
    'APR': APR
}

# 转换为 DataFrame
input_df = pd.DataFrame([input_data])

# ==========================================
# 5. 预测与结果展示 (Prediction & Visualization)
# ==========================================

if predict_btn:
    with st.spinner('Calculating physics...'):
        time.sleep(0.5) # 模拟计算延迟

        try:
            # 1. 数据标准化
            input_scaled = scaler_x.transform(input_df)
            
            # 2. 模型预测
            pred_scaled = model.predict(input_scaled)
            
            # 3. 逆标准化 (还原为真实物理量)
            pred_original = scaler_y.inverse_transform(pred_scaled)
            
            # ---------------------------------------------------------
            # ⚠️【关键修改】：根据您提供的 Index(['aveUTCI', 'stdUTCI', 'ATEC']) 映射结果
            # ---------------------------------------------------------
            utci_val = pred_original[0][0]      # Index 0: aveUTCI
            std_utci_val = pred_original[0][1]  # Index 1: stdUTCI (可选展示)
            atec_val = pred_original[0][2]      # Index 2: ATEC
            # ---------------------------------------------------------

            # --- 结果展示区 ---
            st.subheader("📊 Simulation Results")
            
            col1, col2, col3 = st.columns(3) # 增加一列展示 stdUTCI

            # 结果 1: 热舒适度 (aveUTCI)
            with col1:
                # 动态颜色判定
                if utci_val > 32:
                    status_color = "inverse"
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
                # 简单的进度条可视化 (假设范围 20-40)
                st.progress(min(max((utci_val - 20) / 20, 0.0), 1.0))

            # 结果 2: 舒适度波动 (stdUTCI) - 新增
            with col2:
                st.metric(
                    label="📉 Temp Variation (stdUTCI)",
                    value=f"{std_utci_val:.2f}",
                    help="Standard Deviation of UTCI. Lower means more uniform comfort."
                )
                # 简单的进度条 (假设范围 0-5)
                st.progress(min(std_utci_val / 5.0, 1.0))

            # 结果 3: 能耗 (ATEC)
            with col3:
                # 动态逻辑
                if atec_val > 150: 
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
                # 简单的进度条 (假设最大能耗 300)
                st.progress(min(atec_val / 300, 1.0))

            # --- 智能建议区 ---
            st.divider()
            st.subheader("💡 AI Design Analysis")
            
            suggestions = []
            
            # 逻辑 1: 舒适度与 SVF
            if utci_val > 30 and input_data['SVF'] < 0.3:
                suggestions.append(f"• The UTCI is high ({utci_val:.1f}°C). Considering the low Sky View Factor ({input_data['SVF']}), try **increasing street openness** to facilitate heat dissipation.")
            
            # 逻辑 2: 舒适度与遮阳
            if utci_val > 30 and input_data['SVF'] > 0.7:
                suggestions.append(f"• High solar exposure detected (SVF {input_data['SVF']}). Consider **reducing SVF** (adding shading) to lower the temperature.")
                
            # 逻辑 3: 能耗与密度 (注意这里改成了 FAR)
            if atec_val > 140 and input_data['FAR'] > 4.0:
                suggestions.append(f"• Energy consumption is high due to extreme density (FAR {input_data['FAR']}). Ensure sufficient spacing between buildings.")

            # 逻辑 4: 覆盖率 (注意这里改成了 BCR)
            if input_data['BCR'] > 0.6 and std_utci_val > 2.0:
                 suggestions.append(f"• High Building Coverage ({input_data['BCR']}) might be causing uneven thermal distribution (High stdUTCI).")

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
    
    with st.expander("ℹ️ About the Model"):
        st.write("""
        This model was trained on a dataset of urban morphologies using MLP Regressor.
        - **Inputs:** 14 morphological parameters (FAR, BCR, SVF, etc.)
        - **Outputs:** aveUTCI, stdUTCI, and ATEC.
        """)
