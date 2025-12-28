import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="Urban Design AI Assistant",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 美化
st.markdown("""
    <style>
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        background-color: #FF4B4B; 
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 模型加载
# ==========================================
@st.cache_resource
def load_toolkit():
    try:
        model = joblib.load('best_model_mlp.pkl')
        scaler_x = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        return model, scaler_x, scaler_y
    except FileNotFoundError:
        return None, None, None

model, scaler_x, scaler_y = load_toolkit()

# ==========================================
# 3. 侧边栏：精准参数输入
# ==========================================
with st.sidebar:
    st.title("🎛️ Design Parameters")
    st.info("Ranges calibrated to Singapore dataset.")
    
    # --- Group 1: Density & Massing ---
    st.subheader("1. Density & Massing")
    # FAR: Min 4.0 - Max 6.99
    FAR = st.slider('FAR (Floor Area Ratio)', 4.0, 7.0, 5.5, step=0.1)
    # BCR: Min 0.08 - Max 0.18 (Very sensitive!)
    BCR = st.slider('BCR (Building Coverage)', 0.05, 0.20, 0.12, step=0.01)
    # OSR: Min 0.12 - Max 0.23
    OSR = st.slider('OSR (Open Space Ratio)', 0.10, 0.25, 0.16, step=0.01)
    
    # --- Group 2: Height & Form ---
    st.subheader("2. Height & Form")
    # AH: Min 99 - Max 231
    AH = st.slider('AH (Ave Height)', 90.0, 240.0, 162.0, step=1.0)
    # SD: Min 39 - Max 170
    SD = st.slider('SD (Height Std Dev)', 35.0, 175.0, 107.0, step=1.0)
    # BESA: Min 1.3 - Max 2.5 (CRITICAL FIX!)
    BESA = st.slider('BESA (Energy Surface)', 1.0, 3.0, 1.86, step=0.1)
    
    # --- Group 3: Street & Orientation ---
    st.subheader("3. Orientation & Sky")
    # OR: Min -45 - Max 45
    OR = st.slider('OR (Orientation)', -45.0, 45.0, 0.0, step=5.0)
    # SVF: Min 0.45 - Max 0.68
    SVF = st.slider('SVF (Sky View Factor)', 0.40, 0.70, 0.55, step=0.01)
    # AS: Min 25 - Max 66
    AS = st.slider('AS (Aspect Ratio)', 20.0, 70.0, 45.0, step=1.0)

    # --- Group 4: Advanced Geometry ---
    st.subheader("4. Advanced Geometry")
    # AAR: Min 1.23 - Max 2.78
    AAR = st.slider('AAR (Ave Aspect Ratio)', 1.0, 3.0, 1.92, step=0.1)
    # xAAR: Min 0.9 - Max 2.5
    xAAR = st.slider('xAAR (X-Aspect Ratio)', 0.8, 2.6, 1.63, step=0.1)
    # yAAR: Min 1.22 - Max 3.46
    yAAR = st.slider('yAAR (Y-Aspect Ratio)', 1.0, 3.5, 2.21, step=0.1)
    # SF: Min 0.08 - Max 0.12 (Very small range)
    SF = st.slider('SF (Shape Factor)', 0.05, 0.15, 0.09, step=0.01)
    # APR: Min 8.77 - Max 13.83
    APR = st.slider('APR (Area Perim Ratio)', 8.0, 14.0, 11.3, step=0.1)

    st.markdown("---")
    predict_btn = st.button("🚀 Run Simulation")

# ==========================================
# 4. 主界面逻辑
# ==========================================

st.title("🏙️ AI-Driven Urban Design Support System")
st.markdown("### Real-time Prediction: Thermal Comfort & Energy")

# Check Model
if model is None:
    st.error("❌ Model files not found! Please check .pkl files.")
    st.stop()

# 收集输入 (顺序严格匹配 X_train)
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
    'BESA': BESA, # 关键修复
    'SF': SF,
    'APR': APR
}
input_df = pd.DataFrame([input_data])

# ==========================================
# 5. 预测与结果 (结果精度优化)
# ==========================================

if predict_btn:
    with st.spinner('Calculating...'):
        time.sleep(0.3) 

        try:
            # 预测流程
            input_scaled = scaler_x.transform(input_df)
            pred_scaled = model.predict(input_scaled)
            pred_original = scaler_y.inverse_transform(pred_scaled)
            
            # 提取结果 [aveUTCI, stdUTCI, ATEC]
            utci_val = pred_original[0][0]
            std_utci_val = pred_original[0][1]
            atec_val = pred_original[0][2]

            # --- 结果展示区 ---
            st.subheader("📊 Simulation Results")
            col1, col2, col3 = st.columns(3)

            # UTCI (范围 32.3 - 32.8) -> 这是一个极热的环境
            with col1:
                # 调整了阈值，因为数据本身就在 32 以上
                if utci_val > 32.6:
                    status_msg = "🔥 Extreme Heat"
                    status_color = "inverse"
                else:
                    status_msg = "🟠 High Heat"
                    status_color = "normal"
                
                st.metric(
                    label="🌡️ aveUTCI (Comfort)",
                    value=f"{utci_val:.4f} °C", # 增加小数位以便看到变化
                    delta=status_msg,
                    delta_color=status_color
                )
                # 进度条基于 min/max 归一化
                st.progress((utci_val - 32.0) / 1.5)

            # stdUTCI
            with col2:
                st.metric(
                    label="📉 stdUTCI (Variation)",
                    value=f"{std_utci_val:.4f}", # 增加小数位
                    help="Lower is better (more uniform)"
                )
                st.progress(std_utci_val / 6.0)

            # ATEC (范围 111 - 113)
            with col3:
                if atec_val > 113.0:
                    e_msg = "⚠️ High Energy"
                    e_color = "inverse"
                else:
                    e_msg = "✅ Efficient"
                    e_color = "normal"

                st.metric(
                    label="⚡ ATEC (Energy)",
                    value=f"{atec_val:.4f}", # 增加小数位
                    delta=e_msg,
                    delta_color=e_color
                )
                st.progress((atec_val - 110) / 5.0)

            # --- 调试信息 (可选，如果不放心可以取消注释) ---
            # st.write("Debug - Raw Input:", input_df)

            # --- 智能建议 ---
            st.divider()
            st.subheader("💡 AI Diagnostics")
            
            suggestions = []
            
            # 基于数据分布的建议逻辑
            if utci_val > 32.6:
                suggestions.append(f"• **High Heat Stress ({utci_val:.2f}°C):** Your design is in the upper percentile of heat stress. Consider increasing street ventilation (lower 'AS' or adjust 'OR').")
            
            if atec_val > 113.2:
                suggestions.append(f"• **Energy Intensity:** ATEC is high ({atec_val:.2f}). Check if 'AH' (Height) or 'BESA' is too high.")

            if SVF < 0.5 and utci_val > 32.5:
                suggestions.append("• **Low Sky View:** Low SVF is trapping heat. Try increasing setbacks to let heat escape.")

            if not suggestions:
                st.success("✅ The design performs within the optimal range of the current dataset.")
            else:
                for s in suggestions:
                    st.warning(s)

        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("👈 Please adjust parameters on the sidebar and click **'Run Simulation'**.")
    with st.expander("ℹ️ Dataset Context"):
        st.write("""
        **Note on Data Range:** This model is trained on a highly specific high-density urban dataset.
        - **aveUTCI** typically varies between **32.3°C and 32.8°C**.
        - **ATEC** typically varies between **111 and 114**.
        
        *Even small changes in the decimal points represent significant physical impacts in this context.*
        """)
