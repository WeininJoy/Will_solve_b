import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import base64
from PIL import Image
import matplotlib as mpl
mpl.use("agg")

def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="380" height="390" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


# -- Figure type list
parameter_list = ['a&b match', 'code of Weining', 'logA_SR', 'N_star', 'log10f_i', 'omega_k', 'H0']

# Title the app
st.title('chi_eff and PPS of curved universe')

st.markdown("""
 * Use the menu at left to select which parameters to vary
 * The standard parameter set: logA_SR=np.log(20), N_star=55, log10f_i=5, omega_k=-0.005, H0=70
 * Your plots will appear below
""")


#-- Select which parameter to vary
select_event = st.sidebar.selectbox('Which parameter you would like to choose?', parameter_list)

if select_event == parameter_list[0]:
    st.markdown("""
    * w_value_list = np.linspace(-0.4, -0.99, num=5)
    """)
    st.subheader("Primordial power spectrum:")
    image = Image.open('PPS_primpy/PPS_primpy_standard(w=-1).png')
    st.image(image, caption='PPS')

    st.subheader("chi_eff to w (when a&b match), N_star=55 (standard)")
    image = Image.open('chi_eff/chi_diff_w_Nstar55.png')
    st.image(image, caption='chi_eff')

    st.subheader("chi_eff to w (when a&b match), N_star=50")
    image = Image.open('chi_eff/chi_diff_w_Nstar50.png')
    st.image(image, caption='chi_eff')

    st.subheader("chi_eff to w (when a&b match), N_star=60")
    image = Image.open('chi_eff/chi_diff_w_Nstar60.png')
    st.image(image, caption='chi_eff')


elif select_event == parameter_list[1]:
    st.markdown("""
    * The PPS is output from PPS_will_b.py. 
    """)
    st.subheader("Primordial power spectrum:")
    image = Image.open('PPS_will_b/PPS_will_b.png')
    st.image(image, caption='PPS')

    st.subheader("chi to t , t=0.0~2.0(SR_start), use chi from PPS_will_b.py")
    image = Image.open('chi_eff/chi_PPS_will_b.png')
    st.image(image, caption='chi')

    st.subheader("chi to t , t=0.0~2.0(SR_start), use chi_eff from example_planck_likelihood.py")
    image = Image.open('chi_eff/chi_diff_will_b.png')
    st.image(image, caption='chi_eff')

elif select_event == parameter_list[2]:
    st.markdown("""
    * logA_SR_list = np.linspace(2.5, 3.7, num=5)
    """)
    st.subheader("Primordial power spectrum:")
    image = Image.open('PPS_primpy/PPS_logA_SR_2_5-3_7.png')
    st.image(image, caption='PPS')

    st.subheader("chi_eff to logA_SR")
    image = Image.open('chi_eff/chi_diff_logA_SR.png')
    st.image(image, caption='chi_eff')

elif select_event == parameter_list[3]:
    st.markdown("""
    * N_star_list = np.linspace(50, 60, num=5)
    """)
    st.subheader("Primordial power spectrum:")
    image = Image.open('PPS_primpy/PPS_N_star_50-60.png')
    st.image(image, caption='PPS')

    st.subheader("chi_eff to Nstar")
    image = Image.open('chi_eff/chi_diff_Nstar.png')
    st.image(image, caption='chi_eff')

elif select_event == parameter_list[4]:
    st.markdown("""
    * log10f_i_list = np.linspace(-1, 5, num=5)
    """)
    st.subheader("Primordial power spectrum:")
    image = Image.open('PPS_primpy/PPS_log10f_i_-1-5.png')
    st.image(image, caption='PPS')

    st.subheader("chi_eff to log10f_i")
    image = Image.open('chi_eff/chi_diff_log10f_i.png')
    st.image(image, caption='chi_eff')

elif select_event == parameter_list[5]:
    st.markdown("""
    * omega_k_list = np.linspace(-0.04, -0.001, num=5)
    """)
    st.subheader("Primordial power spectrum:")
    image = Image.open('PPS_primpy/PPS_omega_k_04-001.png')
    st.image(image, caption='PPS')

    st.subheader("chi_eff to omega_k")
    image = Image.open('chi_eff/chi_diff_omega_k.png')
    st.image(image, caption='chi_eff')

elif select_event == parameter_list[6]:
    st.markdown("""
    * H0_list = np.linspace(67, 74, num=5)
    """)
    st.subheader("Primordial power spectrum:")
    image = Image.open('PPS_primpy/PPS_H0_6_7-7_4.png')
    st.image(image, caption='PPS')

    st.subheader("chi_eff to H0")
    image = Image.open('chi_eff/chi_diff_H0.png')
    st.image(image, caption='chi_eff')

