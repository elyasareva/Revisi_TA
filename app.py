import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler hanya jika diperlukan
def load_model():
    return (
        joblib.load('model_kmeans.pkl'),
        joblib.load('scaler.pkl'),
        joblib.load('label_map.pkl')
    )

# Konfigurasi halaman
st.set_page_config(page_title="Kualitas Air", layout="centered")

# Inisialisasi navigasi
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Navigasi Card
if st.session_state.page == 'home':

    # CSS untuk memperbesar tombol
    st.markdown("""
    <style>
    div.stButton > button {
        height: 150px;
        font-size: 24px;
        border-radius: 12px;
        width: 100%;
        padding: 1.5em;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏠 Selamat Datang")

    st.markdown("<h2 style='margin-bottom: 20px;'>Silakan pilih:</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📘 Tentang Aquaponic"):
            st.session_state.page = 'edukasi'

    with col2:
        if st.button("💧 Input Kualitas Air"):
            st.session_state.page = 'kualitas_air'

# Halaman Edukasi
elif st.session_state.page == 'edukasi':
    st.title("📘 Tentang Aquaponic")

    # CSS
    st.markdown("""
    <style>
    .card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .card:hover {
        background-color: #e6ecf3;
    }
    .card h3 {
        margin-top: 0;
        margin-bottom: 10px;
    }
    .card p {
        margin-bottom: 10px;
    }
    .link-bawah {
        margin-top: auto;
        font-weight: bold;
        color: #1f77b4;
        text-decoration: none;
    }
    .link-bawah:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### 📚 Pilih topik untuk dipelajari:")

    # BARIS 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card" style="color: #0b5394;">
            <h3>1. Apa itu Aquaponic?</h3>
            <p>Pelajari dasar-dasar aquaponik melalui video singkat.</p>
            <a class="link-bawah" href="https://youtu.be/eHAdvcepLaY?si=fCh9WDi6hrtIphJt" target="_blank">▶️ Tonton Video</a>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card" style="color: #38761d;">
            <h3>2. Cara Merawat Aquaponic</h3>
            <p>Langkah-langkah penting dalam perawatan sistem aquaponik.</p>
            <a class="link-bawah" href="https://csdt.org/culture/engineeredecosystems/ecology-maintenance.html" target="_blank">🌱 Baca Artikel</a>
        </div>
        """, unsafe_allow_html=True)

    # BARIS 2
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class="card" style="color: #3d85c6;">
            <h3>3. Ikan Terbaik untuk Aquaponic</h3>
            <p>Rekomendasi ikan yang cocok untuk ekosistem aquaponik.</p>
            <a class="link-bawah" href="https://gogreenaquaponics.com/blogs/news/what-are-the-best-fish-for-aquaponics" target="_blank">🐟 Lihat Daftar Ikan</a>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="card" style="color: #6aa84f;">
            <h3>4. Tanaman Terbaik untuk Aquaponic</h3>
            <p>Tanaman yang mudah tumbuh dan memberikan hasil maksimal.</p>
            <a class="link-bawah" href="https://gogreenaquaponics.com/blogs/news/what-are-the-best-plants-for-aquaponics" target="_blank">🌿 Lihat Daftar Tanaman</a>
        </div>
        """, unsafe_allow_html=True)

    # BARIS 3 - Card Tengah
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        st.markdown("""
        <div class="card" style="color: #76a5af;">
            <h3>5. Menjaga Keseimbangan Kualitas Air</h3>
            <p>Tips penting menjaga air tetap sehat bagi ikan dan tanaman.</p>
            <a class="link-bawah" href="https://gogreenaquaponics.com/blogs/news/maintenance-checklist-for-aquaponics-systems" target="_blank">💧 Cek Tips Perawatan</a>
        </div>
        """, unsafe_allow_html=True)

    # Spasi + Tombol
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("⬅️ Kembali", on_click=lambda: st.session_state.update(page='home'))


# Halaman Input Kualitas Air
elif st.session_state.page == 'kualitas_air':
    st.title("💧 Kualitas Air")
    kmeans, scaler, label_map = load_model()

    with st.form("input_form"):
        st.subheader("Input Data")
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1, format="%.1f")
        suhu = st.number_input("Suhu (°C)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
        tds = st.number_input("TDS (mg/L)", min_value=0.0, step=0.1, format="%.1f")

        submitted = st.form_submit_button("SUBMIT")

    if submitted:
        input_data = pd.DataFrame([[ph, suhu, tds]], columns=['ph', 'temperature', 'tds'])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]
        kualitas = label_map[cluster]

        warna = {
            'Baik': 'green',
            'Cukup': 'orange',
            'Buruk': 'red'
        }

        st.markdown(f"""
        <div style="padding: 1em; background-color: {warna[kualitas]}; color: white; border-radius: 8px; text-align: center;">
            <h4>Condition:</h4>
            <h2><b>{kualitas.upper()}</b></h2>
        </div>
        """, unsafe_allow_html=True)

        if kualitas == 'Buruk':
            st.error("⚠️ Kualitas air **BURUK** berdasarkan hasil model.")
        elif kualitas == 'Cukup':
            st.warning("⚠️ Kualitas air **CUKUP** berdasarkan hasil model.")
        elif kualitas == 'Baik':
            st.success("✅ Kualitas air **BAIK** berdasarkan hasil model.")

    st.button("⬅️ Kembali", on_click=lambda: st.session_state.update(page='home'))
