"""
AI News Summarizer - Streamlit Web Application
==============================================
Interactive web interface for bilingual news summarization and translation.
Supports Indonesian and English articles with automatic language detection.
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.language_detector import detect_language, get_language_display_name, LanguageDetectionError
from src.summarizer import NewsSummarizer
from src.translator import NewsTranslator
from src.preprocessing import clean_news_text

# Sample news articles for 1-click testing
SAMPLE_ARTICLES = {
    "Bahasa Indonesia": (
        "Presiden Joko Widodo meresmikan operasional penuh Moda Raya Terpadu (MRT) Fase 1 di Jakarta. "
        "Proyek transportasi massal berbasis rel modern ini membentang sepanjang 16 kilometer dari Bundaran Hotel Indonesia "
        "hingga ke Lebak Bulus di Jakarta Selatan. Dalam pidato peresmiannya, Presiden menegaskan bahwa kehadiran MRT "
        "menandai era baru peradaban transportasi di ibu kota. Masyarakat diimbau untuk beralih dari penggunaan kendaraan pribadi "
        "ke transportasi publik guna mengurangi kemacetan parah dan menekan polusi udara yang telah lama melanda wilayah Jabodetabek. "
        "Pemerintah Provinsi DKI Jakarta dan Kementerian Perhubungan berencana mempercepat pembangunan Fase 2 menuju Kota Tua."
    ),
    "English": (
        "The European Space Agency has officially launched its groundbreaking solar observation satellite from the spaceport in French Guiana. "
        "The spacecraft is equipped with ten cutting-edge scientific instruments designed to capture unprecedented close-up images "
        "of the Sun's polar regions and analyze the intense solar wind that impacts space weather across the solar system. "
        "Scientists believe that understanding how the Sun creates and controls the heliosphere is crucial for protecting communication satellites, "
        "power grids, and astronauts during future missions to the Moon and Mars. The mission is scheduled to operate for at least seven years, "
        "collaborating closely with NASA's Parker Solar Probe."
    ),
}

st.set_page_config(
    page_title="AI News Summarizer (Bilingual)",
    page_icon="📰",
    layout="wide",
)


@st.cache_resource(show_spinner="Memuat model summarizer...")
def get_summarizer():
    return NewsSummarizer()


@st.cache_resource(show_spinner="Memuat modul penerjemah...")
def get_translator():
    return NewsTranslator()


def main():
    st.title("📰 AI News Summarizer (Bilingual ID / EN)")
    st.markdown(
        "Aplikasi peringkas berita otomatis berbasis **mT5-small** dengan deteksi bahasa otomatis "
        "(Bahasa Indonesia & English) dan fitur terjemahan terpisah."
    )

    # Sidebar: Model and Hardware Info
    with st.sidebar:
        st.header("⚙️ Informasi Sistem")
        st.info(
            "**Model:** `google/mt5-small`\n\n"
            "**Supported Languages:**\n"
            "- 🇮🇩 Bahasa Indonesia (`summarize Indonesian:`)\n"
            "- 🇬🇧 English (`summarize English:`)\n\n"
            "**Arsitektur:** Seq2Seq Encoder-Decoder\n\n"
            "**Dataset:** XL-Sum (BBC News)"
        )

        st.subheader("📝 Contoh Berita Cepat")
        chosen_sample = st.selectbox("Pilih contoh artikel:", ["-- Pilih --", "Bahasa Indonesia", "English"])
        if st.button("Muat Contoh"):
            if chosen_sample in SAMPLE_ARTICLES:
                st.session_state["article_input"] = SAMPLE_ARTICLES[chosen_sample]

        st.markdown("---")
        st.subheader("💡 Petunjuk")
        st.markdown(
            "1. Masukkan teks berita pada kolom input.\n"
            "2. Bahasa akan otomatis terdeteksi.\n"
            "3. Klik **Ringkas Berita**.\n"
            "4. Ringkasan akan tetap dalam bahasa asli.\n"
            "5. Anda dapat menerjemahkan hasil ringkasan ke bahasa lawan secara terpisah."
        )

    # Initialize session state keys
    if "article_input" not in st.session_state:
        st.session_state["article_input"] = ""
    if "summary_result" not in st.session_state:
        st.session_state["summary_result"] = None
    if "translation_result" not in st.session_state:
        st.session_state["translation_result"] = None

    # Main Input Area
    st.subheader("1. Masukkan Teks Artikel Berita")
    input_text = st.text_area(
        label="Tempelkan teks berita di sini:",
        value=st.session_state["article_input"],
        height=200,
        placeholder="Ketik atau tempel artikel berita dalam Bahasa Indonesia atau English...",
        key="main_text_area",
    )

    # Quick language detection preview
    col_det1, col_det2 = st.columns([2, 1])
    with col_det1:
        if input_text.strip():
            try:
                detected_code, conf = detect_language(input_text)
                disp_name = get_language_display_name(detected_code)
                flag = "🇮🇩" if detected_code == "id" else "🇬🇧"
                st.caption(f"🔍 Bahasa Terdeteksi: **{flag} {disp_name}** (Confidence: {conf*100:.1f}%)")
            except Exception as e:
                st.caption(f"🔍 Bahasa Terdeteksi: *{str(e)}*")
        else:
            st.caption("🔍 Bahasa Terdeteksi: *Menunggu input teks...*")

    # Action Buttons: Summarize & Clear
    col_btn1, col_btn2, _ = st.columns([1.5, 1, 5])
    with col_btn1:
        summarize_clicked = st.button("🚀 Ringkas Berita", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ Bersihkan", use_container_width=True):
            st.session_state["article_input"] = ""
            st.session_state["summary_result"] = None
            st.session_state["translation_result"] = None
            st.rerun()

    # Process Summarization
    if summarize_clicked:
        clean_input = clean_news_text(input_text)
        if not clean_input:
            st.error("Harap masukkan teks berita terlebih dahulu.")
        else:
            try:
                summarizer = get_summarizer()
                with st.spinner("Sedang memproses ringkasan berita..."):
                    result = summarizer.summarize(clean_input)
                    st.session_state["summary_result"] = result
                    st.session_state["translation_result"] = None
            except Exception as e:
                st.error(f"Gagal membuat ringkasan: {str(e)}")

    # Display Summary Result
    if st.session_state["summary_result"]:
        res = st.session_state["summary_result"]
        st.markdown("---")
        st.subheader("2. Hasil Ringkasan Berita")

        flag = "🇮🇩" if res["detected_language"] == "id" else "🇬🇧"

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bahasa", f"{flag} {res['language_name']}")
        m2.metric("Kata Artikel Asli", f"{res['input_word_count']} kata")
        m3.metric("Kata Ringkasan", f"{res['summary_word_count']} kata")
        m4.metric("Rasio Kompresi", res["compression_ratio"])

        # Summary box
        st.success(f"**Ringkasan ({res.get('engine_used', 'auto')}):**\n\n{res['summary']}")

        # Translation Section (Strictly Separated Process)
        st.markdown("---")
        st.subheader("3. Fitur Terjemahan (Terpisah)")
        st.caption(
            "Sesuai prinsip arsitektur, proses peringkasan dan penerjemahan adalah proses terpisah. "
            "Anda dapat menerjemahkan hasil ringkasan di atas ke bahasa lawan."
        )

        target_lang = "en" if res["detected_language"] == "id" else "id"
        target_name = "English" if target_lang == "en" else "Bahasa Indonesia"
        target_flag = "🇬🇧" if target_lang == "en" else "🇮🇩"

        if st.button(f"🌐 Terjemahkan Ringkasan ke {target_flag} {target_name}"):
            with st.spinner(f"Menerjemahkan ke {target_name}..."):
                try:
                    translator = get_translator()
                    trans_res = translator.translate(
                        res["summary"],
                        target_lang=target_lang,
                        source_lang=res["detected_language"],
                    )
                    st.session_state["translation_result"] = trans_res
                except Exception as e_trans:
                    st.error(f"Penerjemahan gagal: {str(e_trans)}")

        if st.session_state["translation_result"]:
            trans = st.session_state["translation_result"]
            st.info(f"**Terjemahan ({target_flag} {trans['target_name']}):**\n\n{trans['translated_text']}")


if __name__ == "__main__":
    main()
