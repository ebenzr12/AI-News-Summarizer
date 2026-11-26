"""
AI News Summarizer - Bilingual Extractive Summarizer
====================================================
Deterministic, low-latency extractive summarization engine for Indonesian
and English news articles using TF-IDF salience, inverted-pyramid lead-position
bias, numeric/entity density, and Maximal Marginal Relevance (MMR) redundancy removal.
"""

import re
from typing import Dict, List, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.config import SUMMARY_LENGTH_PRESETS
from src.preprocessing import clean_news_text, split_sentences

INDONESIAN_STOPWORDS: Set[str] = {
    "ada", "adalah", "adanya", "adapun", "agak", "agaknya", "agar", "akan",
    "akankah", "akhir", "akhiri", "akhirnya", "aku", "akulah", "amat",
    "amatlah", "anda", "andalah", "antar", "antara", "antaranya", "apa",
    "apaan", "apabila", "apakah", "apalagi", "apatah", "artinya", "asal",
    "asalkan", "atas", "atau", "ataukah", "ataupun", "awal", "awalnya",
    "bagai", "bagaikan", "bagaimana", "bagaimanakah", "bagaimanapun", "bagi",
    "bagian", "bahkan", "bahwa", "bahwasanya", "baik", "bakal", "bakalan",
    "balik", "banyak", "bapak", "baru", "bawah", "beberapa", "begini",
    "beginian", "beginikah", "beginilah", "begitu", "begitukah", "begitulah",
    "begitupun", "bekerja", "belakang", "belakangan", "belum", "belumlah",
    "benar", "benarkah", "benarlah", "berada", "berakhir", "berakhirlah",
    "berakhirnya", "berapa", "berapakah", "berapalah", "berapapun", "berarti",
    "berawal", "berbagai", "berdatangan", "beri", "berikan", "berikut",
    "berikutnya", "berjumlah", "berkali", "berkata", "berkehendak",
    "berkeinginan", "berkenaan", "berlainan", "berlalu", "berlangsung",
    "berlebihan", "bermacam", "bermaksud", "bermula", "bersama", "bersiap",
    "bertanya", "berturut", "bertutur", "berujar", "berupa", "besar",
    "betul", "betulkah", "biasa", "biasanya", "bila", "bilakah", "bisa",
    "bisakah", "boleh", "bolehkah", "bolehlah", "buat", "bukan", "bukankah",
    "bukanlah", "bukannya", "bulan", "bung", "cara", "caranya", "cukup",
    "cukupkah", "cukuplah", "cuma", "dahulu", "dalam", "dan", "dapat", "dari",
    "daripada", "datang", "dekat", "demi", "demikian", "demikianlah",
    "dengan", "depan", "di", "dia", "diakhiri", "diakhirinya", "dialah",
    "diantara", "diantaranya", "diberi", "diberikan", "diberikannya",
    "dibuat", "dibuatnya", "didapat", "didatangkan", "digunakan", "diibaratkan",
    "diibaratkannya", "diingat", "diingatkan", "diinginkan", "dijawab",
    "dijelaskan", "dijelaskannya", "dikarenakan", "dikatakan", "dikatakannya",
    "dikerjakan", "diketahui", "diketahuinya", "dikira", "dilakukan",
    "dilalui", "dilihat", "dimaksud", "dimaksudkan", "dimaksudkannya",
    "dimaksudnya", "diminta", "dimintai", "dimisalkan", "dimulai",
    "dimulailah", "dimulainya", "dimungkinkan", "dini", "dipastikan",
    "diperbuat", "diperbuatnya", "dipergunakan", "diperkirakan", "diperlihatkan",
    "diperlukan", "diperlukannya", "dipersoalkan", "dipertanyakan", "dipunyai",
    "diri", "dirinya", "disampaikan", "disebut", "disebutkan", "disebutkannya",
    "disini", "disinilah", "ditambahkan", "ditandaskan", "ditanya",
    "ditanyai", "ditanyakan", "ditegaskan", "ditujukan", "ditunjuk",
    "ditunjuki", "ditunjukkan", "ditunjukkannya", "ditunjuknya", "dituturkan",
    "dituturkannya", "diucapkan", "diucapkannya", "diungkapkan", "dong", "dua",
    "dulu", "empat", "enggak", "enggaknya", "entah", "entahlah", "guna",
    "gunakan", "hal", "hampir", "hanya", "hanyalah", "hari", "harus",
    "haruslah", "harusnya", "hendak", "hendaklah", "hendaknya", "hingga",
    "ia", "ialah", "ibarat", "ibaratkan", "ibaratnya", "ibu", "ikut", "ingat",
    "ingin", "inginkah", "inginkan", "ini", "inikah", "inilah", "itu",
    "itukah", "itulah", "jadi", "jadilah", "jadinya", "jangan", "jangankan",
    "janganlah", "jauh", "jawab", "jawaban", "jawabnya", "jelas", "jelaskan",
    "jelaslah", "jelasnya", "jika", "jikalau", "juga", "jumlah", "jumlahnya",
    "justru", "kala", "kalau", "kalaulah", "kalaupun", "kalian", "kami",
    "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah", "kapanpun",
    "karena", "karenanya", "kasus", "kata", "katakan", "katakanlah",
    "katanya", "ke", "keadaan", "kebetulan", "kecil", "kedua", "keduanya",
    "keinginan", "kelak", "kelihatan", "kelihatannya", "keluar", "kembali",
    "kemudian", "kemungkinan", "kemungkinannya", "kenapa", "kepada",
    "kepadanya", "kesampaian", "keseluruhan", "keseluruhannya", "keterlaluan",
    "ketika", "khususnya", "kini", "kinilah", "kira", "kiranya", "kita",
    "kitalah", "kok", "kurang", "lagi", "lagian", "lah", "lain", "lainnya",
    "lalu", "lama", "lamanya", "lanjut", "lanjutnya", "lebih", "lewat",
    "lima", "luar", "macam", "maka", "makanya", "makin", "malah", "malahan",
    "mampu", "mampukah", "mana", "manakala", "manalagi", "masa", "masalah",
    "masalahnya", "masih", "masihkah", "masing", "mau", "maupun", "melainkan",
    "melakukan", "melalui", "melihat", "melihatnya", "memang", "memastikan",
    "memberi", "memberikan", "membuat", "memerlukan", "memihak", "meminta",
    "memintakan", "memisalkan", "memperbuat", "mempergunakan", "memperkirakan",
    "memperlihatkan", "mempersiapkan", "mempersoalkan", "mempertanyakan",
    "mempunyai", "memulai", "memungkinkan", "menaiki", "menambahkan",
    "menandaskan", "menanti", "menantikan", "menanya", "menanyai",
    "menanyakan", "mendapat", "mendapatkan", "mendatang", "mendatangi",
    "mendatangkan", "menegaskan", "mengakhiri", "mengapa", "mengatakan",
    "mengatakannya", "mengenai", "mengerjakan", "mengetahui", "menggunakan",
    "menghendaki", "mengibaratkan", "mengibaratkannya", "mengingat",
    "mengingatkan", "menginginkan", "mengira", "mengucapkan", "mengucapkannya",
    "mengungkapkan", "menjadi", "menjawab", "menjelaskan", "menuju",
    "menunjuk", "menunjuki", "menunjukkan", "menunjuknya", "menurut",
    "menuturkan", "menyampaikan", "menyangkut", "menyatakan", "menyebutkan",
    "menyeluruh", "menyiapkan", "merasa", "mereka", "merekalah", "merupakan",
    "meski", "meskipun", "meyakini", "meyakinkan", "minta", "mirip", "misal",
    "misalkan", "misalnya", "mula", "mulai", "mulailah", "mulanya", "mungkin",
    "mungkinkah", "nah", "naik", "namun", "nanti", "nantinya", "nyaris",
    "nyatanya", "oleh", "olehnya", "pada", "padahal", "padanya", "pak",
    "paling", "panjang", "pantas", "para", "pasti", "pastilah", "penting",
    "pentingnya", "per", "percuma", "perlu", "perlukah", "perlunya", "pernah",
    "persoalan", "pertama", "pertanyaan", "pertanyakan", "pihak", "pihaknya",
    "pukul", "pula", "pun", "punya", "rasa", "rasanya", "rata", "rupanya",
    "saat", "saatnya", "saja", "sajalah", "saling", "sama", "sambil",
    "sampai", "sampaikan", "sana", "sangat", "sangatlah", "satu", "saya",
    "sayalah", "se", "sebab", "sebabnya", "sebagai", "sebagaimana",
    "sebagainya", "sebagian", "sebaik", "sebaiknya", "sebaliknya", "sebanyak",
    "sebegini", "sebegitu", "sebelum", "sebelumnya", "sebenarnya", "seberapa",
    "sebesar", "sebetulnya", "sebisanya", "sebuah", "sebut", "sebutlah",
    "sebutnya", "secara", "secukupnya", "sedang", "sedangkan", "sedemikian",
    "sedikit", "sedikitnya", "seenaknya", "segala", "segalanya", "segera",
    "seharusnya", "sehingga", "seingat", "sejak", "sejauh", "sejenak",
    "sejumlah", "sekadar", "sekadarnya", "sekali", "sekalian", "sekaligus",
    "sekalipun", "sekarang", "sekecil", "seketika", "sekiranya", "sekitar",
    "sekitarnya", "sekurang", "sekurangnya", "sela", "selain", "selaku",
    "selalu", "selama", "selamanya", "selanjutnya", "seluruh", "seluruhnya",
    "semacam", "semakin", "semampu", "semampunya", "semasa", "semasih",
    "semata", "semaunya", "sementara", "semisal", "semisalnya", "sempat",
    "semua", "semuanya", "semula", "sendiri", "sendirian", "sendirinya",
    "seolah", "seorang", "sepanjang", "sepantasnya", "sepantasnyalah",
    "seperlunya", "seperti", "sepertinya", "sepihak", "sering", "seringnya",
    "serta", "serupa", "sesaat", "sesama", "sesampai", "sesegera", "sesekali",
    "seseorang", "sesuatu", "sesuatunya", "sesudah", "sesudahnya", "setelah",
    "setempat", "setengah", "seterusnya", "setiap", "setiba", "setibanya",
    "setidaknya", "setinggi", "seusai", "sewaktu", "siap", "siapa", "siapakah",
    "siapapun", "sini", "sinilah", "soal", "soalnya", "suatu", "sudah",
    "sudahkah", "sudahlah", "supaya", "tadi", "tadinya", "tahu", "tahun",
    "tak", "tambah", "tambahnya", "tampak", "tampaknya", "tandas", "tandasnya",
    "tanpa", "tanya", "tanyakan", "tanyanya", "tapi", "tegas", "tegasnya",
    "telah", "tempat", "tengah", "tentang", "tentu", "tentulah", "tentunya",
    "tepat", "terakhir", "terasa", "terbanyak", "terdahulu", "terdapat",
    "terdiri", "terhadap", "terhadapnya", "teringat", "terjadi", "terjadilah",
    "terjadinya", "terkira", "terlalu", "terlebih", "terlihat", "termasuk",
    "ternyata", "tersampaikan", "tersebut", "tersebutlah", "tertentu",
    "tertuju", "terus", "terutama", "tetap", "tetapi", "tiap", "tiba",
    "tidak", "tidakkah", "tidaklah", "tiga", "tinggi", "toh", "tunjuk",
    "turut", "tutur", "tuturnya", "ucap", "ucapnya", "ujar", "ujarnya",
    "umum", "umumnya", "ungkap", "ungkapnya", "untuk", "usah", "usai",
    "waduh", "wah", "wahai", "waktu", "waktunya", "walau", "walaupun",
    "wong", "yaitu", "yakin", "yakni", "yang",
}

ENGLISH_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "cannot", "could", "couldn", "did", "didn", "do", "does",
    "doesn", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn", "has", "hasn", "have", "haven",
    "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "if", "in", "into", "is", "isn", "it", "its",
    "itself", "let", "ll", "me", "more", "most", "mustn", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "re",
    "same", "shan", "she", "should", "shouldn", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "ve", "very", "was", "wasn", "we",
    "were", "weren", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "won", "would", "wouldn", "you", "your",
    "yours", "yourself", "yourselves", "said",
}


class ExtractiveNewsSummarizer:
    """
    Fast, deterministic extractive summarizer tailored for Indonesian and English news.
    """

    def __init__(self, mmr_lambda: float = 0.72):
        self.mmr_lambda = mmr_lambda

    @staticmethod
    def _get_stopwords(lang_code: str) -> List[str]:
        if lang_code == "id":
            return sorted(INDONESIAN_STOPWORDS)
        return sorted(ENGLISH_STOPWORDS)

    @staticmethod
    def _determine_target_sentences(total_sentences: int, length: str) -> int:
        preset = SUMMARY_LENGTH_PRESETS.get(length, SUMMARY_LENGTH_PRESETS["balanced"])
        if total_sentences <= 1:
            return total_sentences
        if total_sentences == 2:
            return 1 if length == "concise" else 2

        ratio_target = round(total_sentences * preset["ratio"])
        bounded = max(preset["min_sentences"], min(preset["max_sentences"], ratio_target))
        # Ensure extractive summary is strictly shorter than multi-sentence input
        return min(bounded, max(1, total_sentences - 1))

    @staticmethod
    def _compute_heuristic_boost(sentence: str, idx: int, total: int) -> float:
        """
        Compute position (inverted pyramid), numeric fact density, and length suitability.
        """
        # Lead-position score: first sentences in news articles carry core who/what/where
        position_score = 1.0 - (idx / max(1, total))
        lead_bonus = 0.35 if idx == 0 else (0.18 if idx == 1 else 0.0)

        # Numeric & proper-noun cues (dates, amounts, percentages, acronyms)
        numbers_count = len(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", sentence))
        numeric_bonus = min(0.25, numbers_count * 0.08)

        # Ideal news sentence length (8 to 35 words)
        word_count = len(sentence.split())
        if word_count < 5:
            length_factor = 0.4
        elif word_count <= 38:
            length_factor = 1.0
        else:
            length_factor = 0.85

        return (0.45 * position_score + lead_bonus + numeric_bonus) * length_factor

    def summarize(
        self,
        text: str,
        language: str = "en",
        length: str = "balanced",
    ) -> str:
        """
        Produce a concise, non-redundant extractive summary preserving original language.
        """
        cleaned = clean_news_text(text)
        if not cleaned:
            raise ValueError("Teks artikel tidak boleh kosong.")

        sentences = split_sentences(cleaned)
        if not sentences:
            return cleaned
        if len(sentences) == 1:
            return sentences[0]

        target_k = self._determine_target_sentences(len(sentences), length)
        stopwords = self._get_stopwords(language)

        # Build TF-IDF matrix over sentences; fall back gracefully if vocabulary is empty
        tfidf_matrix = None
        base_scores = np.zeros(len(sentences), dtype=float)
        try:
            vectorizer = TfidfVectorizer(
                stop_words=stopwords,
                ngram_range=(1, 2),
                sublinear_tf=True,
                token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b",
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
            # Centroid similarity + mean TF-IDF salience
            centroid = np.asarray(tfidf_matrix.mean(axis=0))
            centroid_sim = cosine_similarity(tfidf_matrix, centroid).ravel()
            row_norms = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
            max_norm = row_norms.max() if row_norms.max() > 0 else 1.0
            base_scores = 0.65 * centroid_sim + 0.35 * (row_norms / max_norm)
        except ValueError:
            # Occurs if sentences contain only stopwords or punctuation
            tfidf_matrix = None

        heuristics = np.array(
            [self._compute_heuristic_boost(s, i, len(sentences)) for i, s in enumerate(sentences)],
            dtype=float,
        )

        # Normalize base_scores to [0, 1]
        if base_scores.max() > 0:
            base_scores = base_scores / base_scores.max()

        salience = 0.6 * base_scores + 0.4 * heuristics

        # MMR selection to balance salience and diversity
        sim_matrix = (
            cosine_similarity(tfidf_matrix)
            if tfidf_matrix is not None
            else np.eye(len(sentences))
        )

        selected_indices: List[int] = []
        remaining_indices = list(range(len(sentences)))

        while len(selected_indices) < target_k and remaining_indices:
            if not selected_indices:
                best_idx = max(remaining_indices, key=lambda idx: salience[idx])
            else:
                def mmr_score(idx: int) -> float:
                    redundancy = max(sim_matrix[idx, s_idx] for s_idx in selected_indices)
                    return self.mmr_lambda * salience[idx] - (1.0 - self.mmr_lambda) * redundancy

                best_idx = max(remaining_indices, key=mmr_score)

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Restore original narrative order
        selected_indices.sort()
        return " ".join(sentences[i] for i in selected_indices)
