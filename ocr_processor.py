"""
SkinLogic OCR 전성분 정밀 추출 및 DB 기반 한글 자모 오타 자동 교정 모듈
1. 이미지 2.5배 업스케일링 & 대비·선명도 전처리 (작은 한글 획 및 쉼표 인식 강화)
2. skinlogic.db 화장품 전성분 사전 연동 + 한글 자모(초·중·종성) 유사도 기반 오타 자동 교정
3. 쉼표 누락으로 붙어버린 성분 문자열 자동 분리
"""

import os
import io
import re
import sqlite3
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

try:
    from PIL import Image, ImageOps, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import easyocr
    READER = easyocr.Reader(['ko', 'en'], gpu=False)
except ImportError:
    READER = None

DB_PATH = "skinlogic.db"

# 1. 노이즈 및 주석 제거용 정규식
RE_PREFIX = re.compile(r"^(전성분|성분명|성분)\s*[:：]?\s*")
RE_PARENTHESIS = re.compile(r"\(.*?\)|\[.*?\]")  # (0.25ppm), (CI 77891) 등 부가 표기 제거
RE_SPECIAL_CHARS = re.compile(r"[*•·_~▪▶▷※]")

NOISE_PHRASES = (
    "전성분", "성분명", "제조업자", "책임판매업자", "제조번호", "사용기한",
    "용량", "화장품법", "주의사항", "효능", "효과", "사용법", "보관방법", "소비자상담"
)

# 2. 필수 표준 화장품 전성분 코어 사전 (DB 로드와 병합됨)
CORE_INGREDIENT_VOCAB: List[str] = [
    "정제수", "글리세린", "부틸렌글라이콜", "다이프로필렌글라이콜", "프로판다이올",
    "나이아신아마이드", "1,2-헥산다이올", "베타인", "에리스리톨", "하이드록시아세토페논",
    "글리세릴글루코사이드", "카보머", "알지닌", "아르지닌", "트로메타민", "아데노신",
    "암모늄아크릴로일다이메틸타우레이트/브이피코폴리머", "다이소듐이디티에이",
    "알란토인", "판테놀", "덱스판테놀", "글리세릴아크릴레이트/아크릴릭애씨드코폴리머",
    "글루코오스", "산양삼추출물", "소듐디엔에이", "하이드로제네이티드레시틴",
    "병풀추출물", "세라마이드엔피", "세라마이드엔에스", "세라마이드에이피", "세라마이드이오피",
    "에틸헥실글리세린", "글라이코스핑고리피드", "소듐하이알루로네이트", "히알루론산",
    "하이드롤라이즈드히알루론산", "소듐아세틸레이티드하이알루로네이트", "베타-글루칸",
    "스쿠알란", "시어버터", "카프릴릭/카프릭트라이글리세라이드", "다이메티콘",
    "세테아릴알코올", "스테아릭애씨드", "팔미틱애씨드", "토코페롤", "토코페릴아세테이트",
    "아시아티코사이드", "마데카소사이드", "마데카식애씨드", "아시아틱애씨드",
    "티트리잎추출물", "티트리잎오일", "약모밀추출물", "어성초추출물", "녹차추출물",
    "살리실릭애씨드", "베타인살리실레이트", "글라이콜릭애씨드", "락틱애씨드", "시트릭애씨드",
    "글루코노락톤", "락토바이오닉애씨드", "레티놀", "레티날", "바쿠치올",
    "아스코빅애씨드", "아스코빌글루코사이드", "3-O-에틸아스코빅애씨드",
    "글루타치온", "알부틴", "알파-알부틴", "트라넥사믹애씨드",
    "징크옥사이드", "티타늄디옥사이드", "잔탄검", "펜틸렌글라이콜", "카프릴릴글라이콜",
    "페녹시에탄올", "클로페네신", "폴리글리세릴-10라우레이트", "폴리솔베이트20", "폴리솔베이트60",
    "피이지-60하이드로제네이티드캐스터오일", "소르비탄올리베이트", "세테아릴올리베이트",
    "자작나무수액", "자작나무껍질추출물", "알로에베라잎추출물", "비피다발효용해물",
    "락토바실러스발효물", "갈락토미세스발효여과물", "프로폴리스추출물", "꿀추출물"
]

# 한글 자모 분해 테이블 (초성 19, 중성 21, 종성 28)
CHOSUNG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
JUNGSUNG = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
JONGSUNG = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']


def to_jamo(text: str) -> str:
    """한글 음절을 초성/중성/종성 자모로 분해하여 시각적 오타(ㅌ↔ㄷ, ㅋ↔ㅂ 등) 유사도 비교를 정밀화"""
    out = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            offset = code - 0xAC00
            cho = offset // 588
            jung = (offset % 588) // 28
            jong = offset % 28
            out.append(CHOSUNG[cho] + JUNGSUNG[jung] + JONGSUNG[jong])
        else:
            out.append(ch)
    return "".join(out)


def jamo_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    ja = to_jamo(a)
    jb = to_jamo(b)
    return SequenceMatcher(None, ja, jb).ratio()


def build_ingredient_dictionary() -> Tuple[Dict[str, str], Dict[int, List[Tuple[str, str]]]]:
    """
    skinlogic.db의 실제 화장품 전성분과 코어 사전을 통합하여
    - norm_key -> 표준 성분명 매핑 사전
    - 글자 길이별 (norm_key, 표준 성분명) 인덱스
    를 구축합니다.
    """
    vocab_set = set(CORE_INGREDIENT_VOCAB)

    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5.0)
            cur = conn.cursor()
            cur.execute("SELECT raw_ingredients_text FROM products WHERE category = 'cosmetic' AND raw_ingredients_text IS NOT NULL")
            for (raw_txt,) in cur.fetchall():
                cleaned = RE_PARENTHESIS.sub("", str(raw_txt))
                for part in re.split(r"[,;，、\n]+", cleaned):
                    ing = re.sub(r"\s+", "", part).strip(" .:-")
                    if 2 <= len(ing) <= 35 and re.search(r"[가-힣]{2,}", ing):
                        vocab_set.add(ing)
            conn.close()
        except Exception:
            pass

    exact_map: Dict[str, str] = {}
    by_length: Dict[int, List[Tuple[str, str]]] = {}

    for canonical in sorted(vocab_set, key=lambda x: (-len(x), x)):
        key_full = canonical.replace(" ", "")
        exact_map[key_full] = canonical
        by_length.setdefault(len(key_full), []).append((key_full, canonical))

        # 슬래시(/)가 포함된 복합 성분은 슬래시 제거 버전도 인덱싱하여 OCR 누락 시 복원
        if "/" in key_full:
            key_noslash = key_full.replace("/", "")
            if key_noslash not in exact_map:
                exact_map[key_noslash] = canonical
                by_length.setdefault(len(key_noslash), []).append((key_noslash, canonical))

    return exact_map, by_length


ING_EXACT_MAP, ING_BY_LENGTH = build_ingredient_dictionary()


def preprocess_image_for_ocr(image_bytes: bytes) -> bytes:
    """
    작은 한글 획(틸, 콜, 톨 등)과 쉼표(,) 인식률을 높이기 위해
    흑백 대비 최적화 + 2.5배 고해상도 업스케일링 + 선명화(Sharpen)를 수행합니다.
    """
    if not HAS_PIL:
        return image_bytes
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size

        # 가로 1800px 미만 이미지는 2.5배 확대하여 작은 글씨와 쉼표를 선명하게 만듦
        if w < 1800:
            scale = min(2.8, max(2.0, 2200.0 / max(1, w)))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = gray.filter(ImageFilter.SHARPEN)

        buf = io.BytesIO()
        gray.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return image_bytes


def find_best_dictionary_match(token: str, min_sim: float = 0.78) -> Tuple[str, float]:
    """단일 토큰을 전성분 사전과 대조하여 정확 매칭 또는 한글 자모 유사도 기반 오타 교정"""
    clean = token.replace(" ", "")
    if not clean:
        return ("", 0.0)

    if clean in ING_EXACT_MAP:
        return (ING_EXACT_MAP[clean], 1.0)

    t_len = len(clean)
    if t_len < 3:
        return (clean, 0.0)

    best_canonical = clean
    best_score = 0.0

    # 길이가 비슷한(-1 ~ +1) 사전 후보군만 빠르게 비교
    for cand_len in (t_len, t_len - 1, t_len + 1):
        for cand_key, canonical in ING_BY_LENGTH.get(cand_len, []):
            # 첫 글자 또는 끝 글자 자모가 완전히 다르면 스킵하여 속도 및 정확도 향상
            sim = jamo_similarity(clean, cand_key)
            if sim > best_score:
                best_score = sim
                best_canonical = canonical

    if best_score >= min_sim:
        return (best_canonical, best_score)
    return (clean, best_score)


def segment_glued_ingredients(glued_text: str) -> List[str]:
    """
    EasyOCR이 쉼표를 놓쳐 여러 성분이 한 문자열로 붙은 경우
    (예: '정제수글리세린부딪런글라이볼', '글리세렉골루코사이드카보머')
    전성분 사전을 이용해 개별 성분 단위로 분리하고 오타를 자동 교정합니다.
    """
    s = glued_text.replace(" ", "").strip(" .:-")
    if not s:
        return []

    # 전체 문자열 자체가 하나의 성분과 일치하거나 매우 유사한 경우 바로 반환
    whole_match, whole_sim = find_best_dictionary_match(s, min_sim=0.80)
    if whole_sim >= 0.80:
        return [whole_match]

    results: List[str] = []
    pos = 0
    n = len(s)
    unmatched_buf = ""

    while pos < n:
        best_win_len = 0
        best_word = ""
        best_rank_score = -1.0

        max_win = min(28, n - pos)
        for win_len in range(max_win, 1, -1):
            sub = s[pos : pos + win_len]

            # 1) 정확 매칭 우선 (긴 성분명 우선 선택)
            if sub in ING_EXACT_MAP:
                best_win_len = win_len
                best_word = ING_EXACT_MAP[sub]
                best_rank_score = 2.0 + (win_len * 0.01)
                break

            # 2) 4글자 이상 구간에 대해 자모 유사도 교정 매칭
            if win_len >= 4:
                matched_canon, sim = find_best_dictionary_match(sub, min_sim=0.80)
                if sim >= 0.80:
                    rank_sc = sim + (win_len * 0.005)
                    if rank_sc > best_rank_score:
                        best_rank_score = rank_sc
                        best_win_len = win_len
                        best_word = matched_canon

        if best_win_len > 0:
            if unmatched_buf:
                if len(unmatched_buf) >= 2 and re.search(r"[가-힣]{2,}", unmatched_buf):
                    corr_buf, _ = find_best_dictionary_match(unmatched_buf, min_sim=0.76)
                    results.append(corr_buf)
                unmatched_buf = ""
            results.append(best_word)
            pos += best_win_len
        else:
            unmatched_buf += s[pos]
            pos += 1

    if unmatched_buf and len(unmatched_buf) >= 2 and re.search(r"[가-힣]{2,}", unmatched_buf):
        corr_buf, _ = find_best_dictionary_match(unmatched_buf, min_sim=0.76)
        results.append(corr_buf)

    return results


def merge_wrapped_ocr_lines(raw_lines: List[str]) -> str:
    """줄바꿈으로 끊긴 한글 성분을 결합하고 '1, 2-헥산다이올'의 쉼표 분리 오류를 사전 보정합니다."""
    cleaned_lines = [line.strip() for line in (raw_lines or []) if str(line).strip()]
    if not cleaned_lines:
        return ""

    merged = cleaned_lines[0]
    for next_line in cleaned_lines[1:]:
        if merged.endswith((",", ";", "，", "、")):
            merged += " " + next_line
        elif next_line.startswith((",", ";", "，", "、")):
            merged += next_line
        elif re.search(r"[가-힣0-9\-]$", merged) and re.match(r"^[가-힣0-9]", next_line):
            merged += next_line
        else:
            merged += " " + next_line

    # '1, 2-핵산다이올' 또는 '1,2-핵산다이올'이 쉼표 분리로 쪼개지지 않도록 사전 정규화
    merged = re.sub(r"1\s*[,.\s]\s*2\s*-\s*", "1,2-", merged)
    return merged


def clean_text_chunk(text: str) -> str:
    cleaned = RE_PREFIX.sub("", text)
    cleaned = RE_PARENTHESIS.sub("", cleaned)
    cleaned = RE_SPECIAL_CHARS.sub(" ", cleaned)
    return cleaned.strip()


def extract_and_normalize_tokens(raw_texts: List[str]) -> List[str]:
    """
    OCR 인식 원문 블록을 결합한 뒤 쉼표 분리 + 붙은 성분 사전 분할 + 한글 자모 오타 교정을 수행합니다.
    """
    if not raw_texts:
        return []

    combined_text = merge_wrapped_ocr_lines(raw_texts)
    cleaned_text = clean_text_chunk(combined_text)

    # 1차: 쉼표/세미콜론 기준 분리
    primary_chunks = re.split(r"[,;，、\n]+", cleaned_text)

    deduped: List[str] = []
    seen = set()

    for chunk in primary_chunks:
        chunk_clean = re.sub(r"\s+", " ", chunk).strip(" .:-")
        if len(chunk_clean) < 2:
            continue
        if any(noise in chunk_clean for noise in NOISE_PHRASES):
            continue
        if not re.search(r"[가-힣]{2,}", chunk_clean):
            continue

        # 공백으로 구분된 하위 토큰들도 각각 검사
        sub_space_tokens = chunk_clean.split(" ")
        for st in sub_space_tokens:
            st_clean = st.strip(" .:-")
            if len(st_clean) < 2 or not re.search(r"[가-힣]{2,}", st_clean):
                continue

            # 쉼표 누락으로 길게 붙은 경우 사전 기반 분할 및 오타 교정 실행
            extracted_list = segment_glued_ingredients(st_clean)
            for item in extracted_list:
                if len(item) >= 2 and item not in seen:
                    seen.add(item)
                    deduped.append(item)

    return deduped


def extract_ingredients_from_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    이미지 바이트 스트림을 고해상도 전처리한 뒤 EasyOCR로 전성분을 추출·교정합니다.
    """
    if READER is None:
        return {
            "error": "easyocr 모듈이 설치되어 있지 않습니다.",
            "raw_text_blocks": [],
            "extracted_ingredients": [],
            "total_count": 0
        }

    try:
        enhanced_bytes = preprocess_image_for_ocr(image_bytes)
        results = READER.readtext(
            enhanced_bytes,
            detail=0,
            paragraph=False,
            width_ths=0.35,
            mag_ratio=1.2
        )
        tokens = extract_and_normalize_tokens(results)
        return {
            "raw_text_blocks": results,
            "extracted_ingredients": tokens,
            "total_count": len(tokens)
        }
    except Exception as e:
        return {
            "error": str(e),
            "raw_text_blocks": [],
            "extracted_ingredients": [],
            "total_count": 0
        }


def process_label_image(img_path: str) -> List[str]:
    """
    main.py에서 호출하는 표준 진입점 함수:
    한글 파일명 경로 오류 없이 바이너리로 읽은 뒤 전처리 + OCR + 사전 오타 교정을 거쳐 반환합니다.
    """
    if not img_path or not os.path.exists(img_path):
        return []

    with open(img_path, "rb") as f:
        image_bytes = f.read()

    parsed = extract_ingredients_from_image(image_bytes)
    return parsed.get("extracted_ingredients", [])