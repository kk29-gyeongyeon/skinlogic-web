"""
SkinLogic OCR 전성분 정밀 추출 및 DB 기반 한글 자모 오타 자동 교정 모듈
1. EXIF 자동 회전 보정 + 메모리 안전 해상도 최적화(최대 1100px 제한) & 대비·선명도 전처리 (502 OOM 방지)
2. EasyOCR 좌표 기반 줄/열 읽기 순서 정렬 + '전성분' 구간 자동 발췌(앞뒤 설명·주의사항 노이즈 제거)
3. skinlogic.db 화장품 전성분 사전 연동 + 사전 계산된 한글 자모(초·중·종성) 유사도 기반 고속 오타 교정
4. 쉼표 누락으로 붙어버린 성분 문자열 자동 분리 + 동시 요청 직렬화 락(512MB OOM 방지)
"""

import os
import io
import re
import gc
import sqlite3
import threading
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

try:
    from PIL import Image, ImageOps, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import easyocr
    READER = easyocr.Reader(['ko', 'en'], gpu=False, quantize=True)
except Exception:
    try:
        import easyocr
        READER = easyocr.Reader(['ko', 'en'], gpu=False)
    except Exception:
        READER = None

# 512MB 클라우드 컨테이너에서 동시 OCR 요청으로 인한 PyTorch 메모리 폭증(502) 방지
OCR_LOCK = threading.Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOLUME_DB_PATH = "/app/data/skinlogic.db"
DB_PATH = VOLUME_DB_PATH if os.path.exists(VOLUME_DB_PATH) else os.path.join(BASE_DIR, "skinlogic.db")

# 1. 노이즈 및 주석 제거용 정규식
RE_PREFIX = re.compile(r"^(전성분|성분명|성분)\s*[:：\]\)\-]*\s*")
RE_PARENTHESIS = re.compile(r"\(.*?\)|\[.*?\]")  # (0.25ppm), (CI 77891) 등 부가 표기 제거
RE_SPECIAL_CHARS = re.compile(r"[*•·_~▪▶▷※■□◆◇★☆]")

# 전체 화면/라벨 촬영 시 '전성분' 구간만 자동 발췌하기 위한 시작·종료 앵커 정규식
RE_SECTION_START = re.compile(
    r"(?:화장품법에\s*따라\s*기재[·.,\s]*표시하여야\s*하는\s*모든\s*성분|"
    r"\[?\s*전\s*성\s*분\s*\]?|"
    r"모\s*든\s*성\s*분|"
    r"성\s*분\s*명|"
    r"INGREDIENTS?)\s*[:：\]\)\-]*\s*",
    re.IGNORECASE
)

RE_SECTION_END = re.compile(
    r"(?:사용\s*시(?:의)?\s*주의\s*사항|"
    r"사용할\s*때(?:의)?\s*주의\s*사항|"
    r"주의\s*사항|"
    r"화장품\s*제조\s*업자|"
    r"화장품\s*책임\s*판매\s*업자|"
    r"제조\s*판매\s*업자|"
    r"제조\s*업자|"
    r"책임\s*판매\s*업자|"
    r"소비자\s*상담\s*실|"
    r"품질\s*보증\s*기준|"
    r"보관\s*및\s*취급|"
    r"사용\s*기한|"
    r"제조\s*번호|"
    r"효능\s*[·.\s]*효과|"
    r"용법\s*[·.\s]*용량|"
    r"본\s*제품에\s*이상이\s*있을\s*경우|"
    r"공정거래위원회)"
)

NOISE_PHRASES = (
    "전성분", "성분명", "제조업자", "책임판매업자", "제조번호", "사용기한",
    "용량", "화장품법", "주의사항", "효능", "효과", "사용법", "보관방법", "소비자상담"
)

# 전성분이 아닌 일반 한국어 안내 문구(사용법·주의사항 등) 필터링용 어미/키워드
NON_INGREDIENT_PROSE_CUES = (
    "합니다", "십시오", "하세요", "됩니다", "입니다", "바랍니다", "있으므로",
    "않도록", "경우에는", "전문의", "상담", "직사광선", "어린이", "닿지",
    "상처가", "적당량", "피부결", "세안후", "가볍게", "두드려", "보관하",
    "피해서", "주의하", "붉은반점", "부어오름", "가려움증"
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
    """한글 음절을 초성/중성/종성 자모로 분해하여 시각적 오타 유사도 비교를 정밀화"""
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


def build_ingredient_dictionary() -> Tuple[Dict[str, str], Dict[int, List[Tuple[str, str, str]]]]:
    """
    skinlogic.db의 실제 화장품 전성분과 코어 사전을 통합하여
    - norm_key -> 표준 성분명 매핑 사전
    - 글자 길이별 (norm_key, jamo_key, 표준 성분명) 인덱스 (자모 사전 계산으로 고속화)
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
    by_length: Dict[int, List[Tuple[str, str, str]]] = {}

    for canonical in sorted(vocab_set, key=lambda x: (-len(x), x)):
        key_full = canonical.replace(" ", "")
        exact_map[key_full] = canonical
        by_length.setdefault(len(key_full), []).append((key_full, to_jamo(key_full), canonical))

        if "/" in key_full:
            key_noslash = key_full.replace("/", "")
            if key_noslash not in exact_map:
                exact_map[key_noslash] = canonical
                by_length.setdefault(len(key_noslash), []).append((key_noslash, to_jamo(key_noslash), canonical))

    return exact_map, by_length


ING_EXACT_MAP, ING_BY_LENGTH = build_ingredient_dictionary()


def preprocess_image_for_ocr(image_bytes: bytes) -> bytes:
    """
    1. 모바일 EXIF 회전(Orientation) 자동 보정 (세로 사진 90도 눕힘 방지)
    2. 클라우드 서버 메모리(512MB) 초과(502 에러)를 방지하기 위해
       긴 변을 최대 1100px 이내로 안전하게 조절한 뒤 흑백 대비·선명도 보정을 수행합니다.
    """
    if not HAS_PIL:
        return image_bytes
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        w, h = img.size
        max_dim = max(w, h)

        # 고해상도 캡처/사진은 최대 1100px로 축소하여 PyTorch 메모리 폭증 방지
        if max_dim > 1100:
            scale = 1100.0 / max_dim
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        # 너무 작은 크롭/캡처 조각(600px 미만)만 적정 크기(950px)로 확대
        elif max_dim < 600:
            scale = min(2.0, 950.0 / max(1, max_dim))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray, cutoff=1)
        gray = gray.filter(ImageFilter.SHARPEN)

        buf = io.BytesIO()
        gray.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return image_bytes


def find_best_dictionary_match(token: str, min_sim: float = 0.78) -> Tuple[str, float]:
    """단일 토큰을 전성분 사전과 대조하여 정확 매칭 또는 사전 계산된 한글 자모 유사도 기반 고속 오타 교정"""
    clean = token.replace(" ", "")
    if not clean:
        return ("", 0.0)

    if clean in ING_EXACT_MAP:
        return (ING_EXACT_MAP[clean], 1.0)

    t_len = len(clean)
    if t_len < 3 or t_len > 36:
        return (clean, 0.0)

    ja = to_jamo(clean)
    ja_len = len(ja)
    sm = SequenceMatcher(None, "", ja)

    best_canonical = clean
    best_score = 0.0

    for cand_len in (t_len, t_len - 1, t_len + 1):
        for _, cand_jamo, canonical in ING_BY_LENGTH.get(cand_len, []):
            # 길이 기반 빠른 상한 필터
            jb_len = len(cand_jamo)
            if (2.0 * min(ja_len, jb_len)) / max(1, ja_len + jb_len) < min_sim:
                continue
            sm.set_seq1(cand_jamo)
            if sm.quick_ratio() < min_sim:
                continue
            sim = sm.ratio()
            if sim > best_score:
                best_score = sim
                best_canonical = canonical
                if best_score >= 0.98:
                    return (best_canonical, best_score)

    if best_score >= min_sim:
        return (best_canonical, best_score)
    return (clean, best_score)


def is_likely_prose_noise(text: str) -> bool:
    """화장품 성분이 아닌 일반 한국어 안내문(사용법·주의사항 등) 여부 판별"""
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if any(cue in compact for cue in NON_INGREDIENT_PROSE_CUES):
        return True
    return False


def segment_glued_ingredients(glued_text: str) -> List[str]:
    """
    EasyOCR이 쉼표를 놓쳐 여러 성분이 한 문자열로 붙은 경우
    전성분 사전을 이용해 개별 성분 단위로 분리하고 오타를 자동 교정합니다.
    """
    s = glued_text.replace(" ", "").strip(" .:-")
    if not s:
        return []

    whole_match, whole_sim = find_best_dictionary_match(s, min_sim=0.80)
    if whole_sim >= 0.80:
        return [whole_match]

    # 일반 안내 문구(사용법·주의사항)가 통째로 들어온 경우 고속 스킵하여 CPU/타임아웃 방지
    if is_likely_prose_noise(s) and not any(kw in s for kw in ("정제수", "글리세린", "글라이콜", "추출물", "애씨드", "판테놀", "나이아신")):
        return []

    # 비정상적으로 긴 붙은 노이즈 문자열은 상한을 두어 연산 폭주 방지
    if len(s) > 160:
        s = s[:160]

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

            if sub in ING_EXACT_MAP:
                best_win_len = win_len
                best_word = ING_EXACT_MAP[sub]
                best_rank_score = 2.0 + (win_len * 0.01)
                break

            if win_len >= 4:
                matched_canon, sim = find_best_dictionary_match(sub, min_sim=0.82)
                if sim >= 0.82:
                    rank_sc = sim + (win_len * 0.005)
                    if rank_sc > best_rank_score:
                        best_rank_score = rank_sc
                        best_win_len = win_len
                        best_word = matched_canon

        if best_win_len > 0:
            if unmatched_buf:
                if len(unmatched_buf) >= 2 and re.search(r"[가-힣]{2,}", unmatched_buf) and not is_likely_prose_noise(unmatched_buf):
                    corr_buf, sim_buf = find_best_dictionary_match(unmatched_buf, min_sim=0.76)
                    if sim_buf >= 0.76 or len(unmatched_buf) <= 16:
                        results.append(corr_buf)
                unmatched_buf = ""
            results.append(best_word)
            pos += best_win_len
        else:
            unmatched_buf += s[pos]
            pos += 1

    if unmatched_buf and len(unmatched_buf) >= 2 and re.search(r"[가-힣]{2,}", unmatched_buf) and not is_likely_prose_noise(unmatched_buf):
        corr_buf, sim_buf = find_best_dictionary_match(unmatched_buf, min_sim=0.76)
        if sim_buf >= 0.76 or len(unmatched_buf) <= 16:
            results.append(corr_buf)

    return results


def sort_ocr_results_reading_order(detailed_results: List[Any]) -> List[str]:
    """
    EasyOCR detail=1 결과([(bbox, text, conf), ...])를 위->아래 줄 그룹, 왼->오른쪽 순으로 정렬합니다.
    약간 기울어진 모바일 사진에서도 전성분 읽기 순서가 뒤섞이지 않도록 보정합니다.
    """
    items = []
    for entry in (detailed_results or []):
        try:
            bbox, text, conf = entry
        except Exception:
            continue
        t = str(text or "").strip()
        if not t:
            continue
        # 신뢰도가 극도로 낮은 배경 노이즈 제거 (한글 2자 이상은 보수적으로 유지)
        if float(conf) < 0.12 and not re.search(r"[가-힣]{2,}", t):
            continue
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        x_min = min(xs)
        y_center = (min(ys) + max(ys)) / 2.0
        height = max(1.0, max(ys) - min(ys))
        items.append((y_center, x_min, height, t))

    if not items:
        return []

    items.sort(key=lambda x: x[0])

    rows: List[List[Tuple[float, float, float, str]]] = []
    for item in items:
        y_c, _, h, _ = item
        placed = False
        for row in rows:
            avg_y = sum(r[0] for r in row) / len(row)
            avg_h = sum(r[2] for r in row) / len(row)
            if abs(y_c - avg_y) <= max(avg_h * 0.55, 8.0):
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    rows.sort(key=lambda r: sum(x[0] for x in r) / len(r))
    ordered_texts: List[str] = []
    for row in rows:
        row.sort(key=lambda x: x[1])
        for _, _, _, t in row:
            ordered_texts.append(t)
    return ordered_texts


def slice_ingredients_section(text: str) -> str:
    """
    전체 화면/라벨이 함께 찍힌 경우에도 '전성분' 시작 앵커부터
    '사용 시의 주의사항 / 제조업자 / 책임판매업자' 종료 앵커 전까지 핵심 전성분 구간만 발췌합니다.
    """
    if not text:
        return ""

    section = text
    start_found = False

    start_matches = list(RE_SECTION_START.finditer(section))
    if start_matches:
        core_cues = (
            "정제수", "글리세린", "부틸렌글라이콜", "프로판다이올", "다이프로필렌글라이콜",
            "나이아신아마이드", "판테놀", "추출물", "애씨드", "글라이콜", "히알루론"
        )
        best_tail = None
        best_score = (-1, -1)
        for m in start_matches:
            tail = section[m.end():].strip()
            if len(tail) < 4:
                continue
            head = tail[:100]
            comma_cnt = len(re.findall(r"[,;，、]", head))
            core_cnt = sum(1 for kw in core_cues if kw in head)
            sc = (core_cnt * 5 + comma_cnt * 2, m.start())
            if sc > best_score:
                best_score = sc
                best_tail = tail
        if best_tail:
            section = best_tail
            start_found = True

    end_matches = list(RE_SECTION_END.finditer(section))
    if end_matches:
        ing_cues = ("정제수", "글리세린", "글라이콜", "추출물", "애씨드", "판테놀", "카보머", "베타인", "세라마이드", "히알루론")
        for em in end_matches:
            prefix = section[:em.start()].strip()
            if start_found:
                if len(prefix) >= 8:
                    section = prefix
                    break
            else:
                has_ing_signal = (
                    len(re.findall(r"[,;，、]", prefix)) >= 1
                    or any(cue in prefix for cue in ing_cues)
                )
                if len(prefix) >= 12 and has_ing_signal:
                    section = prefix
                    break

    return section.strip()


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

    merged = re.sub(r"1\s*[,.\s]\s*2\s*-\s*", "1,2-", merged)
    return merged


def clean_text_chunk(text: str) -> str:
    cleaned = RE_PREFIX.sub("", text)
    cleaned = RE_PARENTHESIS.sub("", cleaned)
    cleaned = RE_SPECIAL_CHARS.sub(" ", cleaned)
    return cleaned.strip()


def extract_and_normalize_tokens(raw_texts: List[str]) -> List[str]:
    """
    OCR 인식 원문 블록을 결합한 뒤 '전성분' 구간 자동 발췌 + 쉼표 분리 +
    붙은 성분 사전 분할 + 한글 자모 오타 교정을 수행합니다.
    """
    if not raw_texts:
        return []

    combined_text = merge_wrapped_ocr_lines(raw_texts)
    # 괄호 제거 전에 [전성분] / 주의사항 구간을 먼저 정확히 발췌
    section_text = slice_ingredients_section(combined_text)
    cleaned_text = clean_text_chunk(section_text)

    # '1,2-헥산다이올', '1,3-부틸렌글라이콜' 등 화학명 내부의 숫자 쉼표가 분리되지 않도록 임시 보호
    protected_text = re.sub(r"(\d)\s*,\s*(\d)\s*-", r"\1__NUMCOMMA__\2-", cleaned_text)
    primary_chunks = [c.replace("__NUMCOMMA__", ",") for c in re.split(r"[,;，、\n]+", protected_text)]

    deduped: List[str] = []
    seen = set()

    for chunk in primary_chunks:
        chunk_clean = re.sub(r"\s+", " ", chunk).strip(" .:-")
        if len(chunk_clean) < 2:
            continue
        # 청크 앞머리에 남은 '전성분:' 라벨이 있으면 제거하고 성분을 살림
        chunk_clean = RE_PREFIX.sub("", chunk_clean).strip(" .:-")
        if len(chunk_clean) < 2:
            continue
        if any(noise in chunk_clean for noise in NOISE_PHRASES):
            continue
        if not re.search(r"[가-힣]{2,}", chunk_clean):
            continue
        if is_likely_prose_noise(chunk_clean):
            continue

        sub_space_tokens = chunk_clean.split(" ")
        for st in sub_space_tokens:
            st_clean = st.strip(" .:-")
            if len(st_clean) < 2 or not re.search(r"[가-힣]{2,}", st_clean):
                continue

            extracted_list = segment_glued_ingredients(st_clean)
            for item in extracted_list:
                if len(item) >= 2 and item not in seen and not is_likely_prose_noise(item):
                    seen.add(item)
                    deduped.append(item)

    return deduped


def extract_ingredients_from_image(image_bytes: bytes) -> Dict[str, Any]:
    """
    이미지 바이트 스트림을 EXIF 회전 보정 및 메모리 안전 전처리한 뒤
    EasyOCR로 전성분을 추출·교정합니다.
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
        with OCR_LOCK:
            detailed_results = READER.readtext(
                enhanced_bytes,
                detail=1,
                paragraph=False,
                width_ths=0.35,
                mag_ratio=1.0
            )
        ordered_texts = sort_ocr_results_reading_order(detailed_results)
        tokens = extract_and_normalize_tokens(ordered_texts)
        return {
            "raw_text_blocks": ordered_texts,
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
    finally:
        gc.collect()


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