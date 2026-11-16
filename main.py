"""
SkinLogic - Main FastAPI Application (v3.2.1: Deployment-Ready Single Server & Async OCR)
핵심 업그레이드:
1. 단일 서버(8000번 포트) 통합 서빙:
   - FastAPI 서버에서 루트('/') 접속 시 skinlogic_web_application.html을 직접 반환하여
     HTTPS 터널(Cloudflare Tunnel) 및 클라우드 배포 시 Mixed Content / 포트 분리 문제를 해결.
2. OCR 비동기 워커 스레드 분리(run_in_threadpool) 및 업로드 임시 파일 충돌 방지:
   - EasyOCR 연산 중 메인 이벤트 루프 블로킹 방지, 동시 업로드 파일명 충돌 차단 및 처리 후 임시 파일 자동 정리.
3. 기상 API와 대기질 API 예외 처리 분리 및 SSL/타임아웃 안정화:
   - 대기질 서버 일시 지연 시에도 기온·습도·최고 UV 실측값이 정상 표시되도록 분리 처리.
4. 기존 6대 피부 유형 AI 루틴, 이너뷰티 1~4개 우선순위 추천, 다중 상극 분석, 회원 인증 및 더미 데이터 자동 제거 유지.
"""
import os
import re
import ssl
import time
import json
import urllib.request
import sqlite3
import pickle
import shutil
import hashlib
import secrets
import traceback
from datetime import datetime
from collections import Counter
from itertools import combinations
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from fastapi import FastAPI, Query, UploadFile, File, Form, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ingredient_mapper import (
    extract_standard_ingredients,
    map_for_cpp_engine,
    LIFESTYLE_SUPPLEMENT_MAP
)
from archetypes import (
    get_archetype_by_id,
    find_archetype_id,
    SKIN_CONCERNS
)

try:
    from ocr_processor import process_label_image
except ImportError as ocr_imp_err:
    print(f"⚠️ ocr_processor 로드 실패 ({ocr_imp_err}): 기본 폴백 모드로 구동됩니다.")
    def process_label_image(img_path):
        return ["정제수", "부틸렌글라이콜", "글리세린", "나이아신아마이드", "판테놀", "병풀추출물", "히알루론산"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOLUME_DIR = "/app/data"

# Railway 영구 볼륨(/app/data)이 연결되어 있으면 해당 폴더에 DB를 복사·유지하고,
# 내 맥북에서 실행할 때는 기존 로컬 skinlogic.db를 그대로 사용합니다.
if os.path.isdir(VOLUME_DIR):
    DB_PATH = os.path.join(VOLUME_DIR, "skinlogic.db")
    local_seed_db = os.path.join(BASE_DIR, "skinlogic.db")
    if not os.path.exists(DB_PATH) and os.path.exists(local_seed_db):
        shutil.copy2(local_seed_db, DB_PATH)
else:
    DB_PATH = os.path.join(BASE_DIR, "skinlogic.db")
MODEL_PATH = "routine_model.pkl"
FRONTEND_HTML_PATH = "skinlogic_web_application.html"

DUMMY_TEST_IDS = {9001, 9002, 9003, 9004}
DUMMY_TEST_NAMES = {"샘플A", "샘플B", "샘플C", "샘플D", "샘플 A", "샘플 B", "샘플 C", "샘플 D"}

app = FastAPI(
    title="SkinLogic Core API",
    description="Unified Skin-Type, Multi-Product Conflict, Dynamic Inner Beauty & AI Routine Engine",
    version="3.2.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/images/cosmetics", exist_ok=True)
os.makedirs("static/images/supplements", exist_ok=True)
os.makedirs("static/images/profiles", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==============================================================================
# [단일 서버 프론트엔드 HTML 서빙 라우트 (HTTPS & 단일 포트 배포용)]
# ==============================================================================
@app.get("/", include_in_schema=False)
def serve_frontend_root():
    if os.path.exists(FRONTEND_HTML_PATH):
        return FileResponse(FRONTEND_HTML_PATH, media_type="text/html; charset=utf-8")
    return JSONResponse(status_code=404, content={"status": "error", "message": "skinlogic_web_application.html 파일을 찾을 수 없습니다."})


@app.get("/skinlogic_web_application.html", include_in_schema=False)
def serve_frontend_html_alias():
    if os.path.exists(FRONTEND_HTML_PATH):
        return FileResponse(FRONTEND_HTML_PATH, media_type="text/html; charset=utf-8")
    return JSONResponse(status_code=404, content={"status": "error", "message": "skinlogic_web_application.html 파일을 찾을 수 없습니다."})


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def make_sample_ings(raw_text: Optional[str], max_count: int = 5) -> str:
    if not raw_text:
        return ""
    parts = [p.strip() for p in str(raw_text).split(",") if p.strip()]
    return ", ".join(parts[:max_count])

def ensure_str_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        return [s.strip() for s in val.split(",") if s.strip()] if "," in val else [val]
    return [str(val)]

def is_dummy_product_item(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    try:
        pid = int(item.get("product_id") or item.get("id") or 0)
        if pid in DUMMY_TEST_IDS:
            return True
    except (ValueError, TypeError):
        pass
    raw_name = str(item.get("name") or item.get("product_name") or "").strip()
    clean_name = re.sub(r"^\[.*?\]\s*", "", raw_name).strip()
    if raw_name in DUMMY_TEST_NAMES or clean_name in DUMMY_TEST_NAMES:
        return True
    return False


# ==============================================================================
# [0] 실시간 날씨 및 위치 기반 환경 데이터 엔진 (기상/대기질 분리 수신 & 실측/캐시/실패 구분)
# ==============================================================================
KR_REGION_COORDS: Dict[str, Tuple[float, float, str]] = {
    "수원": (37.2636, 127.0286, "경기도 수원시"),
    "서울": (37.5665, 126.9780, "서울특별시"),
    "인천": (37.4563, 126.7052, "인천광역시"),
    "대전": (36.3504, 127.3845, "대전광역시"),
    "대구": (35.8714, 128.6014, "대구광역시"),
    "광주": (35.1595, 126.8526, "광주광역시"),
    "부산": (35.1796, 129.0756, "부산광역시"),
    "제주": (33.4996, 126.5312, "제주특별자치도")
}

WMO_WEATHER_MAP: Dict[int, str] = {
    0: "맑음 ☀️", 1: "대체로 맑음 🌤️", 2: "구름 조금 ⛅", 3: "흐림 ☁️",
    45: "안개 🌫️", 48: "짙은 안개 🌫️",
    51: "가벼운 이슬비 🌦️", 53: "이슬비 🌦️", 55: "강한 이슬비 🌧️",
    61: "약한 비 🌧️", 63: "비 🌧️", 65: "강한 비 ⛈️",
    71: "약한 눈 🌨️", 73: "눈 ❄️", 75: "강한 눈 ❄️",
    80: "소나기 🌦️", 81: "강한 소나기 ⛈️", 95: "뇌우 ⛈️"
}

LAST_WEATHER_CACHE: Dict[str, Dict[str, Any]] = {}


def classify_uv_level(uv_val: Optional[float]) -> Dict[str, str]:
    if uv_val is None:
        return {
            "level": "측정 불가",
            "code": "UNAVAILABLE",
            "guide": "자외선 측정값을 수신하지 못했습니다."
        }
    v = float(uv_val)
    if v < 3.0:
        return {
            "level": "낮음",
            "code": "LOW",
            "guide": "자외선 지수가 낮음(0~2.9) 수준입니다. 야간 또는 흐린 날씨에는 자외선 자극 우려가 적습니다."
        }
    if v < 6.0:
        return {
            "level": "보통",
            "code": "MODERATE",
            "guide": "자외선 지수가 보통(3.0~5.9) 수준입니다. 2~3시간 이상 야외 활동 시 자외선 차단제를 도포하세요."
        }
    if v < 8.0:
        return {
            "level": "높음",
            "code": "HIGH",
            "guide": "자외선 지수가 높음(6.0~7.9) 수준입니다. SPF50+ 선케어를 필수로 바르고 레티놀·AHA는 저녁에만 사용하세요."
        }
    if v < 11.0:
        return {
            "level": "매우 높음",
            "code": "VERY_HIGH",
            "guide": "자외선 지수가 매우 높음(8.0~10.9) 수준입니다. 한낮 직사광선 노출을 피하고 선크림을 2시간마다 덧바르세요."
        }
    return {
        "level": "위험",
        "code": "EXTREME",
        "guide": "자외선 지수가 위험(11.0+) 수준입니다. 야외 노출을 최소화하고 철저한 자외선 방어가 필요합니다."
    }


def build_env_skincare_tips_and_rationale(
    temp: Optional[float],
    humidity: Optional[int],
    uv_current: Optional[float],
    uv_max_today: Optional[float],
    uv_max_tomorrow: Optional[float],
    pm25: Optional[float],
    pm10: Optional[float],
    is_day: Optional[int] = None
) -> Tuple[List[str], List[Dict[str, str]]]:
    tips: List[str] = []
    rationale_items: List[Dict[str, str]] = []

    cur_uv_meta = classify_uv_level(uv_current)
    max_uv_meta = classify_uv_level(uv_max_today)

    if uv_current is not None:
        if uv_current >= 6.0:
            msg = f"현재 실시간 자외선(UV {uv_current} · {cur_uv_meta['level']})이 높습니다. 외출 시 SPF50+ PA++++ 선크림을 꼼꼼히 도포하고, 레티놀·고농도 AHA 제품은 낮 사용을 피하세요."
            tips.append(msg)
            rationale_items.append({
                "factor": f"실시간 UV {uv_current} ({cur_uv_meta['level']})",
                "action": "고차단 선케어 필수 · 광감작 성분 저녁 분리",
                "reason": msg
            })
        elif uv_current >= 3.0:
            msg = f"현재 실시간 자외선(UV {uv_current} · {cur_uv_meta['level']})입니다. 외출 시 데일리 선크림·선세럼으로 광노화 및 잡티를 예방하세요."
            tips.append(msg)
            rationale_items.append({
                "factor": f"실시간 UV {uv_current} ({cur_uv_meta['level']})",
                "action": "데일리 선크림 도포 권장",
                "reason": msg
            })
        else:
            night_hint = "일몰 후 야간 시간대이므로 " if is_day == 0 else "현재 자외선 세기가 약하므로 "
            if uv_max_today is not None and uv_max_today >= 3.0:
                msg = (
                    f"현재 실시간 자외선은 UV {uv_current}({cur_uv_meta['level']})입니다. "
                    f"{night_hint}저녁 나이트 케어(레티놀·각질 정돈)를 활용하기 좋으며, "
                    f"낮 최고 자외선(UV {uv_max_today} · {max_uv_meta['level']}) 시간대 외출 시에는 선케어를 챙겨주세요."
                )
            else:
                msg = f"현재 실시간 자외선은 UV {uv_current}({cur_uv_meta['level']})입니다. {night_hint}자외선 자극 부담이 적어 보습·진정 및 집중 케어에 적합한 시간대입니다."
            tips.append(msg)
            rationale_items.append({
                "factor": f"실시간 UV {uv_current} ({cur_uv_meta['level']})" + (f" / 최고 UV {uv_max_today}" if uv_max_today is not None else ""),
                "action": "야간·저자외선 집중 케어 적합 / 낮 외출 시 선케어 유지",
                "reason": msg
            })
    elif uv_max_today is not None:
        msg = f"실시간 UV는 미수신되었으나 오늘 최고 예보 자외선은 UV {uv_max_today}({max_uv_meta['level']})입니다. 낮 시간대 외출 시 자외선 차단제를 권장합니다."
        tips.append(msg)
        rationale_items.append({
            "factor": f"오늘 최고 UV {uv_max_today} ({max_uv_meta['level']})",
            "action": "낮 시간대 선케어 권장",
            "reason": msg
        })

    if humidity is not None:
        if humidity <= 45:
            msg = f"현재 습도({humidity}%)가 낮아 피부 수분 증발이 빨라집니다. 히알루론산·판테놀 세럼 후 세라마이드 보습 크림으로 수분 보호막을 형성하세요."
            tips.append(msg)
            rationale_items.append({
                "factor": f"습도 {humidity}% (건조)",
                "action": "히알루론산 · 세라마이드 보습 레이어링",
                "reason": msg
            })
        elif humidity >= 75:
            msg = f"현재 습도({humidity}%)가 높아 피지 및 번들거림이 증가하기 쉽습니다. 무거운 오일·밤 제형은 줄이고 수분 젤·수딩 세럼 위주로 가볍게 케어하세요."
            tips.append(msg)
            rationale_items.append({
                "factor": f"습도 {humidity}% (다습)",
                "action": "산뜻한 수딩 젤 · 저유분 밸런싱",
                "reason": msg
            })
        else:
            msg = f"현재 습도({humidity}%)는 쾌적한 적정 범위(46~74%)입니다. 피부 유형에 맞춘 기본 수분·유분 밸런스 루틴을 유지하세요."
            rationale_items.append({
                "factor": f"습도 {humidity}% (적정)",
                "action": "기본 유수분 밸런스 루틴 유지",
                "reason": msg
            })

    if pm25 is not None:
        if pm25 >= 36.0 or (pm10 is not None and pm10 >= 81.0):
            msg = f"초미세먼지(PM2.5 {pm25}㎍/㎥ · 나쁨) 수치가 높습니다. 귀가 후 약산성 딥클렌징으로 모공 노폐물을 닦아내고 시카·어성초 진정 케어를 권장합니다."
            tips.append(msg)
            rationale_items.append({
                "factor": f"초미세먼지 {pm25}㎍/㎥ (나쁨)",
                "action": "귀가 후 꼼꼼한 약산성 세안 · 시카 진정",
                "reason": msg
            })
        elif pm25 <= 15.0:
            rationale_items.append({
                "factor": f"초미세먼지 {pm25}㎍/㎥ (좋음)",
                "action": "과도한 이중 세안 자제 · 저자극 세안 유지",
                "reason": f"초미세먼지({pm25}㎍/㎥)가 '좋음' 수준이므로 피부 장벽 보호를 위해 과도한 세정 대신 순한 약산성 세안을 권장합니다."
            })

    if temp is not None:
        if temp <= 10.0:
            msg = f"현재 기온({temp}°C)이 낮아 실내외 온도차로 피부 장벽이 건조하고 민감해지기 쉽습니다. 판테놀·세라마이드 장벽 보습 비중을 높여주세요."
            tips.append(msg)
            rationale_items.append({
                "factor": f"기온 {temp}°C (저온)",
                "action": "장벽 보호 · 고보습 크림 강화",
                "reason": msg
            })
        elif temp >= 28.0:
            msg = f"현재 기온({temp}°C)이 높아 피부 열감과 모공 확장이 발생하기 쉽습니다. 쿨링 수딩 케어와 피지 조절 성분(나이아신아마이드)을 권장합니다."
            tips.append(msg)
            rationale_items.append({
                "factor": f"기온 {temp}°C (고온)",
                "action": "피부 열감 쿨링 · 피지 케어",
                "reason": msg
            })

    return tips, rationale_items


@app.get("/api/v1/environment/weather", summary="실시간 위치/지역 기반 기온·습도·실시간UV·최고UV·미세먼지 조회")
def get_environment_weather(
    region: Optional[str] = Query("수원", description="지역명"),
    lat: Optional[float] = Query(None, description="위도"),
    lon: Optional[float] = Query(None, description="경도")
):
    region_key = (region or "수원").strip()
    if lat is not None and lon is not None:
        target_lat, target_lon = float(lat), float(lon)
        nearest_city = "현재 위치 (GPS)"
        min_dist = 999.0
        for r_k, (r_lat, r_lon, r_full) in KR_REGION_COORDS.items():
            dist = ((target_lat - r_lat) ** 2 + (target_lon - r_lon) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest_city = f"{r_full} (GPS)" if dist < 0.5 else "사용자 현재 위치 (GPS)"
        display_region = nearest_city
        cache_key = f"GPS_{round(target_lat, 2)}_{round(target_lon, 2)}"
    else:
        if region_key not in KR_REGION_COORDS:
            region_key = "수원"
        target_lat, target_lon, display_region = KR_REGION_COORDS[region_key]
        cache_key = region_key

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ssl_ctx = ssl._create_unverified_context()

    temp: Optional[float] = None
    humidity: Optional[int] = None
    weather_desc: str = "기상 확인"
    is_day: Optional[int] = None
    uv_max_today: Optional[float] = None
    uv_max_tomorrow: Optional[float] = None
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    uv_current: Optional[float] = None

    forecast_ok = False
    aq_ok = False
    last_err_msg = ""

    # 1. 기상 및 일별 최고 UV 예보 조회 (Open-Meteo Forecast API)
    try:
        w_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={target_lat}&longitude={target_lon}"
            f"&current=temperature_2m,relative_humidity_2m,weather_code,is_day"
            f"&daily=uv_index_max&forecast_days=2&timezone=Asia%2FSeoul"
        )
        req_w = urllib.request.Request(w_url, headers={"User-Agent": "SkinLogic/3.2.1"})
        with urllib.request.urlopen(req_w, context=ssl_ctx, timeout=5.0) as resp_w:
            w_data = json.loads(resp_w.read().decode("utf-8"))
            cur_w = w_data.get("current", {})
            daily_w = w_data.get("daily", {})

            raw_temp = cur_w.get("temperature_2m")
            raw_hum = cur_w.get("relative_humidity_2m")
            raw_wcode = cur_w.get("weather_code")
            raw_is_day = cur_w.get("is_day")

            temp = round(float(raw_temp), 1) if raw_temp is not None else None
            humidity = int(raw_hum) if raw_hum is not None else None
            weather_desc = WMO_WEATHER_MAP.get(int(raw_wcode), "맑음 ☀️") if raw_wcode is not None else "기상 정보 확인"
            is_day = int(raw_is_day) if raw_is_day is not None else None

            uv_max_list = daily_w.get("uv_index_max") or []
            uv_max_today = round(float(uv_max_list[0]), 1) if len(uv_max_list) >= 1 and uv_max_list[0] is not None else None
            uv_max_tomorrow = round(float(uv_max_list[1]), 1) if len(uv_max_list) >= 2 and uv_max_list[1] is not None else None
            forecast_ok = True
    except Exception as f_err:
        last_err_msg = f"기상 예보 API 오류({type(f_err).__name__})"

    # 2. 실시간 대기질 및 현재 UV 지수 조회 (독립 예외 처리로 한쪽 지연 시에도 기상 실측값 보호)
    try:
        aq_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={target_lat}&longitude={target_lon}"
            f"&current=pm10,pm2_5,uv_index&timezone=Asia%2FSeoul"
        )
        req_aq = urllib.request.Request(aq_url, headers={"User-Agent": "SkinLogic/3.2.1"})
        with urllib.request.urlopen(req_aq, context=ssl_ctx, timeout=5.0) as resp_aq:
            aq_data = json.loads(resp_aq.read().decode("utf-8"))
            cur_aq = aq_data.get("current", {})

            raw_pm25 = cur_aq.get("pm2_5")
            raw_pm10 = cur_aq.get("pm10")
            raw_uv = cur_aq.get("uv_index")

            pm25 = round(float(raw_pm25), 1) if raw_pm25 is not None else None
            pm10 = round(float(raw_pm10), 1) if raw_pm10 is not None else None
            uv_current = round(float(raw_uv), 1) if raw_uv is not None else None
            aq_ok = True
    except Exception as aq_err:
        if not last_err_msg:
            last_err_msg = f"대기질·UV API 오류({type(aq_err).__name__})"

    # 3. 기상 또는 대기질 중 하나라도 정상 수신된 경우: 실제 수신 데이터 반환
    if forecast_ok or aq_ok:
        cur_uv_meta = classify_uv_level(uv_current)
        max_uv_meta = classify_uv_level(uv_max_today)
        tmr_uv_meta = classify_uv_level(uv_max_tomorrow)

        uv_context_note = ""
        if uv_current is not None and uv_current == 0.0:
            uv_context_note = "일몰 후 야간 시간대로 실제 실시간 UV는 0.0(낮음)입니다. 낮 시간대 최고 UV는 별도로 표시됩니다."
        elif uv_current is None:
            uv_context_note = "실시간 UV 센서 응답이 지연되어 임의 기본값 대신 '측정 불가'로 표시하고, 오늘 최고 예보 UV를 반영합니다."

        tips, rationale_items = build_env_skincare_tips_and_rationale(
            temp=temp,
            humidity=humidity,
            uv_current=uv_current,
            uv_max_today=uv_max_today,
            uv_max_tomorrow=uv_max_tomorrow,
            pm25=pm25,
            pm10=pm10,
            is_day=is_day
        )

        result = {
            "status": "success",
            "is_live": True,
            "data_source": "LIVE",
            "source_label": "실시간 관측 데이터",
            "region_key": region_key,
            "region_name": display_region,
            "lat": target_lat,
            "lon": target_lon,
            "is_day": is_day,
            "temp": temp,
            "humidity": humidity,
            "uv_index": uv_current,
            "uv_level": cur_uv_meta["level"],
            "uv_level_code": cur_uv_meta["code"],
            "uv_guide": cur_uv_meta["guide"],
            "uv_max_today": uv_max_today,
            "uv_max_level": max_uv_meta["level"],
            "uv_max_level_code": max_uv_meta["code"],
            "uv_max_tomorrow": uv_max_tomorrow,
            "uv_tomorrow_level": tmr_uv_meta["level"],
            "uv_tomorrow_level_code": tmr_uv_meta["code"],
            "uv_context_note": uv_context_note,
            "pm25": pm25,
            "pm10": pm10,
            "weather_desc": weather_desc if forecast_ok else "기상 예보 지연",
            "updated_at": now_str,
            "skincare_tips": tips,
            "recommendation_rationale": rationale_items
        }
        LAST_WEATHER_CACHE[cache_key] = result
        return result

    # 4. 둘 다 실패했지만 이전 정상 캐시가 있는 경우
    if cache_key in LAST_WEATHER_CACHE:
        cached = dict(LAST_WEATHER_CACHE[cache_key])
        cached["status"] = "cached"
        cached["is_live"] = False
        cached["data_source"] = "CACHED"
        cached["source_label"] = "최근 수신 캐시 (실시간 연결 지연)"
        cached["fallback_notice"] = f"기상 API 일시 지연({last_err_msg})으로 마지막 정상 관측 데이터({cached['updated_at']})를 표시합니다."
        return cached

    # 5. 둘 다 실패 + 캐시도 없는 경우: 임의의 기본값을 실측값처럼 표시하지 않음
    return {
        "status": "unavailable",
        "is_live": False,
        "data_source": "UNAVAILABLE",
        "source_label": "데이터 수신 실패 (실측값 없음)",
        "region_key": region_key,
        "region_name": display_region,
        "lat": target_lat,
        "lon": target_lon,
        "is_day": None,
        "temp": None,
        "humidity": None,
        "uv_index": None,
        "uv_level": "측정 불가",
        "uv_level_code": "UNAVAILABLE",
        "uv_guide": "기상 데이터 수신에 실패하여 UV 지수를 판단할 수 없습니다.",
        "uv_max_today": None,
        "uv_max_level": "측정 불가",
        "uv_max_level_code": "UNAVAILABLE",
        "uv_max_tomorrow": None,
        "uv_tomorrow_level": "측정 불가",
        "uv_tomorrow_level_code": "UNAVAILABLE",
        "uv_context_note": "실시간 기상·대기질 서버 연결에 실패하여 임의의 기본값을 표시하지 않았습니다.",
        "pm25": None,
        "pm10": None,
        "weather_desc": "수신 실패 ⚠️",
        "updated_at": now_str,
        "error_reason": f"{last_err_msg}: 네트워크 연결 또는 외부 기상 서버 상태를 확인해 주세요.",
        "fallback_notice": "기상 데이터를 수신하지 못했습니다. 임의의 기본값을 실측값처럼 표시하지 않습니다.",
        "skincare_tips": [
            "실시간 날씨·UV 데이터를 수신하지 못해 환경 보정 대신 선택한 피부 유형과 생활패턴 기준으로 루틴을 구성합니다.",
            "낮 시간대 외출 시에는 기본적으로 자외선 차단제 사용을 권장합니다."
        ],
        "recommendation_rationale": []
    }

@app.get("/api/v1/environment/uv")
def get_env_uv(lat: float = 37.2636, lon: float = 127.0286):
    return get_environment_weather(region="수원", lat=lat, lon=lon)


# ==============================================================================
# [1] 회원(users) 및 회원별 개인화 데이터(user_states) 테이블 & 보안 인증
# ==============================================================================
def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return pw_hash, salt

def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    calc_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(calc_hash, stored_hash)

DEFAULT_USER_STATE = {
    "selected_region": "수원",
    "scan_product_list": [],
    "selected_supplements": [],
    "vanity_basket": [],
    "routine_prefs": {
        "skin_type": "수부지",
        "concern": "수분/속건조",
        "care_style_id": 3,
        "supplement_limit": 3,
        "makeup": 1,
        "sleep": 1,
        "outdoor": 1,
        "alcohol": 0,
        "workout": 1
    },
    "custom_routine": {
        "morning_steps": [],
        "night_steps": []
    },
    "last_ai_result": None,
    "archive_list": []
}

def cleanup_test_dummy_data():
    if not os.path.exists(DB_PATH):
        return
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            DELETE FROM products
            WHERE (product_id IN (9001, 9002, 9003, 9004)
               OR product_name IN ('샘플A', '샘플B', '샘플C', '샘플D', '샘플 A', '샘플 B', '샘플 C', '샘플 D'))
              AND COALESCE(report_no, '') NOT LIKE 'USER_OCR_%'
        """)
        deleted_cnt = cur.rowcount
        conn.commit()
        conn.close()
        if deleted_cnt > 0:
            print(f"🧹 테스트용 더미 데이터 {deleted_cnt}건 자동 정리 완료 (실제 사용자 데이터 보호됨)")
    except Exception as e:
        print(f"[WARN] Dummy data cleanup skipped: {e}")

def init_user_tables():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            nickname TEXT NOT NULL,
            profile_image TEXT DEFAULT '',
            session_token TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_states (
            user_id INTEGER PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

ai_package: Dict[str, Any] = {}

@app.on_event("startup")
def on_startup():
    global ai_package
    init_user_tables()
    cleanup_test_dummy_data()
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            ai_package = pickle.load(f)
        print(f"🤖 AI Routine Model 로드 완료 ({ai_package.get('model_name')}, 정확도: {ai_package.get('accuracy')*100:.2f}%)")
    else:
        print("⚠️ 'routine_model.pkl'을 찾을 수 없습니다. 기본 룰셋 모드로 구동됩니다.")

def get_user_by_token(conn: sqlite3.Connection, token: Optional[str]) -> Optional[sqlite3.Row]:
    if not token:
        return None
    clean_tok = token.replace("Bearer ", "").strip()
    if not clean_tok:
        return None
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE session_token = ?", (clean_tok,))
    return cur.fetchone()

def get_or_create_user_state(conn: sqlite3.Connection, user_id: int) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT state_json FROM user_states WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    if row and row["state_json"]:
        try:
            loaded = json.loads(row["state_json"])
            merged = json.loads(json.dumps(DEFAULT_USER_STATE))
            merged.update(loaded)
            if "routine_prefs" in loaded and isinstance(loaded["routine_prefs"], dict):
                merged["routine_prefs"] = {**DEFAULT_USER_STATE["routine_prefs"], **loaded["routine_prefs"]}
            if "custom_routine" in loaded and isinstance(loaded["custom_routine"], dict):
                merged["custom_routine"] = {**DEFAULT_USER_STATE["custom_routine"], **loaded["custom_routine"]}
            merged["vanity_basket"] = [x for x in (merged.get("vanity_basket") or []) if not is_dummy_product_item(x)]
            merged["scan_product_list"] = [x for x in (merged.get("scan_product_list") or []) if not is_dummy_product_item(x)]
            return merged
        except Exception:
            pass
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fresh_state = json.loads(json.dumps(DEFAULT_USER_STATE))
    cur.execute(
        "INSERT OR REPLACE INTO user_states (user_id, state_json, updated_at) VALUES (?, ?, ?)",
        (user_id, json.dumps(fresh_state, ensure_ascii=False), now_str)
    )
    conn.commit()
    return fresh_state


@app.get("/api/v1/auth/check-username", summary="회원가입 시 아이디 중복 확인")
def check_username_duplicate(username: str = Query(..., min_length=2)):
    uname = username.strip()
    if len(uname) < 3:
        return {"available": False, "message": "아이디는 최소 3자 이상 입력해 주세요."}
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE username = ?", (uname,))
    exists = cur.fetchone() is not None
    conn.close()
    if exists:
        return {"available": False, "message": f"'{uname}'은(는) 이미 사용 중인 아이디입니다."}
    return {"available": True, "message": f"'{uname}'은(는) 사용 가능한 아이디입니다."}


@app.post("/api/v1/auth/signup", summary="회원가입 (프로필 사진 업로드 및 회원별 독립 상태 생성)")
async def signup_user(
    username: str = Form(...),
    password: str = Form(...),
    nickname: str = Form(...),
    profile_file: Optional[UploadFile] = File(None)
):
    uname = username.strip()
    nick = nickname.strip()
    if len(uname) < 3:
        return JSONResponse(status_code=400, content={"status": "error", "message": "아이디는 3자 이상이어야 합니다."})
    if len(password) < 4:
        return JSONResponse(status_code=400, content={"status": "error", "message": "비밀번호는 4자 이상이어야 합니다."})
    if not nick:
        return JSONResponse(status_code=400, content={"status": "error", "message": "닉네임을 입력해 주세요."})

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE username = ?", (uname,))
    if cur.fetchone():
        conn.close()
        return JSONResponse(status_code=400, content={"status": "error", "message": "이미 등록된 아이디입니다. 중복 확인을 진행해 주세요."})

    profile_url = ""
    if profile_file and profile_file.filename:
        ext = os.path.splitext(profile_file.filename)[1] or ".png"
        fname = f"user_{uname}_{int(time.time())}{ext}"
        save_path = os.path.join("static/images/profiles", fname)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(profile_file.file, buffer)
        profile_url = f"/static/images/profiles/{fname}"

    pw_hash, pw_salt = hash_password(password)
    token = secrets.token_hex(24)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute("""
        INSERT INTO users (username, password_hash, password_salt, nickname, profile_image, session_token, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (uname, pw_hash, pw_salt, nick, profile_url, token, now_str))
    new_uid = int(cur.lastrowid)

    fresh_state = json.loads(json.dumps(DEFAULT_USER_STATE))
    cur.execute("""
        INSERT INTO user_states (user_id, state_json, updated_at)
        VALUES (?, ?, ?)
    """, (new_uid, json.dumps(fresh_state, ensure_ascii=False), now_str))
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "message": f"환영합니다, {nick}님! 회원가입이 완료되었습니다.",
        "token": token,
        "user": {
            "user_id": new_uid,
            "username": uname,
            "nickname": nick,
            "profile_image": profile_url,
            "created_at": now_str
        },
        "state": fresh_state
    }


class LoginRequest(BaseModel):
    username: str
    password: str
    remember_me: bool = True

@app.post("/api/v1/auth/login", summary="아이디 로그인 및 회원별 저장 상태 복원")
def login_user(req: LoginRequest):
    uname = (req.username or "").strip()
    pw = req.password or ""
    if not uname or not pw:
        return JSONResponse(status_code=400, content={"status": "error", "message": "아이디와 비밀번호를 모두 입력해 주세요."})

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (uname,))
    user = cur.fetchone()
    if not user:
        conn.close()
        return JSONResponse(status_code=401, content={"status": "error", "error_code": "USER_NOT_FOUND", "message": "존재하지 않는 아이디입니다. 아이디를 다시 확인해 주세요."})

    if not verify_password(pw, user["password_hash"], user["password_salt"]):
        conn.close()
        return JSONResponse(status_code=401, content={"status": "error", "error_code": "WRONG_PASSWORD", "message": "비밀번호가 일치하지 않습니다. 다시 입력해 주세요."})

    token = secrets.token_hex(24)
    cur.execute("UPDATE users SET session_token = ? WHERE user_id = ?", (token, user["user_id"]))
    conn.commit()

    user_state = get_or_create_user_state(conn, user["user_id"])
    conn.close()

    return {
        "status": "success",
        "message": f"{user['nickname']}님, 환영합니다! 저장된 개인 데이터가 복원되었습니다.",
        "token": token,
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "nickname": user["nickname"],
            "profile_image": user["profile_image"] or "",
            "created_at": user["created_at"]
        },
        "state": user_state
    }


@app.get("/api/v1/auth/me", summary="자동 로그인 검증 및 내 프로필·저장 상태 불러오기")
def get_current_user_session(authorization: Optional[str] = Header(None)):
    conn = get_db()
    user = get_user_by_token(conn, authorization)
    if not user:
        conn.close()
        return JSONResponse(status_code=401, content={"status": "error", "message": "유효하지 않거나 만료된 로그인 세션입니다."})

    user_state = get_or_create_user_state(conn, user["user_id"])
    conn.close()
    return {
        "status": "success",
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "nickname": user["nickname"],
            "profile_image": user["profile_image"] or "",
            "created_at": user["created_at"]
        },
        "state": user_state
    }


@app.post("/api/v1/auth/logout", summary="로그아웃 (DB의 회원 개인 데이터는 영구 보존하고 세션 토큰만 해제)")
def logout_user(authorization: Optional[str] = Header(None)):
    conn = get_db()
    user = get_user_by_token(conn, authorization)
    if user:
        cur = conn.cursor()
        cur.execute("UPDATE users SET session_token = '' WHERE user_id = ?", (user["user_id"],))
        conn.commit()
    conn.close()
    return {"status": "success", "message": "안전하게 로그아웃되었습니다. 회원님의 데이터는 계정에 안전하게 보관되어 있습니다."}


class FindAccountRequest(BaseModel):
    mode: str = Field("FIND_ID", description="FIND_ID 또는 RESET_PW")
    nickname: Optional[str] = ""
    username: Optional[str] = ""
    new_password: Optional[str] = ""

@app.post("/api/v1/auth/find-account", summary="아이디 찾기 및 비밀번호 재설정")
def find_or_reset_account(req: FindAccountRequest):
    conn = get_db()
    cur = conn.cursor()

    if req.mode == "FIND_ID":
        nick = (req.nickname or "").strip()
        if not nick:
            conn.close()
            return JSONResponse(status_code=400, content={"status": "error", "message": "가입 시 설정한 닉네임을 입력해 주세요."})
        cur.execute("SELECT username, created_at FROM users WHERE nickname = ?", (nick,))
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return JSONResponse(status_code=404, content={"status": "error", "message": f"닉네임 '{nick}'(으)로 등록된 계정을 찾을 수 없습니다."})
        found_ids = [r["username"] for r in rows]
        return {
            "status": "success",
            "message": f"회원님의 아이디를 찾았습니다: {', '.join(found_ids)}",
            "found_usernames": found_ids
        }

    uname = (req.username or "").strip()
    nick = (req.nickname or "").strip()
    new_pw = req.new_password or ""
    if not uname or not nick or len(new_pw) < 4:
        conn.close()
        return JSONResponse(status_code=400, content={"status": "error", "message": "아이디, 닉네임 및 새 비밀번호(4자 이상)를 정확히 입력해 주세요."})

    cur.execute("SELECT user_id FROM users WHERE username = ? AND nickname = ?", (uname, nick))
    user = cur.fetchone()
    if not user:
        conn.close()
        return JSONResponse(status_code=404, content={"status": "error", "message": "아이디와 닉네임 정보가 일치하는 계정이 없습니다."})

    pw_hash, pw_salt = hash_password(new_pw)
    cur.execute("UPDATE users SET password_hash = ?, password_salt = ? WHERE user_id = ?", (pw_hash, pw_salt, user["user_id"]))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "비밀번호가 성공적으로 재설정되었습니다. 새 비밀번호로 로그인해 주세요."}


@app.put("/api/v1/auth/profile", summary="내 프로필 수정 (닉네임·비밀번호·프로필 사진 변경)")
async def update_user_profile(
    authorization: Optional[str] = Header(None),
    nickname: Optional[str] = Form(None),
    current_password: Optional[str] = Form(None),
    new_password: Optional[str] = Form(None),
    profile_file: Optional[UploadFile] = File(None)
):
    conn = get_db()
    user = get_user_by_token(conn, authorization)
    if not user:
        conn.close()
        return JSONResponse(status_code=401, content={"status": "error", "message": "로그인이 필요합니다."})

    cur = conn.cursor()
    uid = user["user_id"]
    updated_nick = user["nickname"]
    updated_img = user["profile_image"] or ""

    if nickname and nickname.strip():
        updated_nick = nickname.strip()

    if profile_file and profile_file.filename:
        ext = os.path.splitext(profile_file.filename)[1] or ".png"
        fname = f"user_{user['username']}_{int(time.time())}{ext}"
        save_path = os.path.join("static/images/profiles", fname)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(profile_file.file, buffer)
        updated_img = f"/static/images/profiles/{fname}"

    if new_password and len(new_password.strip()) > 0:
        if not current_password or not verify_password(current_password, user["password_hash"], user["password_salt"]):
            conn.close()
            return JSONResponse(status_code=400, content={"status": "error", "message": "현재 비밀번호가 일치하지 않아 비밀번호를 변경할 수 없습니다."})
        if len(new_password.strip()) < 4:
            conn.close()
            return JSONResponse(status_code=400, content={"status": "error", "message": "새 비밀번호는 4자 이상이어야 합니다."})
        pw_hash, pw_salt = hash_password(new_password.strip())
        cur.execute("UPDATE users SET password_hash = ?, password_salt = ? WHERE user_id = ?", (pw_hash, pw_salt, uid))

    cur.execute("UPDATE users SET nickname = ?, profile_image = ? WHERE user_id = ?", (updated_nick, updated_img, uid))
    conn.commit()
    conn.close()

    return {
        "status": "success",
        "message": "회원 프로필 정보가 성공적으로 수정되었습니다.",
        "user": {
            "user_id": uid,
            "username": user["username"],
            "nickname": updated_nick,
            "profile_image": updated_img,
            "created_at": user["created_at"]
        }
    }


class DeleteAccountRequest(BaseModel):
    password: str

@app.delete("/api/v1/auth/account", summary="회원 탈퇴")
def delete_user_account(req: DeleteAccountRequest, authorization: Optional[str] = Header(None)):
    conn = get_db()
    user = get_user_by_token(conn, authorization)
    if not user:
        conn.close()
        return JSONResponse(status_code=401, content={"status": "error", "message": "로그인이 필요합니다."})

    if not verify_password(req.password or "", user["password_hash"], user["password_salt"]):
        conn.close()
        return JSONResponse(status_code=400, content={"status": "error", "message": "비밀번호가 일치하지 않아 탈퇴 처리를 진행할 수 없습니다."})

    cur = conn.cursor()
    cur.execute("DELETE FROM user_states WHERE user_id = ?", (user["user_id"],))
    cur.execute("DELETE FROM users WHERE user_id = ?", (user["user_id"],))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "회원 탈퇴가 완료되었습니다."}


class UserStateSaveRequest(BaseModel):
    selected_region: Optional[str] = "수원"
    scan_product_list: List[Dict[str, Any]] = Field(default_factory=list)
    selected_supplements: List[Dict[str, Any]] = Field(default_factory=list)
    vanity_basket: List[Dict[str, Any]] = Field(default_factory=list)
    routine_prefs: Dict[str, Any] = Field(default_factory=dict)
    custom_routine: Dict[str, Any] = Field(default_factory=dict)
    last_ai_result: Optional[Dict[str, Any]] = None
    archive_list: List[Dict[str, Any]] = Field(default_factory=list)

@app.put("/api/v1/user/state", summary="회원별 작업 상태 실시간 DB 영구 저장 (user_id별 격리)")
def save_user_state(req: UserStateSaveRequest, authorization: Optional[str] = Header(None)):
    conn = get_db()
    user = get_user_by_token(conn, authorization)
    if not user:
        conn.close()
        return JSONResponse(status_code=401, content={"status": "error", "message": "로그인이 필요합니다."})

    clean_scan = [x for x in (req.scan_product_list or []) if not is_dummy_product_item(x)]
    clean_vanity = [x for x in (req.vanity_basket or []) if not is_dummy_product_item(x)]

    state_dict = {
        "selected_region": req.selected_region or "수원",
        "scan_product_list": clean_scan,
        "selected_supplements": req.selected_supplements,
        "vanity_basket": clean_vanity,
        "routine_prefs": req.routine_prefs,
        "custom_routine": req.custom_routine,
        "last_ai_result": req.last_ai_result,
        "archive_list": req.archive_list
    }
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO user_states (user_id, state_json, updated_at)
        VALUES (?, ?, ?)
    """, (user["user_id"], json.dumps(state_dict, ensure_ascii=False), now_str))
    conn.commit()
    conn.close()
    return {"status": "success", "user_id": user["user_id"], "updated_at": now_str}


# ==============================================================================
# [2] 6대 배타적 제형 역할군(Role Bucket) 분류기 및 DB 카테고리 추론기
# ==============================================================================
def classify_cosmetic_role(item_type: str = "", name: str = "", category: str = "") -> Tuple[str, str]:
    it = (item_type or "").strip()
    nm = (name or "").strip()
    cat = (category or "").strip()
    combined = f"{it} {nm}"

    if cat == "Cleansing" or any(k in combined for k in [
        "클렌징", "클렌저", "세안", "리무버", "페이셜 워시", "폼클렌징", "약산성 폼", "딥 클렌징", "버블 폼", "파우더 워시"
    ]):
        if any(k in combined for k in ["오일", "클렌징 밤", "밀크밤", "클렌징 밀크", "클렌징 워터", "리무버", "트랜스포밍"]) and "폼" not in it:
            return ("CLEANSING", "CLEANSING_OIL")
        return ("CLEANSING", "CLEANSING_FOAM")

    if cat == "SunCare" or any(k in combined for k in [
        "선크림", "선세럼", "선젤", "선플루이드", "선스틱", "선쿠션", "선팩트",
        "선스프레이", "선패치", "선케어", "선로션", "선밀크", "선밤", "톤업선",
        "자외선", "무기자차", "유기자차", "SPF", "spf"
    ]):
        return ("SUNCARE", "SUNCARE")

    if cat == "MaskPack" or any(k in combined for k in [
        "마스크팩", "시트마스크", "워시오프팩", "슬리핑팩", "필오프팩", "모델링팩", "코팩", "부분패치", "마스크", "나이트팩"
    ]):
        return ("MASKPACK", "MASKPACK")

    if any(k in combined for k in ["스킨/토너", "토너패드", "토너", "스킨", "미스트", "패드"]):
        if "패드" in combined:
            return ("TONER", "TONER_PAD")
        return ("TONER", "TONER_LIQUID")

    if any(k in combined for k in [
        "크림/밤", "크림", "수딩크림", "재생크림", "보습크림", "수분크림",
        "로션/에멀젼", "로션", "에멀젼", "에멀전", "올인원", "페이스오일"
    ]):
        if any(k in combined for k in ["로션", "에멀젼", "에멀전", "올인원"]):
            return ("CREAM", "LOTION")
        return ("CREAM", "CREAM")

    return ("SERUM", "SERUM")


def infer_db_category_and_item_type(product_name: str, item_type_hint: str = "AUTO") -> Tuple[str, str]:
    hint = (item_type_hint or "AUTO").strip()
    explicit_map = {
        "스킨/토너": ("Skincare", "스킨/토너"),
        "토너패드": ("Skincare", "토너패드"),
        "에센스/세럼/앰플": ("Skincare", "에센스/세럼/앰플"),
        "로션/에멀젼": ("Skincare", "로션/에멀젼"),
        "크림/밤": ("Skincare", "크림/밤"),
        "클렌징 폼/젤": ("Cleansing", "클렌징 폼/젤"),
        "클렌징 오일": ("Cleansing", "클렌징 오일"),
        "클렌징 오일/밤": ("Cleansing", "클렌징 오일"),
        "클렌징 밤": ("Cleansing", "클렌징 밤"),
        "선크림/로션": ("SunCare", "선크림/로션"),
        "선세럼/선젤/플루이드": ("SunCare", "선세럼/선젤/플루이드"),
        "선세럼/선젤": ("SunCare", "선세럼/선젤/플루이드"),
        "선스틱": ("SunCare", "선스틱"),
        "시트마스크": ("MaskPack", "시트마스크"),
        "시트마스크/팩": ("MaskPack", "시트마스크"),
        "워시오프팩/슬리핑팩": ("MaskPack", "워시오프팩")
    }
    if hint in explicit_map:
        return explicit_map[hint]

    macro_role, sub_role = classify_cosmetic_role("", product_name, "")
    if macro_role == "CLEANSING":
        return ("Cleansing", "클렌징 오일" if sub_role == "CLEANSING_OIL" else "클렌징 폼/젤")
    if macro_role == "SUNCARE":
        if any(k in product_name for k in ["선세럼", "선젤", "플루이드"]):
            return ("SunCare", "선세럼/선젤/플루이드")
        if "선스틱" in product_name:
            return ("SunCare", "선스틱")
        return ("SunCare", "선크림/로션")
    if macro_role == "MASKPACK":
        return ("MaskPack", "시트마스크")
    if macro_role == "TONER":
        return ("Skincare", "토너패드" if sub_role == "TONER_PAD" else "스킨/토너")
    if macro_role == "CREAM":
        return ("Skincare", "로션/에멀젼" if sub_role == "LOTION" else "크림/밤")
    return ("Skincare", "에센스/세럼/앰플")


def upsert_cosmetic_to_db(
    conn: sqlite3.Connection,
    brand: str,
    product_name: str,
    sub_category: str,
    item_type: str,
    raw_ingredients_text: str,
    active_ingredients: List[str]
) -> Tuple[int, str]:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(products)")
    table_cols = {row["name"] for row in cur.fetchall()}

    cur.execute("""
        SELECT product_id FROM products
        WHERE category = 'cosmetic' AND company = ? AND product_name = ?
        LIMIT 1
    """, (brand, product_name))
    existing = cur.fetchone()

    primary_fn = f"핵심 유효성분: {', '.join(active_ingredients[:5])}" if active_ingredients else "사용자 라벨 스캔 등록 화장품"
    report_code = f"USER_OCR_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if existing:
        pid = int(existing["product_id"])
        update_pairs = [
            ("sub_category", sub_category),
            ("item_type", item_type),
            ("raw_ingredients_text", raw_ingredients_text)
        ]
        if "primary_function" in table_cols:
            update_pairs.append(("primary_function", primary_fn))
        set_clause = ", ".join([f"{col} = ?" for col, _ in update_pairs if col in table_cols])
        vals = [val for col, val in update_pairs if col in table_cols] + [pid]
        cur.execute(f"UPDATE products SET {set_clause} WHERE product_id = ?", vals)
        conn.commit()
        return (pid, "UPDATED")

    candidate_data = {
        "report_no": report_code,
        "company": brand,
        "product_name": product_name,
        "category": "cosmetic",
        "sub_category": sub_category,
        "item_type": item_type,
        "primary_function": primary_fn,
        "caution": "화장품 라벨 OCR 스캔을 통해 자동 적재된 제품입니다.",
        "raw_ingredients_text": raw_ingredients_text
    }
    insert_cols = [col for col in candidate_data.keys() if col in table_cols]
    placeholders = ", ".join(["?"] * len(insert_cols))
    col_names = ", ".join(insert_cols)
    insert_vals = [candidate_data[col] for col in insert_cols]

    cur.execute(f"INSERT INTO products ({col_names}) VALUES ({placeholders})", insert_vals)
    conn.commit()
    return (int(cur.lastrowid), "INSERTED")


# ==============================================================================
# [3] 6대 피부 유형(Skin Type) & 7대 피부 고민(Concern) & 이너뷰티 도메인 지식베이스
# ==============================================================================
SKIN_TYPE_PROFILES: Dict[str, Dict[str, Any]] = {
    "건성": {
        "label": "건성 (유·수분 모두 부족)",
        "good_ings": ["세라마이드", "히알루론산", "판테놀", "스쿠알란", "시어버터", "콜레스테롤"],
        "avoid_ings": ["변성알코올(에탄올)", "고농도 BHA(살리실산)", "설페이트계 강알칼리 세정제"],
        "prefer_keywords": ["보습", "크림", "밤", "장벽", "세라마이드", "히알루론", "판테놀", "밀크", "오일"],
        "avoid_keywords": ["피지", "노세범", "매트", "딥클렌징", "필링"],
        "inner_nutrients": ["히알루론산", "오메가3", "저분자 콜라겐"],
        "type_guide": "건성 피부는 피지 분비와 수분 보유력이 모두 낮으므로 세라마이드·스쿠알란 기반의 보습막 형성이 핵심이며, 피지를 과도하게 뺏는 BHA나 알코올 성분을 피해야 합니다."
    },
    "지성": {
        "label": "지성 (피지 분비 활발·번들거림)",
        "good_ings": ["나이아신아마이드", "BHA", "PHA", "어성초", "티트리", "녹차추출물", "판테놀"],
        "avoid_ings": ["고함량 시어버터·코코넛오일", "미네랄오일", "무거운 오일밤 제형"],
        "prefer_keywords": ["수딩", "젤", "워터", "산뜻", "약산성", "어성초", "티트리", "모공", "세럼", "패드"],
        "avoid_keywords": ["리치", "고영양 밤", "오일밤", "고보습 오일"],
        "inner_nutrients": ["판토텐산", "아연(징크)", "비타민B군"],
        "type_guide": "지성 피부는 과잉 피지로 인해 모공이 막히기 쉬우므로 무거운 오일·밤 크림 대신 산뜻한 수분 세럼·수딩젤 제형과 피지 조절 성분(나이아신아마이드·BHA·판토텐산)을 중심으로 구성해야 합니다."
    },
    "수부지": {
        "label": "수부지 (속건조·겉유분 복합)",
        "good_ings": ["히알루론산", "판테놀", "나이아신아마이드", "병풀추출물(시카)", "베타글루칸", "PHA"],
        "avoid_ings": ["고점도 오일/바세린(겉돌면서 모공 막힘)", "고농도 알코올(속건조 악화)"],
        "prefer_keywords": ["수분", "다이브인", "히알루론", "수딩크림", "진정", "선세럼", "약산성", "토너"],
        "avoid_keywords": ["헤비 크림", "오일밤", "강알칼리"],
        "inner_nutrients": ["히알루론산", "판토텐산", "비타민B군"],
        "type_guide": "수부지(수분부족형 지성) 피부는 속수분이 부족해 피지가 과다 분비되는 상태이므로, 유분기가 적은 저분자 히알루론산·판테놀로 속보습을 채우고 나이아신아마이드·판토텐산으로 겉유분을 잡아주어야 합니다."
    },
    "복합성": {
        "label": "복합성 (T존 지성 · U존 건성)",
        "good_ings": ["판테놀", "나이아신아마이드", "히알루론산", "병풀추출물(시카)", "세라마이드", "PHA"],
        "avoid_ings": ["전면 고농도 산성 박리제", "과도한 중성 오일"],
        "prefer_keywords": ["밸런스", "수분", "진정", "패드", "세럼", "수딩", "마일드"],
        "avoid_keywords": ["극건성용 밤"],
        "inner_nutrients": ["비타민B군", "히알루론산", "판토텐산"],
        "type_guide": "복합성 피부는 유분이 많은 T존(이마·코)에는 산뜻한 토너패드·세럼을, 건조한 U존(볼·턱)에는 세라마이드·판테놀 보습 크림을 레이어링하는 부위별 밸런스 케어가 가장 효과적입니다."
    },
    "민감성": {
        "label": "민감성 (외부 자극·홍조·장벽 약화)",
        "good_ings": ["병풀추출물(시카)", "판테놀", "알란토인", "세라마이드", "쑥추출물", "마데카소사이드"],
        "avoid_ings": ["고농도 AHA/BHA", "고농도 레티놀", "인공향료·에센셜오일", "멘톨·알코올"],
        "prefer_keywords": ["시카", "마데카", "더마", "약산성", "진정", "무기자차", "마일드", "장벽"],
        "avoid_keywords": ["필링", "고함량 비타민C", "스크럽", "레티놀"],
        "inner_nutrients": ["유산균(프로바이오틱스)", "비타민D", "오메가3"],
        "type_guide": "민감성 피부는 장벽이 얇아 각질 제거 성분(AHA/BHA)이나 고농도 레티놀에 쉽게 붉어지므로, 시카·판테놀·세라마이드 위주의 저자극 진정 루틴과 면역·장벽 영양소(유산균·비타민D) 보충이 필수입니다."
    },
    "여드름성": {
        "label": "여드름성 (트러블·면포·피지 과다)",
        "good_ings": ["병풀추출물(시카)", "티트리", "어성초", "BHA", "징크", "나이아신아마이드", "판테놀"],
        "avoid_ings": ["시어버터·코코넛오일·미리스틱애씨드(모공 막힘 유발)", "고유분 영양크림"],
        "prefer_keywords": ["어성초", "티트리", "시카", "블레미쉬", "트러블", "약산성", "논코메도", "수딩"],
        "avoid_keywords": ["오일", "영양크림", "고보습 밤"],
        "inner_nutrients": ["아연(징크)", "판토텐산", "유산균(프로바이오틱스)"],
        "type_guide": "여드름성 피부는 모공을 막는 오일·스테아릭·시어버터 성분을 철저히 배제(논코메도제닉)하고, 어성초·티트리·BHA 외용 케어와 피지·면역 관리 영양소(판토텐산·아연)를 병행해야 합니다."
    }
}

CONCERN_CRITERIA_MAP: Dict[str, Dict[str, Any]] = {
    "수분/속건조": {
        "good_ings": ["히알루론산", "판테놀", "세라마이드", "베타글루칸", "스쿠알란", "글리세린"],
        "avoid_ings": ["고농도 변성알코올(에탄올)", "고농도 AHA/BHA(과도한 각질 탈락)", "강한 계면활성제(SLS)"],
        "prefer_keywords": ["수분", "다이브인", "히알루론", "보습", "자작나무", "세라마이드"],
        "inner_nutrients": ["히알루론산", "저분자 콜라겐"],
        "summary_guide": "저분자·고분자 히알루론산과 판테놀·세라마이드가 함께 들어있어 수분 공급과 장벽 보습막을 동시에 형성하고, 수분을 증발시키는 알코올이나 강한 산성 성분이 없는 제품이 좋습니다."
    },
    "미백/잡티": {
        "good_ings": ["나이아신아마이드", "비타민C", "글루타치온", "알부틴", "트라넥사믹애씨드", "비타민E"],
        "avoid_ings": ["인공향료·색소(광감작 주의)", "아침 고농도 레티놀·AHA 동시 사용"],
        "prefer_keywords": ["비타", "잡티", "미백", "브라이트닝", "흔적", "글루타치온", "나이아신"],
        "inner_nutrients": ["비타민C", "글루타치온"],
        "summary_guide": "멜라닌 이동을 억제하는 나이아신아마이드와 항산화 비타민C·글루타치온이 함유되고, 자외선 노출 시 자극을 줄 수 있는 광감작 향료나 고자극 박리 성분이 배제된 제품이 좋습니다."
    },
    "트러블/피지": {
        "good_ings": ["병풀추출물(시카)", "티트리", "어성초", "BHA", "PHA", "판테놀", "나이아신아마이드", "징크"],
        "avoid_ings": ["고함량 시어버터·코코넛오일(모공 막힘 유발)", "이소프로필미리스테이트", "인공향료"],
        "prefer_keywords": ["티트리", "어성초", "시카", "블레미쉬", "진정", "트러블", "녹두"],
        "inner_nutrients": ["판토텐산", "아연(징크)"],
        "summary_guide": "어성초·시카·티트리 및 저자극 살리실산(BHA)/PHA로 피지와 염증을 케어하고, 모공을 막아 좁쌀·화농성 트러블을 유발하는 무거운 오일(코메도제닉) 성분이 없는 논코메도제닉 제품이 좋습니다."
    },
    "모공/각질": {
        "good_ings": ["BHA", "AHA", "PHA", "나이아신아마이드", "녹차추출물", "판테놀", "펩타이드"],
        "avoid_ings": ["굵은 물리적 스크럽 알갱이", "고점도 미네랄오일", "고농도 산성 성분 중복 레이어링"],
        "prefer_keywords": ["모공", "패드", "각질", "제로", "피지", "녹두", "바하"],
        "inner_nutrients": ["판토텐산", "비타민B군"],
        "summary_guide": "모공 속 피지를 녹이는 BHA(살리실산)·PHA와 모공 탄력을 돕는 나이아신아마이드가 적정 농도로 배합되고, 피부 표면에 미세 상처를 내는 물리적 스크럽이나 무거운 오일이 없는 제품이 좋습니다."
    },
    "탄력/주름": {
        "good_ings": ["레티놀", "펩타이드", "아데노신", "콜라겐", "세라마이드", "비타민E", "PDRN"],
        "avoid_ings": ["레티놀 사용 시 AHA/BHA 동시 도포", "건조함을 유발하는 알코올류"],
        "prefer_keywords": ["탄력", "콜라겐", "레티놀", "펩타이드", "리프팅", "안티에이징", "흔적 리페어"],
        "inner_nutrients": ["저분자 콜라겐", "코엔자임Q10", "비타민C"],
        "summary_guide": "콜라겐 합성과 턴오버를 돕는 레티놀·펩타이드·아데노신·PDRN과 보습 장벽 성분이 함께 들어있고, 탄력 저하의 원인인 건조함을 유발하는 알코올이나 자극적인 각질산이 배제된 제품이 좋습니다."
    },
    "홍조/열감": {
        "good_ings": ["병풀추출물(시카)", "알란토인", "판테놀", "알로에", "아줄렌", "세라마이드"],
        "avoid_ings": ["멘톨·페퍼민트(자극성 쿨링제)", "고농도 알코올", "고함량 순수 비타민C·AHA"],
        "prefer_keywords": ["수딩", "진정", "쿨링", "시카", "알로에", "자작나무", "수분"],
        "inner_nutrients": ["유산균(프로바이오틱스)", "오메가3"],
        "summary_guide": "피부 온도를 부드럽게 낮추고 모세혈관 자극을 진정시키는 시카·아줄렌·판테놀이 포함되고, 일시적 화끈거림을 유발하는 멘톨·알코올·강산성 성분이 없는 저자극 약산성 제품이 좋습니다."
    },
    "장벽/민감진정": {
        "good_ings": ["세라마이드", "판테놀", "콜레스테롤", "지방산", "병풀추출물(시카)", "스쿠알란", "알란토인"],
        "avoid_ings": ["AHA/BHA/레티놀(장벽 손상 시 주의)", "인공향료·에센셜오일", "설페이트계 계면활성제"],
        "prefer_keywords": ["장벽", "세라마이드", "더마", "판테놀", "시카", "리페어", "아토"],
        "inner_nutrients": ["유산균(프로바이오틱스)", "비타민D", "오메가3"],
        "summary_guide": "피부 지질 구조와 유사한 세라마이드·콜레스테롤·지방산 및 판테놀이 함유되어 무너진 장벽을 메워주고, 각질 용해 성분(AHA/BHA)과 인공향료가 완전히 배제된 순한 처방의 제품이 좋습니다."
    }
}

SUPPLEMENT_DOSING_GUIDE: Dict[str, Dict[str, str]] = {
    "비타민C": {
        "when_slot": "☀️ 아침 식사 직후 (또는 점심 식후)",
        "when_category": "MORNING",
        "how_to_take": "산성을 띠므로 속쓰림 방지를 위해 반드시 식사 직후에 충분한 물(200ml)과 함께 섭취하세요.",
        "why_recommended": "자외선(UV) 노출로 생성되는 활성산소를 제거하고 콜라겐 합성 및 멜라닌 색소 침착 방어를 돕습니다.",
        "caution_tip": "글루타치온·비타민E·콜라겐과 함께 섭취 시 항산화 및 탄력 시너지가 극대화됩니다.",
        "search_keyword": "비타민C"
    },
    "판토텐산": {
        "when_slot": "☀️ 아침 또는 점심 식사 직후",
        "when_category": "MORNING",
        "how_to_take": "수용성 비타민(비타민 B5)으로 아침·점심 식사 후 물과 함께 꾸준히 섭취하세요.",
        "why_recommended": "코엔자임A 합성을 도와 피지선의 과도한 피지 분비를 조절하고 지성·수부지·트러블 피부의 유수분 밸런스를 돕습니다.",
        "caution_tip": "비타민B군 복합제나 아연(징크)과 함께 섭취하면 피지 및 트러블 관리에 시너지가 납니다.",
        "search_keyword": "판토텐산"
    },
    "비타민B군": {
        "when_slot": "☀️ 아침 식사 직후",
        "when_category": "MORNING",
        "how_to_take": "에너지 대사를 활성화하므로 하루 일과를 시작하는 아침 식사 직후에 물과 함께 섭취하세요.",
        "why_recommended": "수면 부족·음주·피로로 저하된 세포 재생 에너지를 공급하고 피부 컨디션을 빠르게 회복시킵니다.",
        "caution_tip": "늦은 밤에 섭취하면 신경이 각성되어 수면에 방해가 될 수 있으니 오전~점심 사이에 섭취하세요.",
        "search_keyword": "비타민B"
    },
    "글루타치온": {
        "when_slot": "☀️ 아침 식전(미온수) 또는 아침 식후 (비타민C 병용)",
        "when_category": "MORNING",
        "how_to_take": "필름 형태는 입안 점막에 붙여 녹여 드시고, 정제/캡슐 형태는 아침에 비타민C와 함께 섭취하세요.",
        "why_recommended": "자외선과 피로로 칙칙해진 피부 톤(미백/잡티)을 맑게 가꾸고 산화 스트레스를 해독합니다.",
        "caution_tip": "비타민C, 셀레늄, 밀크씨슬과 함께 복용하면 체내 글루타치온 재생 효율이 높아집니다.",
        "search_keyword": "글루타치온"
    },
    "유산균(프로바이오틱스)": {
        "when_slot": "☀️ 아침 공복 (기상 직후 물 1컵과 함께)",
        "when_category": "MORNING",
        "how_to_take": "위산이 적은 아침 기상 직후 공복에 미온수 1컵을 먼저 마신 뒤 섭취하면 장까지 생존율이 가장 높습니다.",
        "why_recommended": "장-피부 축(Gut-Skin Axis) 면역 밸런스를 조절하여 민감성·홍조·여드름성 피부의 염증 반응을 완화하고 장벽을 강화합니다.",
        "caution_tip": "뜨거운 물이나 항생제와 동시에 복용하면 유익균이 사멸할 수 있으니 시간차를 두세요.",
        "search_keyword": "프로바이오틱스"
    },
    "아연(징크)": {
        "when_slot": "🌤️ 점심 또는 저녁 식사 직후",
        "when_category": "LUNCH_DINNER",
        "how_to_take": "공복에 섭취하면 속쓰림이나 울렁거림이 발생할 수 있으므로 반드시 식사 직후에 섭취하세요.",
        "why_recommended": "피부 조직 재생과 면역 기능을 정상화하여 여드름성·지성·트러블 피부의 염증 진정과 상처 회복을 돕습니다.",
        "caution_tip": "고함량 철분제나 칼슘제와 동시에 먹으면 흡수 경쟁이 일어날 수 있으니 식사 시간을 나누어 드세요.",
        "search_keyword": "아연"
    },
    "오메가3": {
        "when_slot": "🌤️ 점심 또는 저녁 식사 직후 (기름기 있는 식사 후)",
        "when_category": "LUNCH_DINNER",
        "how_to_take": "지용성 지방산이므로 담즙산이 분비되는 점심·저녁 식사 직후에 바로 섭취해야 흡수율이 가장 높고 어취가 올라오지 않습니다.",
        "why_recommended": "건성·민감성·장벽 약화 피부에 필수 지방산을 공급하여 세포막 보습력을 높이고 홍조·염증을 완화합니다.",
        "caution_tip": "비타민E·코엔자임Q10·비타민D 등 다른 지용성 영양제와 식후에 함께 드시면 시너지가 좋습니다.",
        "search_keyword": "오메가"
    },
    "비타민D": {
        "when_slot": "🌤️ 아침 또는 점심 식사 직후",
        "when_category": "LUNCH_DINNER",
        "how_to_take": "지용성 비타민이므로 식사 직후(지방이 포함된 식사 후)에 섭취하면 흡수율이 50% 이상 높아집니다.",
        "why_recommended": "피부 표피 세포의 분화와 장벽 면역 기능을 조절하여 민감성·건성 피부 장벽 회복과 뼈 건강을 동시에 돕습니다.",
        "caution_tip": "칼슘 및 마그네슘과 함께 섭취하면 체내 흡수 및 대사 효율이 크게 높아집니다.",
        "search_keyword": "비타민D"
    },
    "칼슘": {
        "when_slot": "🌙 저녁 식사 직후",
        "when_category": "NIGHT",
        "how_to_take": "위산이 분비되어야 흡수가 잘 되므로 저녁 식사 직후에 충분한 물과 함께 섭취하세요.",
        "why_recommended": "신경과 근육 기능을 유지하고 피부 표피층의 칼슘 기울기(Calcium Gradient)를 정상화하여 피부 장벽 형성을 돕습니다.",
        "caution_tip": "비타민D·마그네슘과 함께 섭취하면 흡수율이 높아지며, 철분제와는 시간차를 두고 드세요.",
        "search_keyword": "칼슘"
    },
    "비타민A": {
        "when_slot": "🌤️ 점심 또는 저녁 식사 직후",
        "when_category": "LUNCH_DINNER",
        "how_to_take": "지용성 비타민이므로 식사 직후에 일일 권장량 이내로 섭취하세요.",
        "why_recommended": "상피세포의 성장과 발달, 피부 점막 유지 및 각질 턴오버 주기를 정상화합니다.",
        "caution_tip": "⚠️ 외용 레티놀(비타민A) 화장품과 고함량 비타민A 영양제를 중복 사용할 경우 과잉 자극·건조 우려가 있으니 각별히 주의하세요.",
        "search_keyword": "비타민A"
    },
    "코엔자임Q10": {
        "when_slot": "🌤️ 점심 또는 저녁 식사 직후",
        "when_category": "LUNCH_DINNER",
        "how_to_take": "지용성 항산화 성분이므로 담즙산이 분비되는 식사 직후에 섭취해야 흡수율이 높습니다.",
        "why_recommended": "자외선 노출 및 노화로 발생하는 활성산소를 막고 탄력/주름 피부에 세포 에너지를 공급합니다.",
        "caution_tip": "오메가3나 비타민E와 식후에 함께 복용하시면 흡수율이 더욱 좋아집니다.",
        "search_keyword": "코엔자임"
    },
    "아스타잔틴": {
        "when_slot": "🌤️ 아침 또는 점심 식사 직후",
        "when_category": "LUNCH_DINNER",
        "how_to_take": "지용성 카로티노이드 성분이므로 기름기가 있는 식사 직후에 섭취하세요.",
        "why_recommended": "자외선(UV)에 의한 광노화 및 피부 수분 손실을 방어하고 눈의 피로를 함께 케어합니다.",
        "caution_tip": "제품별 일일 권장량(4~12mg)을 지켜 꾸준히 복용하세요.",
        "search_keyword": "아스타잔틴"
    },
    "밀크씨슬": {
        "when_slot": "🌤️ 점심 또는 저녁 식사 후",
        "when_category": "LUNCH_DINNER",
        "how_to_take": "실리마린 성분의 위장 자극을 줄이고 흡수율을 높이기 위해 식사 직후 물과 함께 섭취하세요.",
        "why_recommended": "음주 및 피로로 저하된 간 해독 기능을 도와 독소로 인한 피부 트러블과 칙칙한 안색을 예방합니다.",
        "caution_tip": "비타민B군과 함께 섭취하면 음주·피로 회복 시너지가 높습니다.",
        "search_keyword": "밀크씨슬"
    },
    "루테인": {
        "when_slot": "🌤️ 아침 또는 점심 식사 직후",
        "when_category": "LUNCH_DINNER",
        "how_to_take": "지용성 성분이므로 식사 직후에 물과 함께 섭취하세요.",
        "why_recommended": "황반 색소 밀도를 유지하여 눈 건강을 돕고 블루라이트 및 산화 스트레스 방어를 지원합니다.",
        "caution_tip": "오메가3나 아스타잔틴과 함께 식후 복용 시 흡수율이 좋습니다.",
        "search_keyword": "루테인"
    },
    "마그네슘": {
        "when_slot": "🌙 저녁 식후 또는 취침 1시간 전",
        "when_category": "NIGHT",
        "how_to_take": "근육 이완과 신경 안정, 수면의 질 개선을 위해 저녁 식사 후 또는 잠들기 30분~1시간 전에 미온수와 함께 섭취하세요.",
        "why_recommended": "수면 부족 및 스트레스에 의한 피부 장벽 회복 지연과 피로를 완화합니다.",
        "caution_tip": "공복에 고함량 섭취 시 묽은 변을 유발할 수 있으니 위장이 약하면 저녁 식사 직후에 드세요.",
        "search_keyword": "마그네슘"
    },
    "L-테아닌": {
        "when_slot": "🌙 취침 30분~1시간 전",
        "when_category": "NIGHT",
        "how_to_take": "알파(α)파 생성을 도와 긴장을 완화하므로 잠들기 전 미온수와 함께 섭취하세요.",
        "why_recommended": "수면 부족 및 스트레스 호르몬 과다로 인한 성인 트러블·피지 분비를 완화합니다.",
        "caution_tip": "카페인 음료(커피, 에너지드링크, 녹차)와 함께 마시면 효과가 반감되므로 카페인을 피해 드세요.",
        "search_keyword": "테아닌"
    },
    "히알루론산": {
        "when_slot": "🌙 저녁 식후 또는 취침 1시간 전",
        "when_category": "NIGHT",
        "how_to_take": "수분을 끌어당기므로 섭취 시 반드시 물 1~2컵(250ml 이상)을 충분히 함께 마셔주세요.",
        "why_recommended": "건성·수부지 피부 및 속건조 완화를 위해 진피층 수분 보유력을 높여줍니다.",
        "caution_tip": "하루 수분 섭취량을 충분히 유지해 주시는 것이 핵심입니다.",
        "search_keyword": "히알루론산"
    },
    "저분자 콜라겐": {
        "when_slot": "🌙 취침 1시간 전 또는 아침 공복",
        "when_category": "NIGHT",
        "how_to_take": "피부 재생 세포 분열이 활발한 밤 시간대(취침 전)에 비타민C와 함께 충분한 물로 섭취하세요.",
        "why_recommended": "탄력/주름 개선 및 건성 피부의 진피 치밀도와 수분 보유력을 높여줍니다.",
        "caution_tip": "비타민C와 히알루론산을 함께 섭취하면 체내 콜라겐 합성이 촉진됩니다.",
        "search_keyword": "콜라겐"
    }
}

def extract_nutrients_from_user_supplement(name: str, ings: List[str], raw_text: str = "") -> List[str]:
    combined = f"{name} {' '.join(ings or [])} {raw_text or ''}"
    found = []
    keyword_map = [
        ("칼슘", ["칼슘", "해조칼슘", "탄산칼슘", "구연산칼슘"]),
        ("비타민D", ["비타민D", "비타민 D", "콜레칼시페롤"]),
        ("비타민A", ["비타민A", "비타민 A", "레티닐", "베타카로틴"]),
        ("비타민C", ["비타민C", "비타민 C", "아스코산", "아스코빅"]),
        ("비타민B군", ["비타민B", "비타민 B", "비오틴", "나이아신", "엽산", "티아민", "리보플라빈"]),
        ("판토텐산", ["판토텐산"]),
        ("마그네슘", ["마그네슘"]),
        ("밀크씨슬", ["밀크씨슬", "실리마린"]),
        ("히알루론산", ["히알루론"]),
        ("저분자 콜라겐", ["콜라겐"]),
        ("오메가3", ["오메가", "EPA", "DHA", "초임계 알티지"]),
        ("유산균(프로바이오틱스)", ["유산균", "프로바이오틱", "락토바실러스", "비피더스"]),
        ("아연(징크)", ["아연", "징크", "산화아연", "글루콘산아연"]),
        ("루테인", ["루테인", "지아잔틴"]),
        ("코엔자임Q10", ["코엔자임", "코큐텐"]),
        ("글루타치온", ["글루타치온"]),
        ("아스타잔틴", ["아스타잔틴", "헤마토코쿠스"]),
        ("L-테아닌", ["테아닌"])
    ]
    for std_nut, kw_list in keyword_map:
        if any(k in combined for k in kw_list):
            found.append(std_nut)
    if not found:
        found.append(ings[0] if ings else "종합기능성영양소")
    return list(dict.fromkeys(found))


CAUTION_INGREDIENT_META: Dict[str, Dict[str, str]] = {
    "레티놀": {"time_slot": "PM_ONLY", "time_label": "🌙 저녁 권장", "caution_reason": "자외선 노출 시 광분해 및 피부 자극 가능성이 있어 저녁 사용 권장", "freq_guide": "초기 주 2~3회 소량 사용"},
    "AHA": {"time_slot": "PM_ONLY", "time_label": "🌙 저녁 권장", "caution_reason": "수용성 각질 제거 성분으로 각질 탈락 후 자외선 민감도 증가 주의", "freq_guide": "주 1~3회 빈도 조절"},
    "BHA": {"time_slot": "PM_PREF", "time_label": "🌙 저녁 권장 (낮 사용 시 선케어 필수)", "caution_reason": "지용성 각질·피지 용해 성분으로 고농도 중복 사용 시 건조 주의", "freq_guide": "국소 도포 또는 격일 사용"},
    "PHA": {"time_slot": "BOTH", "time_label": "☀️/🌙 겸용 (자극 주의)", "caution_reason": "다른 산성 성분과 중첩 시 자극 가능성 존재", "freq_guide": "민감성 피부는 하루 1회 사용"},
    "비타민C": {"time_slot": "AM_PREF", "time_label": "☀️ 아침 권장 (선크림 병용) / 🌙 저녁 가능", "caution_reason": "낮은 pH로 인해 레티놀·각질산과 동시 사용 시 따가움 유발 가능", "freq_guide": "아침 사용 시 선크림 필수"},
    "나이아신아마이드": {"time_slot": "BOTH", "time_label": "☀️/🌙 아침·저녁 겸용", "caution_reason": "고농도 사용 시 강산성 비타민C와 겹치면 일시적 홍조 가능성", "freq_guide": "일반 함량은 매일 사용 가능"},
    "티트리": {"time_slot": "BOTH", "time_label": "☀️/🌙 아침·저녁 겸용", "caution_reason": "고농축 오일 형태일 경우 민감 피부 접촉성 자극 주의", "freq_guide": "국소 부위 진정 활용"}
}

PAIR_INTERACTION_RULES: List[Dict[str, Any]] = [
    {
        "ings": {"레티놀", "AHA"},
        "risk_level": "높은 주의",
        "risk_code": "HIGH_CAUTION",
        "short_desc": "강한 각질 박리 및 피부 장벽 손상 가능성으로 동시 사용 조정 필요",
        "rationale": "레티놀의 세포 턴오버 촉진 작용과 AHA의 표피 각질 용해 작용이 중첩되어 홍반, 각질 들뜸, 화끈거림 등 자극 가능성이 매우 높습니다.",
        "recommended_usage": "같은 날 동시에 바르지 마시고, 요일을 나누어 격일(예: 월/수 레티놀, 금 AHA)로 저녁에 분리 사용하세요.",
        "am_pm_split": "동시 사용 지양 (저녁 요일별 교차 사용 권장)",
        "conc_freq_note": "두 성분 모두 농도가 높을수록 자극이 배가되므로 주 1~2회 이하로 빈도를 제한하세요.",
        "sensitivity_note": "민감성·장벽 손상 피부는 두 제품의 병용을 중단하고 진정·보습제를 우선 사용하세요."
    },
    {
        "ings": {"레티놀", "BHA"},
        "risk_level": "높은 주의",
        "risk_code": "HIGH_CAUTION",
        "short_desc": "각질 제거 및 피지 억제 작용 중복으로 심한 건조·자극 주의",
        "rationale": "레티놀과 살리실산(BHA)을 한 루틴에서 레이어링하면 피부 보호막이 약화되고 수분 손실 및 자극 반응이 크게 증가할 수 있습니다.",
        "recommended_usage": "BHA 클렌저/토너는 아침 또는 레티놀을 쉬는 날 저녁에 사용하고, 레티놀 도포일에는 수분·장벽 제품만 함께 사용하세요.",
        "am_pm_split": "아침(BHA 저농도) / 저녁(레티놀) 분리 또는 격일 교차 사용",
        "conc_freq_note": "BHA 0.5% 이상 리브온 제형과 레티놀 병용 시 특히 주의가 필요합니다.",
        "sensitivity_note": "여드름성·민감성 피부는 국소 부위에만 제한적으로 적용하세요."
    },
    {
        "ings": {"레티놀", "비타민A"},
        "risk_level": "높은 주의",
        "risk_code": "HIGH_CAUTION",
        "short_desc": "외용 레티놀과 경구 비타민A 중복으로 과잉 노출 주의",
        "rationale": "화장품의 레티놀과 건강기능식품의 비타민A가 동시에 고함량으로 적용될 경우 지용성 비타민 과잉 및 피부 건조·박리 부작용 가능성이 커집니다.",
        "recommended_usage": "영양제의 비타민A 일일 권장량 초과 여부를 확인하고, 필요 시 복용 시기를 조절하세요.",
        "am_pm_split": "외용제는 저녁 소량 도포, 경구제는 함량 점검 필요",
        "conc_freq_note": "고농축 레티놀 사용 시 고함량 비타민A 단일제 병용 주의",
        "sensitivity_note": "임산부 및 가임기 여성은 레티놀 및 고함량 비타민A 중복을 피해야 합니다."
    },
    {
        "ings": {"레티놀", "비타민C"},
        "risk_level": "주의",
        "risk_code": "CAUTION",
        "short_desc": "pH 산도 차이 및 자극 가능성을 고려해 아침·저녁 시간대 분리 권장",
        "rationale": "순수 비타민C는 산성 환경에서 안정적이며 레티놀은 중성 환경에서 작동합니다. 한 번에 겹쳐 바르면 피부 자극이 발생할 수 있습니다.",
        "recommended_usage": "비타민C는 아침(선크림과 함께)에 바르고, 레티놀은 저녁 나이트 케어로 분리해 사용하세요.",
        "am_pm_split": "☀️ 아침: 비타민C / 🌙 저녁: 레티놀 시분할 권장",
        "conc_freq_note": "순수 아스코빅애씨드 10% 이상 제품은 레티놀과 반드시 시간을 분리하세요.",
        "sensitivity_note": "따가움이 느껴지면 보습 크림을 먼저 도포하세요."
    },
    {
        "ings": {"비타민C", "AHA"},
        "risk_level": "주의",
        "risk_code": "CAUTION",
        "short_desc": "산성(Low pH) 성분 중첩으로 인한 피부 따가움 및 빈도 조절 필요",
        "rationale": "비타민C와 AHA 모두 산성 성분으로, 동시에 도포하면 피부 표면 pH가 급격히 낮아져 따가움이나 홍반을 유발할 수 있습니다.",
        "recommended_usage": "비타민C는 아침에, AHA 토너/세럼은 저녁(주 1~2회)에 나누어 사용하세요.",
        "am_pm_split": "☀️ 아침: 비타민C / 🌙 저녁: AHA 분리 권장",
        "conc_freq_note": "두 제품 모두 고농도일 경우 매일 연속 사용을 피하세요.",
        "sensitivity_note": "민감성 및 홍조 피부는 산성 성분 중복 시 장벽 손상에 주의하세요."
    },
    {
        "ings": {"비타민C", "BHA"},
        "risk_level": "주의",
        "risk_code": "CAUTION",
        "short_desc": "자극 가능성을 고려해 사용 시간대 분리 또는 빈도 조절 필요",
        "rationale": "살리실산(BHA)과 비타민C를 연달아 바르면 각질층 자극과 건조함이 동반될 수 있습니다.",
        "recommended_usage": "아침에는 비타민C 세럼, 저녁에는 BHA 토너/패드로 시간대를 나누어 관리하세요.",
        "am_pm_split": "☀️ 아침: 비타민C / 🌙 저녁: BHA 분리 권장",
        "conc_freq_note": "워시오프 클렌저는 무방하나 리브온 토너/앰플은 분리를 권장합니다.",
        "sensitivity_note": "피부가 얇은 부위는 중복 도포를 피하세요."
    },
    {
        "ings": {"AHA", "BHA"},
        "risk_level": "주의",
        "risk_code": "CAUTION",
        "short_desc": "각질 제거 성분 중복 포함으로 과도한 각질 탈락 주의",
        "rationale": "서로 다른 두 제품에서 AHA와 BHA가 각각 공급되면 각질 제거 강도가 과도해져 장벽이 얇아질 수 있습니다.",
        "recommended_usage": "두 제품 중 하나만 선택해 주 1~2회 사용하거나 부위(T존/U존)를 나누어 사용하세요.",
        "am_pm_split": "저녁 시간대 주 1~2회 제한적 사용",
        "conc_freq_note": "2개 이상의 각질 제품을 겹쳐 바를 때 주의하세요.",
        "sensitivity_note": "세안 후 당김이 심해지면 사용 빈도를 줄이세요."
    }
]

SYNERGY_RULES: List[Tuple[set, str]] = [
    ({"레티놀", "세라마이드"}, "레티놀 사용 시 발생할 수 있는 건조함과 장벽 약화를 세라마이드가 효과적으로 보완해 주는 우수한 조합입니다."),
    ({"레티놀", "판테놀"}, "판테놀의 진정·보습 작용이 레티놀의 자극 가능성을 낮춰주어 함께 활용하기 좋습니다."),
    ({"레티놀", "히알루론산"}, "수분 공급 성분인 히알루론산이 레티놀의 턴오버 과정에서 부족해지기 쉬운 수분을 채워줍니다."),
    ({"AHA", "히알루론산"}, "각질 정돈 후 히알루론산 세럼을 레이어링하면 즉각적인 수분 공급으로 자극을 완화할 수 있습니다."),
    ({"BHA", "병풀추출물(시카)"}, "피지·각질 관리 후 시카(병풀) 성분이 피부를 진정시켜 트러블 케어에 적합한 조합입니다."),
    ({"비타민C", "비타민E"}, "수용성 항산화제(비타민C)와 지용성 항산화제(비타민E)가 서로의 산화를 막고 항산화 효능을 높여줍니다.")
]


def normalize_ing_name(ing: str) -> str:
    s = ing.strip()
    if any(k in s.upper() for k in ["AHA", "글라이콜릭", "락틱", "구연산", "씨트릭"]):
        return "AHA"
    if any(k in s.upper() for k in ["BHA", "살리실산", "살리실릭", "베타인살리실레이트"]):
        return "BHA"
    if any(k in s.upper() for k in ["PHA", "글루코노락톤", "락토바이오닉"]):
        return "PHA"
    if "레티놀" in s or "레티날" in s or "바쿠치올" in s:
        return "레티놀"
    if "비타민C" in s or "아스코빅" in s or "아스코빌" in s:
        return "비타민C"
    if "비타민A" in s or "레티닐팔미테이트" in s:
        return "비타민A"
    if "시카" in s or "병풀" in s or "마데카" in s:
        return "병풀추출물(시카)"
    if "히알루론" in s or "하이알루로닉" in s:
        return "히알루론산"
    if "세라마이드" in s:
        return "세라마이드"
    return s


def classify_product_time_slot(item_type: str, active_ings: List[str], name: str = "", category: str = "") -> Dict[str, str]:
    macro_role, _ = classify_cosmetic_role(item_type, name, category)
    norm_ings = [normalize_ing_name(i) for i in active_ings]

    if macro_role == "SUNCARE":
        return {"slot_code": "AM_ONLY", "slot_badge": "☀️ 아침 전용 (선케어)", "reason": "자외선 차단 제품으로 아침 마무리 단계에 사용합니다."}
    if macro_role == "CLEANSING":
        return {"slot_code": "BOTH", "slot_badge": "☀️ 아침 · 🌙 저녁 세안", "reason": "워시오프 세안제로 아침과 저녁 세안 단계에 모두 사용할 수 있습니다."}
    if any(k in norm_ings for k in ["레티놀", "AHA"]) or any(k in (item_type + " " + name) for k in ["나이트", "슬리핑"]):
        return {"slot_code": "PM_ONLY", "slot_badge": "🌙 저녁 집중 권장", "reason": "레티놀/AHA 등 밤 시간대 집중 케어 성분이 포함되어 저녁 사용을 권장합니다."}
    if "BHA" in norm_ings:
        return {"slot_code": "PM_PREF", "slot_badge": "🌙 저녁 권장 (낮 선케어 필수)", "reason": "살리실산(BHA) 리브온 제형으로 저녁 사용이 적합하며 낮 사용 시 자외선 차단이 필수입니다."}
    return {"slot_code": "BOTH", "slot_badge": "☀️ 아침 · 🌙 저녁 겸용", "reason": "아침과 저녁 루틴 모두 안전하게 활용할 수 있습니다."}


# ==============================================================================
# [4] 피부 유형(Skin Type) + 피부 고민(Concern) 결합 제형별 복수 화장품 매칭 & 추천 사유 생성기
# ==============================================================================
def map_step_to_db_filter(step_text: str) -> Tuple[str, str, str, List[str], str]:
    macro_role, sub_role = classify_cosmetic_role(step_text, step_text)
    if macro_role == "CLEANSING":
        if sub_role == "CLEANSING_OIL":
            return ("Cleansing", "CLEANSING", "CLEANSING_OIL", ["클렌징 오일", "클렌징 밤", "밀크"], "클렌징 오일/밤")
        return ("Cleansing", "CLEANSING", "CLEANSING_FOAM", ["클렌징 폼/젤", "폼"], "클렌징 폼/젤")
    if macro_role == "SUNCARE":
        return ("SunCare", "SUNCARE", "SUNCARE", ["선크림", "선세럼", "선젤", "선스틱", "선쿠션"], "선케어(선크림/선세럼)")
    if macro_role == "MASKPACK":
        return ("MaskPack", "MASKPACK", "MASKPACK", ["시트마스크", "워시오프팩", "슬리핑팩"], "마스크팩")
    if macro_role == "TONER":
        if sub_role == "TONER_PAD":
            return ("Skincare", "TONER", "TONER_PAD", ["토너패드", "패드"], "토너패드")
        return ("Skincare", "TONER", "TONER_LIQUID", ["스킨/토너", "토너"], "스킨/토너")
    if macro_role == "CREAM":
        if sub_role == "LOTION":
            return ("Skincare", "CREAM", "LOTION", ["로션/에멀젼", "올인원"], "로션/에멀젼")
        return ("Skincare", "CREAM", "CREAM", ["크림/밤", "크림"], "크림/밤")
    return ("Skincare", "SERUM", "SERUM", ["에센스/세럼/앰플", "세럼", "앰플"], "에센스/세럼/앰플")


def get_step_ingredient_guide(
    step_text: str,
    skin_type: str,
    concern: str,
    time_of_day: str,
    vanity_caution_ings: List[str]
) -> Dict[str, Any]:
    st_info = SKIN_TYPE_PROFILES.get(skin_type, SKIN_TYPE_PROFILES["수부지"])
    c_info = CONCERN_CRITERIA_MAP.get(concern, CONCERN_CRITERIA_MAP["수분/속건조"])
    sub_cat, macro_role, sub_role, _, std_type = map_step_to_db_filter(step_text)

    merged_good = list(dict.fromkeys(st_info["good_ings"][:3] + c_info["good_ings"][:3]))
    merged_avoid = list(dict.fromkeys(st_info["avoid_ings"][:2] + c_info["avoid_ings"][:2]))

    freq_badge = "☀️ 매일 아침 데일리" if time_of_day == "AM" else "🌙 매일 저녁 데일리"
    if any(k in step_text for k in ["레티놀", "AHA", "BHA", "각질", "고농축", "필링"]):
        freq_badge = "🌙 주 1~2회 저녁 소량" if skin_type == "민감성" else "🌙 주 2~3회 저녁 집중 (격일)"

    if macro_role == "CLEANSING":
        if skin_type in ["지성", "여드름성", "수부지"]:
            good = ["BHA(살리실산)", "어성초", "티트리", "녹차추출물", "판테놀"]
            avoid = ["모공을 막는 고점도 미네랄오일", "강알칼리성 계면활성제(SLS)"]
            tip = f"[{skin_type} · {concern}] 맞춤 세안을 위해 과잉 피지와 모공 노폐물을 비워내는 어성초·BHA 약산성 폼/젤이 적합합니다."
        else:
            good = ["판테놀", "병풀추출물(시카)", "세라마이드", "히알루론산"]
            avoid = ["설페이트계 계면활성제(SLS/SLES)", "높은 농도의 각질산", "인공향료"]
            tip = f"[{skin_type} · {concern}] 피부는 세안 후 속당김을 막는 보습 아미노산계 약산성 클렌저가 좋습니다."
    elif macro_role == "SUNCARE":
        if skin_type in ["민감성", "여드름성"]:
            good = ["무기자차(징크옥사이드)", "병풀추출물(시카)", "판테놀", "나이아신아마이드"]
            avoid = ["모공 막힘 유발 오일", "눈 시림 유발 유기자차 성분"]
        elif skin_type in ["건성", "수부지"]:
            good = ["히알루론산", "판테놀", "저자극 수분 선세럼/유기자차", "나이아신아마이드"]
            avoid = ["수분을 뺏는 과도한 파우더/알코올"]
        else:
            good = ["무기자차/혼합자차", "나이아신아마이드", "병풀추출물(시카)"]
            avoid = ["끈적이는 고점도 오일"]
        tip = f"[{skin_type}] 특성에 맞춰 들뜸이나 모공 막힘 없이 자외선(SPF50+)을 방어하는 선케어가 좋습니다."
        freq_badge = "☀️ 매일 아침 필수 (외출 30분 전)"
    elif macro_role == "CREAM":
        if skin_type in ["지성", "여드름성", "수부지"]:
            good = ["판테놀", "병풀추출물(시카)", "히알루론산", "나이아신아마이드"]
            avoid = ["고함량 시어버터·코코넛오일(모공 막힘)", "무거운 오일밤"]
            tip = f"[{skin_type}] 피부에는 무거운 유분 밤 대신 산뜻하게 수분막만 형성하는 수딩젤/수분크림 제형이 좋습니다."
        else:
            good = ["세라마이드", "판테놀", "스쿠알란", "콜레스테롤"] + c_info["good_ings"][:2]
            good = list(dict.fromkeys(good))
            avoid = merged_avoid[:3]
            tip = f"[{skin_type} · {concern}] 장벽 보호를 위해 [{', '.join(good[:3])}] 성분이 탄탄하게 들어있는 보습 크림이 좋습니다."
    else:
        good = merged_good[:5]
        avoid = list(merged_avoid[:3])
        if time_of_day == "AM" and "레티놀" in good:
            good = [g for g in good if g != "레티놀"] + ["펩타이드", "비타민C"]
            avoid.append("레티놀(낮 광독성 주의 → 저녁 전용)")
        if skin_type == "민감성":
            avoid.append("고농도 AHA/BHA/레티놀(민감 피부 자극 주의)")
        tip = f"[{skin_type} · {concern}] 케어를 위해 [{', '.join(good[:3])}] 성분이 포함되고 [{', '.join(avoid[:2])}] 성분이 배제된 제품이 좋습니다."

    return {
        "step_type": std_type,
        "good_ingredients": good,
        "avoid_ingredients": avoid,
        "criteria_tip": tip,
        "frequency_badge": freq_badge
    }


def find_recommended_cosmetics_from_db(
    conn: sqlite3.Connection,
    step_text: str,
    skin_type: str,
    concern: str,
    good_ings: List[str],
    avoid_ings: List[str],
    owned_product_ids: set,
    limit: int = 4
) -> List[Dict[str, Any]]:
    sub_cat, target_macro, target_sub, type_keywords, std_type = map_step_to_db_filter(step_text)
    st_info = SKIN_TYPE_PROFILES.get(skin_type, SKIN_TYPE_PROFILES["수부지"])
    c_info = CONCERN_CRITERIA_MAP.get(concern, CONCERN_CRITERIA_MAP["수분/속건조"])

    prefer_kws = st_info.get("prefer_keywords", []) + c_info.get("prefer_keywords", [])
    avoid_kws = st_info.get("avoid_keywords", [])

    cur = conn.cursor()
    cur.execute("""
        SELECT product_id, company, product_name, sub_category, item_type, raw_ingredients_text
        FROM products
        WHERE category = 'cosmetic' AND sub_category = ?
          AND product_id NOT IN (9001, 9002, 9003, 9004)
          AND product_name NOT IN ('샘플A', '샘플B', '샘플C', '샘플D')
    """, (sub_cat,))
    rows = cur.fetchall()

    scored_items = []
    for r in rows:
        if r["product_id"] in owned_product_ids:
            continue
        raw_txt = r["raw_ingredients_text"] or ""
        itype = r["item_type"] or ""
        pname = r["product_name"] or ""
        prod_macro, prod_sub = classify_cosmetic_role(itype, pname, r["sub_category"] or "")

        if prod_macro != target_macro:
            continue

        ext_ings = [normalize_ing_name(x) for x in extract_standard_ingredients(raw_txt)]
        ext_ings = list(dict.fromkeys(ext_ings))

        score = 0
        matched_kws = []
        if prod_sub == target_sub:
            score += 18
        elif any(k in itype or k in pname for k in type_keywords):
            score += 10

        matched_good = [g for g in good_ings if any(g in e or e in g for e in ext_ings) or g in raw_txt]
        score += len(matched_good) * 9

        for pkw in prefer_kws:
            if pkw in pname or pkw in itype:
                score += 11
                if pkw not in matched_kws:
                    matched_kws.append(pkw)

        for akw in avoid_kws:
            if akw in pname or akw in itype:
                score -= 18

        has_avoid_ing = False
        for av in avoid_ings:
            av_clean = av.split("(")[0].strip()
            if av_clean and (av_clean in ext_ings or av_clean in raw_txt):
                score -= 14
                has_avoid_ing = True

        fit_score = min(99, max(76, 80 + len(matched_good) * 4 + (4 if score > 35 else 0)))
        scored_items.append((score, fit_score, matched_good, matched_kws, has_avoid_ing, ext_ings, r))

    scored_items.sort(key=lambda x: (-x[0], x[6]["product_id"]))

    results = []
    for rank_idx, (score, fit_score, matched_good, matched_kws, has_avoid_ing, ext_ings, r) in enumerate(scored_items[:limit], start=1):
        raw_txt = r["raw_ingredients_text"] or ""
        display_ings = ext_ings[:6] if ext_ings else [s.strip() for s in raw_txt.split(",")[:4] if s.strip()]

        reason_parts = []
        if matched_good:
            reason_parts.append(f"[{skin_type} · {concern}] 권장 성분({', '.join(matched_good[:3])}) 함유")
        elif display_ings:
            reason_parts.append(f"보습·진정 핵심 성분({', '.join(display_ings[:2])}) 배합")
        if matched_kws:
            reason_parts.append(f"'{matched_kws[0]}' 맞춤 제형 라인")
        if not has_avoid_ing:
            reason_parts.append(f"{skin_type} 기피·주의 성분 배제")

        recommend_reason = " · ".join(reason_parts) if reason_parts else f"[{skin_type}] 피부의 [{std_type}] 단계에 적합한 균형 처방 제품"

        results.append({
            "rank": rank_idx,
            "product_id": r["product_id"],
            "brand": r["company"],
            "name": r["product_name"],
            "category": r["sub_category"],
            "item_type": r["item_type"] or std_type,
            "fit_score": fit_score,
            "recommend_reason": recommend_reason,
            "active_ingredients": display_ings,
            "matched_good_ingredients": matched_good,
            "sample_ingredients": make_sample_ings(raw_txt, 4),
            "raw_ingredients_text": raw_txt
        })
    return results


def match_owned_product_for_step(
    step_title: str,
    skin_type: str,
    time_of_day: str,
    vanity_items: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    _, target_macro, target_sub, _, _ = map_step_to_db_filter(step_title)

    exact_sub_matches = []
    macro_matches = []

    for v in vanity_items:
        v_name = v.get("name", "")
        v_type = v.get("item_type", "")
        v_macro, v_sub = classify_cosmetic_role(v_type, v_name, v.get("category", ""))
        if v_macro != target_macro:
            continue

        v_ings = [normalize_ing_name(i) for i in v.get("active_ingredients", [])]
        if time_of_day == "AM" and v_macro != "CLEANSING" and any(k in v_ings for k in ["레티놀", "AHA"]):
            continue
        if time_of_day == "PM" and v_macro == "SUNCARE":
            continue
        if skin_type == "민감성" and any(k in v_ings for k in ["레티놀", "AHA"]):
            continue
        if skin_type in ["지성", "여드름성"] and time_of_day == "AM" and any(k in (v_name + v_type) for k in ["밤", "오일", "나이트"]):
            continue

        if v_sub == target_sub:
            exact_sub_matches.append(v)
        else:
            macro_matches.append(v)

    if exact_sub_matches:
        return exact_sub_matches[0]
    if macro_matches:
        return macro_matches[0]
    return None


def build_detailed_routine_steps(
    conn: sqlite3.Connection,
    step_list: List[str],
    skin_type: str,
    concern: str,
    time_of_day: str,
    vanity_items: List[Dict[str, Any]],
    vanity_caution_ings: List[str],
    owned_product_ids: set,
    env_summary_text: str = ""
) -> Tuple[List[Dict[str, Any]], set]:
    detailed_steps = []
    used_vanity_keys = set()

    for idx, step_title in enumerate(step_list):
        _, macro_role, _, _, std_type = map_step_to_db_filter(step_title)
        guide = get_step_ingredient_guide(step_title, skin_type, concern, time_of_day, vanity_caution_ings)

        owned_match = match_owned_product_for_step(step_title, skin_type, time_of_day, vanity_items)
        if owned_match:
            v_key = f"{owned_match.get('product_id', 0)}_{owned_match.get('name', '')}"
            used_vanity_keys.add(v_key)

        db_recs = find_recommended_cosmetics_from_db(
            conn, step_title, skin_type, concern, guide["good_ingredients"], guide["avoid_ingredients"], owned_product_ids, limit=4
        )

        time_kor = "아침(AM) 방어" if time_of_day == "AM" else "저녁(PM) 집중 회복"
        env_reason = (
            f"실제 기상 데이터({env_summary_text}) 및 [{skin_type}] 피부 특성을 반영했습니다."
            if env_summary_text
            else f"실시간 기상 데이터 미수신으로 [{skin_type}] 피부와 {time_kor} 목적 기준으로 배치했습니다."
        )

        if owned_match:
            prod_ings = ", ".join(owned_match.get("active_ingredients", [])[:4]) or "보습·진정 성분"
            fit_score = 95 if skin_type in ["수부지", "복합성"] else 93
            xai_reasons = [
                f"내 화장대 보유 제품 중 [{skin_type}] 피부의 [{std_type}] 단계에 적합하여 1순위로 배치했습니다.",
                f"선택하신 [{concern}] 고민 관리에 유효한 성분({prod_ings})이 확인되었습니다.",
                env_reason
            ]
            xai_evidence = [
                f"[{skin_type}] 제형 적합성 및 6대 역할군({macro_role}) 기준 충족",
                f"{time_kor} 시간대 광독성·산도(pH) 안전성 검증 통과"
            ]
            xai_extra = f"보유 제품으로 우선 관리할 수 있으며, [{skin_type}] 컨디션에 따라 사용량을 조절하세요."
        else:
            top_rec = db_recs[0] if db_recs else {}
            fit_score = top_rec.get("fit_score", 91)
            rec_ings = ", ".join((top_rec.get("active_ingredients") or guide["good_ingredients"])[:3])
            xai_reasons = [
                f"현재 내 화장대에 [{skin_type}] 피부용 [{std_type}] 제품이 비어 있어, DB 중 적합도 상위 {len(db_recs)}개 제품을 추천했습니다.",
                f"[{skin_type} · {concern}] 맞춤 유효 성분({rec_ings})이 포함되어 있습니다.",
                env_reason
            ]
            xai_evidence = [
                f"피부 유형([{skin_type}]) 제형 선호도 및 고민([{concern}]) 기능적 연관성 평가",
                f"보유 화장대 주의 성분({', '.join(vanity_caution_ings) if vanity_caution_ings else '없음'})과의 상극 회피 검증"
            ]
            xai_extra = "새로 제품을 구비하실 때 위 포함 권장(O) 성분이 들어있고 배제 권장(X) 성분이 없는지 확인하세요."

        detailed_steps.append({
            "step_number": idx + 1,
            "step_title": step_title,
            "target_item_type": std_type,
            "frequency_badge": guide["frequency_badge"],
            "fit_score": fit_score,
            "status": "OWNED" if owned_match else "RECOMMENDED",
            "owned_product": owned_match,
            "recommended_products": db_recs,
            "good_ingredients": guide["good_ingredients"],
            "avoid_ingredients": guide["avoid_ingredients"],
            "criteria_tip": guide["criteria_tip"],
            "xai_explanation": {
                "reasons": xai_reasons,
                "evidence": xai_evidence,
                "extra_guide": xai_extra
            }
        })
    return detailed_steps, used_vanity_keys


def build_supplement_dosing_recommendations(
    conn: sqlite3.Connection,
    ranked_nutrient_candidates: List[Dict[str, Any]],
    user_registered_nutrients: set,
    supplement_limit: int = 3
) -> List[Dict[str, Any]]:
    limit_cnt = max(1, min(4, int(supplement_limit or 3)))

    novel_candidates = [c for c in ranked_nutrient_candidates if c["nutrient"] not in user_registered_nutrients]
    already_taking_candidates = [c for c in ranked_nutrient_candidates if c["nutrient"] in user_registered_nutrients]

    selected_list: List[Dict[str, Any]] = []
    seen_nuts = set()

    for cand in novel_candidates:
        if cand["nutrient"] not in seen_nuts:
            seen_nuts.add(cand["nutrient"])
            selected_list.append(cand)
        if len(selected_list) >= limit_cnt:
            break

    if len(selected_list) < limit_cnt:
        for cand in already_taking_candidates:
            if cand["nutrient"] not in seen_nuts:
                seen_nuts.add(cand["nutrient"])
                selected_list.append(cand)
            if len(selected_list) >= limit_cnt:
                break

    cur = conn.cursor()
    detailed_supps = []

    for rank_idx, cand in enumerate(selected_list, start=1):
        nut = cand["nutrient"]
        meta = SUPPLEMENT_DOSING_GUIDE.get(nut, {
            "when_slot": "☀️ 아침 또는 점심 식사 직후",
            "when_category": "MORNING",
            "how_to_take": "위장 자극을 줄이고 흡수율을 높이기 위해 식사 직후 충분한 물(200ml)과 함께 섭취하세요.",
            "why_recommended": "피부 컨디션 회복과 체내 영양 밸런스 유지를 돕습니다.",
            "caution_tip": "제품 라벨의 일일 권장 섭취량을 확인하여 복용하세요.",
            "search_keyword": nut
        })

        kw = meta.get("search_keyword", nut)
        cur.execute("""
            SELECT product_id, company, product_name, primary_function, caution, raw_ingredients_text
            FROM products
            WHERE category = 'Supplement' AND (product_name LIKE ? OR raw_ingredients_text LIKE ?)
            ORDER BY product_id ASC LIMIT 2
        """, (f"%{kw}%", f"%{kw}%"))
        rows = cur.fetchall()

        db_supps = []
        for r in rows:
            db_supps.append({
                "product_id": r["product_id"],
                "brand": r["company"],
                "name": r["product_name"],
                "primary_function": r["primary_function"] or "건강기능식품 기능성 원료",
                "caution": r["caution"] or "특이체질 및 알레르기 체질은 성분을 확인 후 섭취하세요."
            })

        detailed_supps.append({
            "priority_rank": rank_idx,
            "priority_score": cand["score"],
            "priority_tag": cand["priority_tag"],
            "nutrient": nut,
            "is_already_taking": nut in user_registered_nutrients,
            "skin_type_rationale": cand["skin_type_rationale"],
            "custom_match_reason": cand["custom_reason"],
            "when_slot": meta["when_slot"],
            "when_category": meta["when_category"],
            "how_to_take": meta["how_to_take"],
            "why_recommended": meta["why_recommended"],
            "caution_tip": meta["caution_tip"],
            "recommended_products": db_supps
        })

    return detailed_supps


def analyze_user_registered_supplements(
    user_supplements: List[Dict[str, Any]],
    vanity_ings: set,
    skin_type: str,
    concern: str
) -> Tuple[List[Dict[str, Any]], List[str], List[str], set]:
    analyzed_cards = []
    ing_counter = Counter()
    all_user_nutrients = set()

    for s in (user_supplements or []):
        s_name = s.get("name", "등록 영양제")
        s_ings = s.get("ingredients", [])
        s_raw = s.get("raw_ingredients_text", "")
        detected_nuts = extract_nutrients_from_user_supplement(s_name, s_ings, s_raw)

        for dn in detected_nuts:
            ing_counter[dn] += 1
            all_user_nutrients.add(dn)

        primary_nut = detected_nuts[0]
        meta = SUPPLEMENT_DOSING_GUIDE.get(primary_nut, {
            "when_slot": "🌤️ 아침 또는 점심 식사 직후",
            "when_category": "LUNCH_DINNER",
            "how_to_take": "위장 부담을 줄이고 흡수율을 높이기 위해 식사 직후 물 1컵과 함께 섭취하세요.",
            "why_recommended": f"[{s_name}]에 포함된 기능성 원료가 체내 대사 및 피부 건강 유지를 돕습니다.",
            "caution_tip": "일일 권장 섭취량을 지켜 복용하세요."
        })

        if "칼슘" in detected_nuts and "비타민D" in detected_nuts:
            when_slot = "🌙 저녁 식사 직후 (칼슘 흡수 및 숙면 시너지)"
            when_cat = "NIGHT"
            how_txt = "칼슘은 위산이 분비되는 저녁 식사 직후에 섭취할 때 흡수율이 가장 높으며, 함께 들어있는 비타민D가 장내 칼슘 흡수와 피부 장벽 면역을 동시에 도와줍니다."
        else:
            when_slot = meta["when_slot"]
            when_cat = meta["when_category"]
            how_txt = meta["how_to_take"]

        compat_status = "양호 (병용 가능)"
        compat_badge_code = "GOOD"
        compat_note = f"[{skin_type} · {concern}] 관리 루틴과 안전하게 병행 섭취할 수 있습니다."

        if "비타민A" in detected_nuts and "레티놀" in vanity_ings:
            compat_status = "높은 주의 (레티놀 중복)"
            compat_badge_code = "HIGH_CAUTION"
            compat_note = "🚨 내 화장대에 등록된 '레티놀' 화장품과 비타민A 성분이 겹칩니다. 과잉 자극 및 피부 건조·박리 위험이 있으므로 섭취 주기를 조절하세요."
        elif "비타민C" in detected_nuts and "비타민C" in vanity_ings:
            compat_status = "우수 (내·외용 항산화 시너지)"
            compat_badge_code = "GOOD"
            compat_note = "✨ 화장대의 비타민C 외용제와 경구 비타민C가 만나 자외선 방어 및 미백·콜라겐 합성 시너지가 극대화됩니다."

        analyzed_cards.append({
            "id": s.get("id", ""),
            "name": s_name,
            "detected_nutrients": detected_nuts,
            "when_slot": when_slot,
            "when_category": when_cat,
            "how_to_take": how_txt,
            "why_recommended": meta["why_recommended"],
            "caution_tip": meta["caution_tip"],
            "compat_status": compat_status,
            "compat_badge_code": compat_badge_code,
            "compat_note": compat_note
        })

    duplicate_warnings = [
        f"현재 등록하신 복용 영양제 중 '{nut}' 성분이 {cnt}개 제품에 중복 포함되어 있습니다. 일일 상한 섭취량 초과 여부를 확인하세요."
        for nut, cnt in ing_counter.items() if cnt >= 2
    ]

    cross_notes = []
    if "레티놀" in vanity_ings and "비타민A" in all_user_nutrients:
        cross_notes.append("🚨 [높은 주의] 내 화장대의 '레티놀' 화장품과 복용 중인 '비타민A' 영양제가 중복됩니다. 지용성 비타민A 과잉 및 피부 박리 자극 우려가 있습니다.")
    if "비타민C" in vanity_ings and "비타민C" in all_user_nutrients:
        cross_notes.append("✨ [항산화 시너지] 외용 비타민C 화장품과 경구 비타민C 영양제를 병행 중입니다. 낮 자외선 방어 및 미백 시너지가 매우 우수합니다.")

    return analyzed_cards, duplicate_warnings, cross_notes, all_user_nutrients


# ==============================================================================
# [5] 다중 제품 교차 상극 분석 API
# ==============================================================================
class ProductAnalyzeItem(BaseModel):
    id: str = Field(..., description="제품 식별자")
    name: str = Field(..., description="제품명")
    brand: Optional[str] = Field("", description="브랜드명")
    item_type: Optional[str] = Field("스킨케어", description="제형 분류")
    category: Optional[str] = Field("cosmetic", description="cosmetic 또는 Supplement")
    ingredients: List[str] = Field(default_factory=list, description="주요/추출 성분 리스트")
    raw_ingredients_text: Optional[str] = Field("", description="전성분 원문 텍스트")

class MultiAnalyzeRequest(BaseModel):
    products: List[ProductAnalyzeItem] = Field(default_factory=list)
    supplements: List[ProductAnalyzeItem] = Field(default_factory=list)
    uv_index: Optional[float] = Field(None)
    skin_sensitivity: Optional[str] = Field("보통")

def evaluate_pair_conflict(prod_a: Dict[str, Any], prod_b: Dict[str, Any], uv_index: Optional[float] = None) -> Dict[str, Any]:
    ings_a_raw = prod_a.get("ingredients", [])
    ings_b_raw = prod_b.get("ingredients", [])
    raw_text_a = prod_a.get("raw_ingredients_text", "")
    raw_text_b = prod_b.get("raw_ingredients_text", "")

    ings_a = list(dict.fromkeys([normalize_ing_name(i) for i in ings_a_raw if i]))
    ings_b = list(dict.fromkeys([normalize_ing_name(i) for i in ings_b_raw if i]))

    if (len(ings_a) == 0 and len(str(raw_text_a).strip()) < 5) or (len(ings_b) == 0 and len(str(raw_text_b).strip()) < 5):
        return {
            "pair_id": f"{prod_a['id']}_{prod_b['id']}",
            "product_a_id": prod_a["id"],
            "product_a_name": prod_a["name"],
            "product_b_id": prod_b["id"],
            "product_b_name": prod_b["name"],
            "risk_level": "확인 필요",
            "risk_display": "확인 필요 (분석 불충분)",
            "risk_code": "INSUFFICIENT",
            "short_desc": "성분 정보나 농도 데이터가 부족해 상호작용을 판단하기 어려움",
            "causative_ingredients": [],
            "duplicate_ingredients": [],
            "rationale": "선택한 제품 중 전성분 목록이나 주요 유효 성분 데이터가 충분히 확인되지 않아 임의로 위험 등급을 부여하지 않았습니다.",
            "recommended_usage": "제품 용기 뒷면의 전성분 라벨을 스캔하거나 전성분 정보를 추가 등록한 후 재분석해 주세요.",
            "am_pm_split": "전성분 확인 후 결정",
            "conc_freq_note": "성분 함량 및 농도 정보 미확인",
            "sensitivity_note": "민감성 피부는 팔 안쪽에 패치 테스트 후 사용을 권장합니다."
        }

    set_a = set(ings_a)
    set_b = set(ings_b)
    combined_set = set_a.union(set_b)
    duplicates = sorted(list(set_a.intersection(set_b)))

    matched_rules = []
    causative_ings = set()

    for rule in PAIR_INTERACTION_RULES:
        r_ings = rule["ings"]
        if r_ings.issubset(combined_set):
            ing_list = list(r_ings)
            if (ing_list[0] in set_a and ing_list[1] in set_b) or (ing_list[1] in set_a and ing_list[0] in set_b):
                matched_rules.append(rule)
                causative_ings.update(r_ings)

    high_potency_dups = [d for d in duplicates if d in ["레티놀", "AHA", "BHA", "비타민C", "비타민A"]]
    if "레티놀" in high_potency_dups or "비타민A" in high_potency_dups:
        causative_ings.add("레티놀(중복)")
        matched_rules.append({
            "risk_level": "높은 주의",
            "risk_code": "HIGH_CAUTION",
            "short_desc": "두 제품 모두 레티놀/비타민A 계열이 중복 포함되어 과잉 자극 우려",
            "rationale": f"'{prod_a['name']}'와(과) '{prod_b['name']}' 두 제품 모두에 레티놀(비타민A) 계열 성분이 중복 포함되어 있어 함께 바르면 유효 농도가 과도해질 수 있습니다.",
            "recommended_usage": "한 루틴에서 두 제품을 동시에 사용하지 마시고, 한 가지 제품만 선택하여 저녁에 단독 사용하세요.",
            "am_pm_split": "중복 사용 지양 (택 1 사용 권장)",
            "conc_freq_note": "동일 활성 성분 중복 도포 시 부작용 확률이 상승합니다.",
            "sensitivity_note": "피부 각질 들뜸이나 따가움이 발생할 수 있으니 1종만 사용하세요."
        })
    elif any(d in high_potency_dups for d in ["AHA", "BHA", "비타민C"]):
        dup_names = ", ".join([d for d in high_potency_dups if d in ["AHA", "BHA", "비타민C"]])
        causative_ings.update([d for d in high_potency_dups if d in ["AHA", "BHA", "비타민C"]])
        matched_rules.append({
            "risk_level": "주의",
            "risk_code": "CAUTION",
            "short_desc": f"동일 활성 성분({dup_names}) 중복 포함으로 사용 빈도 조절 필요",
            "rationale": f"두 제품에 '{dup_names}' 성분이 중복 포함되어 있어 한 번에 레이어링하면 산도(pH) 저하 및 자극 부담이 커질 수 있습니다.",
            "recommended_usage": "아침과 저녁으로 나누어 사용하거나, 하루에 한 제품만 번갈아 사용하는 것을 권장합니다.",
            "am_pm_split": "아침/저녁 분리 또는 격일 교차 사용 권장",
            "conc_freq_note": f"{dup_names} 중복 사용 시 일일 적정 노출량을 초과할 수 있습니다.",
            "sensitivity_note": "민감성 피부는 두 제품 중 자극이 적은 제형 하나만 루틴에 포함하세요."
        })

    if matched_rules:
        high_rules = [r for r in matched_rules if r["risk_code"] == "HIGH_CAUTION"]
        primary_rule = high_rules[0] if high_rules else matched_rules[0]
        return {
            "pair_id": f"{prod_a['id']}_{prod_b['id']}",
            "product_a_id": prod_a["id"],
            "product_a_name": prod_a["name"],
            "product_b_id": prod_b["id"],
            "product_b_name": prod_b["name"],
            "risk_level": primary_rule["risk_level"],
            "risk_display": primary_rule["risk_level"],
            "risk_code": primary_rule["risk_code"],
            "short_desc": primary_rule["short_desc"],
            "causative_ingredients": sorted(list(causative_ings)),
            "duplicate_ingredients": duplicates,
            "rationale": primary_rule["rationale"],
            "recommended_usage": primary_rule["recommended_usage"],
            "am_pm_split": primary_rule["am_pm_split"],
            "conc_freq_note": primary_rule["conc_freq_note"],
            "sensitivity_note": primary_rule["sensitivity_note"]
        }

    synergy_msg = "일반적인 사용 조건에서 두 제품의 주요 성분 간 상극이나 화학적 충돌 근거가 확인되지 않았습니다."
    for syn_set, msg in SYNERGY_RULES:
        if syn_set.issubset(combined_set):
            synergy_msg = msg
            break

    dup_note = f" (공통 보습·진정 성분 '{', '.join(duplicates)}' 중복 포함 — 병용 무방)" if duplicates else ""
    return {
        "pair_id": f"{prod_a['id']}_{prod_b['id']}",
        "product_a_id": prod_a["id"],
        "product_a_name": prod_a["name"],
        "product_b_id": prod_b["id"],
        "product_b_name": prod_b["name"],
        "risk_level": "양호",
        "risk_display": "양호",
        "risk_code": "GOOD",
        "short_desc": "일반적으로 함께 활용 가능" + (" (보습·진정 시너지)" if "보완" in synergy_msg or "진정" in synergy_msg else ""),
        "causative_ingredients": [],
        "duplicate_ingredients": duplicates,
        "rationale": synergy_msg + dup_note,
        "recommended_usage": "제형이 묽은 제품(토너/세럼)부터 되직한 제품(크림/선케어) 순서로 흡수시켜 함께 사용하시면 됩니다.",
        "am_pm_split": "시간대 분리 불필요 (함께 사용 가능)",
        "conc_freq_note": "일반 권장 사용량 기준 데일리 병용 가능",
        "sensitivity_note": "특이 알레르기 유발 성분이 없다면 민감성 피부도 무난히 병용할 수 있습니다."
    }


@app.post("/api/v1/analyze/multi", summary="다중 화장품 & 영양제 N:N 교차 상극 및 4단계 위험도 분석")
def analyze_multi_products(req: MultiAnalyzeRequest):
    all_items: List[Dict[str, Any]] = []

    for idx, p in enumerate(req.products):
        extracted = list(p.ingredients)
        if p.raw_ingredients_text:
            extracted.extend(extract_standard_ingredients(p.raw_ingredients_text))
        norm_ings = list(dict.fromkeys([normalize_ing_name(i) for i in extracted if i]))
        all_items.append({
            "id": str(p.id or f"P{idx+1}"),
            "label": chr(65 + idx) if idx < 26 else f"P{idx+1}",
            "name": p.name,
            "brand": p.brand or "",
            "item_type": p.item_type or "스킨케어",
            "category": p.category or "cosmetic",
            "ingredients": norm_ings,
            "raw_ingredients_text": p.raw_ingredients_text or ", ".join(norm_ings)
        })

    for s_idx, s in enumerate(req.supplements):
        norm_ings = list(dict.fromkeys([normalize_ing_name(i) for i in extract_nutrients_from_user_supplement(s.name, s.ingredients, s.raw_ingredients_text or "")]))
        all_items.append({
            "id": str(s.id or f"SUPP_{s_idx+1}"),
            "label": f"영양제{s_idx+1}",
            "name": s.name,
            "brand": s.brand or "건강기능식품",
            "item_type": "영양제(경구)",
            "category": "Supplement",
            "ingredients": norm_ings,
            "raw_ingredients_text": s.raw_ingredients_text or ", ".join(norm_ings)
        })

    if len(all_items) < 2:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error_code": "INSUFFICIENT_PRODUCTS",
                "user_message": "교차 상극 분석을 위해 최소 2개 이상의 제품을 선택해 주세요."
            }
        )

    ing_occurrence: Dict[str, List[str]] = {}
    for item in all_items:
        for ing in item["ingredients"]:
            ing_occurrence.setdefault(ing, []).append(item["name"])

    duplicate_ingredients_summary = [
        {
            "ingredient": ing,
            "count": len(prod_names),
            "products": prod_names,
            "is_caution_duplicate": ing in CAUTION_INGREDIENT_META
        }
        for ing, prod_names in ing_occurrence.items() if len(prod_names) >= 2
    ]
    duplicate_ing_set = {d["ingredient"] for d in duplicate_ingredients_summary}

    product_summaries = []
    for item in all_items:
        ings = item["ingredients"]
        caution_ings = [i for i in ings if i in CAUTION_INGREDIENT_META]
        dup_ings = [i for i in ings if i in duplicate_ing_set]
        major_ings = [i for i in ings if i not in caution_ings] or ings

        time_info = classify_product_time_slot(item["item_type"], ings, item["name"], item["category"]) \
            if item["category"] != "Supplement" else {
                "slot_code": "BOTH",
                "slot_badge": "💊 경구 섭취 (식후 권장)",
                "reason": "건강기능식품으로 권장 섭취 시간에 맞춰 복용하세요."
            }

        product_summaries.append({
            "id": item["id"],
            "label": item["label"],
            "name": item["name"],
            "brand": item["brand"],
            "item_type": item["item_type"],
            "category": item["category"],
            "major_ingredients": major_ings,
            "caution_ingredients": caution_ings,
            "duplicate_ingredients": dup_ings,
            "time_slot": time_info
        })

    pairwise_results = []
    for item_a, item_b in combinations(all_items, 2):
        res = evaluate_pair_conflict(item_a, item_b, uv_index=req.uv_index)
        res["pair_label"] = f"{item_a['label']} + {item_b['label']}"
        pairwise_results.append(res)

    risk_counts = Counter([p["risk_code"] for p in pairwise_results])
    if risk_counts["HIGH_CAUTION"] > 0:
        overall_risk, overall_code = "높은 주의", "HIGH_CAUTION"
    elif risk_counts["CAUTION"] > 0:
        overall_risk, overall_code = "주의", "CAUTION"
    elif risk_counts["INSUFFICIENT"] > 0 and risk_counts["GOOD"] == 0:
        overall_risk, overall_code = "확인 필요", "INSUFFICIENT"
    else:
        overall_risk, overall_code = "양호", "GOOD"

    penalty = (risk_counts["HIGH_CAUTION"] * 25) + (risk_counts["CAUTION"] * 12) + (risk_counts["INSUFFICIENT"] * 5)
    safety_score = max(35, 98 - penalty)

    am_usable = [p["name"] for p in product_summaries if p["category"] != "Supplement" and p["time_slot"]["slot_code"] in ("BOTH", "AM_ONLY", "AM_PREF")]
    pm_usable = [p["name"] for p in product_summaries if p["category"] != "Supplement" and p["time_slot"]["slot_code"] in ("BOTH", "PM_ONLY", "PM_PREF")]
    pm_only_list = [p["name"] for p in product_summaries if p["category"] != "Supplement" and p["time_slot"]["slot_code"] in ("PM_ONLY", "PM_PREF")]

    return {
        "status": "success",
        "overall_risk": overall_risk,
        "overall_risk_code": overall_code,
        "safety_score": safety_score,
        "total_products_count": len(all_items),
        "total_pairs_count": len(pairwise_results),
        "risk_summary_counts": {
            "high_caution": risk_counts["HIGH_CAUTION"],
            "caution": risk_counts["CAUTION"],
            "good": risk_counts["GOOD"],
            "insufficient": risk_counts["INSUFFICIENT"]
        },
        "duplicate_ingredients_summary": duplicate_ingredients_summary,
        "product_summaries": product_summaries,
        "pairwise_results": pairwise_results,
        "time_division_guide": {
            "am_available_products": am_usable,
            "pm_available_products": pm_usable,
            "pm_recommended_only": pm_only_list
        }
    }


# ==============================================================================
# [6] 통합 AI 루틴 & 이너뷰티 추천 API
# ==============================================================================
class LifestyleInput(BaseModel):
    makeup: Optional[int] = Field(1, ge=0, le=2)
    sleep: Optional[int] = Field(1, ge=0, le=2)
    outdoor: Optional[int] = Field(1, ge=0, le=2)
    alcohol: Optional[int] = Field(0, ge=0, le=2)
    workout: Optional[int] = Field(1, ge=0, le=2)

class VanityCustomItem(BaseModel):
    product_id: Optional[int] = 0
    brand: Optional[str] = ""
    name: str
    category: Optional[str] = ""
    item_type: Optional[str] = "스킨케어"
    active_ingredients: List[str] = Field(default_factory=list)
    raw_ingredients_text: Optional[str] = ""

class EnvContextInput(BaseModel):
    region_name: Optional[str] = None
    data_source: Optional[str] = None
    is_live: Optional[bool] = False
    temp: Optional[float] = None
    humidity: Optional[int] = None
    uv_index: Optional[float] = None
    uv_level: Optional[str] = None
    uv_max_today: Optional[float] = None
    uv_max_level: Optional[str] = None
    pm25: Optional[float] = None

class RoutineRecommendRequest(BaseModel):
    skin_type: str = Field("수부지", description="6대 피부 유형: 건성, 지성, 수부지, 복합성, 민감성, 여드름성")
    concern: str = Field("수분/속건조", description="7대 피부 고민")
    care_style_id: int = Field(3, ge=1, le=5, description="1단계 미니멀 ~ 5단계 풀케어")
    supplement_limit: Optional[int] = Field(3, ge=1, le=4, description="희망 이너뷰티 영양제 추천 개수 (1~4개)")
    lifestyle: Optional[LifestyleInput] = Field(default_factory=LifestyleInput)
    env_context: Optional[EnvContextInput] = Field(default_factory=EnvContextInput)
    cosmetic_product_ids: List[int] = Field(default_factory=list)
    vanity_items: List[VanityCustomItem] = Field(default_factory=list)
    user_supplements: List[Dict[str, Any]] = Field(default_factory=list, description="사용자가 등록한 복용 영양제 객체 목록")
    supplement_product_ids: List[int] = Field(default_factory=list)
    extra_ingredients: List[str] = Field(default_factory=list)

@app.post("/api/v1/routine/recommend", summary="통합 AI 루틴 & 이너뷰티 추천 (복수 제품 추천 + 영양제 1~4개 우선순위)")
def recommend_routine(req: RoutineRecommendRequest):
    start_ts = time.time()
    try:
        target_skin_type = req.skin_type if req.skin_type in SKIN_TYPE_PROFILES else "수부지"
        target_concern = req.concern if req.concern in SKIN_CONCERNS else "수분/속건조"
        style_id = max(1, min(5, int(req.care_style_id or 3)))
        supp_limit = max(1, min(4, int(req.supplement_limit or 3)))

        st_profile = SKIN_TYPE_PROFILES[target_skin_type]
        concern_criteria = CONCERN_CRITERIA_MAP.get(target_concern, CONCERN_CRITERIA_MAP["수분/속건조"])

        lifestyle = req.lifestyle or LifestyleInput()
        makeup_val = lifestyle.makeup if lifestyle.makeup is not None else 1
        sleep_val = lifestyle.sleep if lifestyle.sleep is not None else 1
        outdoor_val = lifestyle.outdoor if lifestyle.outdoor is not None else 1
        alcohol_val = lifestyle.alcohol if lifestyle.alcohol is not None else 0
        workout_val = lifestyle.workout if lifestyle.workout is not None else 1

        env_ctx = req.env_context or EnvContextInput()
        has_env_data = (
            env_ctx.data_source in ("LIVE", "CACHED")
            and (env_ctx.uv_index is not None or env_ctx.temp is not None)
        )
        env_uv = float(env_ctx.uv_index) if (has_env_data and env_ctx.uv_index is not None) else None
        env_uv_max = float(env_ctx.uv_max_today) if (has_env_data and env_ctx.uv_max_today is not None) else None
        env_hum = int(env_ctx.humidity) if (has_env_data and env_ctx.humidity is not None) else None
        env_temp = float(env_ctx.temp) if (has_env_data and env_ctx.temp is not None) else None
        env_pm25 = float(env_ctx.pm25) if (has_env_data and env_ctx.pm25 is not None) else None

        cur_uv_meta = classify_uv_level(env_uv)
        max_uv_meta = classify_uv_level(env_uv_max)

        if has_env_data:
            parts = []
            if env_ctx.region_name:
                parts.append(env_ctx.region_name)
            if env_temp is not None:
                parts.append(f"{env_temp}°C")
            if env_hum is not None:
                parts.append(f"습도 {env_hum}%")
            if env_uv is not None:
                parts.append(f"실시간 UV {env_uv}({cur_uv_meta['level']})")
            if env_uv_max is not None:
                parts.append(f"최고 UV {env_uv_max}({max_uv_meta['level']})")
            env_summary_text = " · ".join(parts)
            _, env_rationale_list = build_env_skincare_tips_and_rationale(
                temp=env_temp,
                humidity=env_hum,
                uv_current=env_uv,
                uv_max_today=env_uv_max,
                uv_max_tomorrow=None,
                pm25=env_pm25,
                pm10=None,
                is_day=None
            )
        else:
            env_summary_text = ""
            env_rationale_list = []

        conn = get_db()
        cur = conn.cursor()

        selected_cosmetics: List[Dict[str, Any]] = []
        cosmetic_ings = [normalize_ing_name(i) for i in (req.extra_ingredients or []) if i]
        basket_item_types = []
        owned_product_ids = set()

        for v in (req.vanity_items or []):
            if is_dummy_product_item({"product_id": v.product_id, "name": v.name}):
                continue
            norm_ext = [normalize_ing_name(x) for x in (v.active_ingredients or [])]
            if v.raw_ingredients_text:
                norm_ext.extend([normalize_ing_name(x) for x in extract_standard_ingredients(v.raw_ingredients_text)])
            norm_ext = list(dict.fromkeys(norm_ext))
            selected_cosmetics.append({
                "product_id": v.product_id or 0,
                "brand": v.brand or "",
                "name": v.name,
                "category": v.category or "",
                "item_type": v.item_type or "스킨케어",
                "active_ingredients": norm_ext
            })
            if v.product_id:
                owned_product_ids.add(v.product_id)
            if v.item_type:
                basket_item_types.append(v.item_type)
            cosmetic_ings.extend(norm_ext)

        db_ids = [pid for pid in (req.cosmetic_product_ids or []) if pid not in owned_product_ids and pid not in DUMMY_TEST_IDS]
        if db_ids:
            placeholders = ",".join(["?"] * len(db_ids))
            cur.execute(f"""
                SELECT product_id, company, product_name, sub_category, item_type, raw_ingredients_text
                FROM products WHERE product_id IN ({placeholders})
            """, db_ids)
            for row in cur.fetchall():
                raw_txt = row["raw_ingredients_text"] or ""
                norm_ext = list(dict.fromkeys([normalize_ing_name(x) for x in extract_standard_ingredients(raw_txt)]))
                selected_cosmetics.append({
                    "product_id": row["product_id"],
                    "brand": row["company"],
                    "name": row["product_name"],
                    "category": row["sub_category"],
                    "item_type": row["item_type"],
                    "active_ingredients": norm_ext
                })
                owned_product_ids.add(row["product_id"])
                if row["item_type"]:
                    basket_item_types.append(row["item_type"])
                cosmetic_ings.extend(norm_ext)

        owned_macro_roles = set()
        owned_role_counts = Counter()
        for sc in selected_cosmetics:
            m_role, _ = classify_cosmetic_role(sc.get("item_type", ""), sc.get("name", ""), sc.get("category", ""))
            owned_macro_roles.add(m_role)
            owned_role_counts[m_role] += 1

        mapped_ings = list(dict.fromkeys(cosmetic_ings))
        vanity_caution_ings = [i for i in mapped_ings if i in CAUTION_INGREDIENT_META]

        conflicts = []
        if "레티놀" in mapped_ings and "비타민C" in mapped_ings:
            conflicts.append({"level": "주의", "message": "레티놀과 비타민C 동시 도포 시 pH 차이로 인한 자극 가능성이 있어 아침(비타민C)·저녁(레티놀) 분리를 권장합니다."})
        if "레티놀" in mapped_ings:
            effective_high_uv = (env_uv is not None and env_uv >= 6.0) or (env_uv_max is not None and env_uv_max >= 6.0) or (outdoor_val == 2)
            if effective_high_uv:
                uv_note = (
                    f"실시간 UV {env_uv}({cur_uv_meta['level']})"
                    if env_uv is not None
                    else (f"오늘 최고 UV {env_uv_max}({max_uv_meta['level']})" if env_uv_max is not None else "야외활동 지수 높음")
                )
                conflicts.append({"level": "높은 주의", "message": f"자외선 주의({uv_note}): 레티놀 제품은 낮 시간대를 피하고 저녁 루틴에만 단독 사용하세요."})
        if "레티놀" in mapped_ings and ("BHA" in mapped_ings or "AHA" in mapped_ings):
            conflicts.append({"level": "높은 주의", "message": "레티놀과 각질제거 성분(AHA/BHA) 중복 시 장벽 손상 우려가 있으므로 격일 교차 사용하세요."})
        if target_skin_type == "민감성" and any(k in mapped_ings for k in ["레티놀", "AHA"]):
            conflicts.append({"level": "주의", "message": "선택하신 [민감성] 피부 유형에서는 보유 중인 레티놀/AHA 제품이 자극을 유발할 수 있어 데일리 루틴에서 제외했습니다."})

        safety_score = max(50, 98 - len(conflicts) * 16)

        predicted_arch_id = find_archetype_id(target_concern, style_id)
        match_rate = 94.2

        if ai_package and "model" in ai_package:
            try:
                model = ai_package["model"]
                c_to_idx = ai_package.get("concern_to_idx", {})
                t_to_idx = ai_package.get("item_type_to_idx", {})
                i_to_idx = ai_package.get("ing_to_idx", {})

                c_vec = np.zeros(len(SKIN_CONCERNS), dtype=np.float32)
                if target_concern in c_to_idx:
                    c_vec[c_to_idx[target_concern]] = 1.0

                life_vec = np.array([style_id, makeup_val, sleep_val, outdoor_val, alcohol_val, workout_val], dtype=np.float32)

                type_vec = np.zeros(len(t_to_idx), dtype=np.float32)
                for itype in basket_item_types:
                    if itype in t_to_idx:
                        type_vec[t_to_idx[itype]] += 1.0

                ing_vec = np.zeros(len(i_to_idx), dtype=np.float32)
                for ing in cosmetic_ings:
                    if ing in i_to_idx:
                        ing_vec[i_to_idx[ing]] += 1.0

                input_vector = np.concatenate([c_vec, life_vec, type_vec, ing_vec]).reshape(1, -1)
                expected_features = getattr(model, "n_features_in_", input_vector.shape[1])
                if input_vector.shape[1] == expected_features:
                    predicted_arch_id = int(model.predict(input_vector)[0])
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(input_vector)[0]
                        match_rate = round(float(np.max(probs)) * 100, 1)
            except Exception as ml_err:
                print(f"[WARN] ML Inference fallback used: {ml_err}")

        archetype_data = get_archetype_by_id(predicted_arch_id)
        required_types = ensure_str_list(archetype_data.get("required_types", []))
        morning_steps = ensure_str_list(archetype_data.get("morning_steps", []))
        night_steps = ensure_str_list(archetype_data.get("night_steps", []))
        weekly_special = ensure_str_list(archetype_data.get("weekly_special", []))

        if target_skin_type in ["지성", "여드름성", "수부지"]:
            morning_steps = [s.replace("보습 크림", "산뜻 수딩 젤/크림").replace("영양 크림", "수분 수딩크림") for s in morning_steps]
        elif target_skin_type == "건성":
            night_steps = [s.replace("수분크림", "고보습 세라마이드 장벽 크림") for s in night_steps]

        missing_steps = []
        missing_step_recommendations = []
        for req_type in required_types:
            req_macro, _ = classify_cosmetic_role(req_type, req_type)
            if req_macro not in owned_macro_roles:
                missing_steps.append(req_type)
                guide = get_step_ingredient_guide(req_type, target_skin_type, target_concern, "BOTH", vanity_caution_ings)
                recs = find_recommended_cosmetics_from_db(
                    conn, req_type, target_skin_type, target_concern, guide["good_ingredients"], guide["avoid_ingredients"], owned_product_ids, limit=4
                )
                missing_step_recommendations.append({
                    "missing_step": req_type,
                    "good_ingredients": guide["good_ingredients"],
                    "avoid_ingredients": guide["avoid_ingredients"],
                    "criteria_tip": guide["criteria_tip"],
                    "recommended_products": recs
                })

        overused_steps = []
        role_kor_names = {"SERUM": "에센스/세럼/앰플", "CREAM": "크림/보습제", "TONER": "스킨/토너/패드", "SUNCARE": "선케어"}
        for r_key, cnt in owned_role_counts.items():
            if cnt >= 2 and r_key in role_kor_names:
                overused_steps.append(f"{role_kor_names[r_key]} {cnt}종 중복 (아침/저녁 분리 또는 택 1 사용 권장)")

        morning_detailed, am_used_keys = build_detailed_routine_steps(
            conn, morning_steps, target_skin_type, target_concern, "AM", selected_cosmetics, vanity_caution_ings, owned_product_ids, env_summary_text
        )
        night_detailed, pm_used_keys = build_detailed_routine_steps(
            conn, night_steps, target_skin_type, target_concern, "PM", selected_cosmetics, vanity_caution_ings, owned_product_ids, env_summary_text
        )
        all_used_keys = am_used_keys.union(pm_used_keys)

        excluded_vanity_products = []
        for sc in selected_cosmetics:
            v_key = f"{sc.get('product_id', 0)}_{sc.get('name', '')}"
            v_ings = sc.get("active_ingredients", [])
            m_role, _ = classify_cosmetic_role(sc.get("item_type", ""), sc.get("name", ""), sc.get("category", ""))

            if v_key not in all_used_keys:
                if target_skin_type == "민감성" and any(k in v_ings for k in ["레티놀", "AHA"]):
                    ex_reason = "선택하신 [민감성] 피부 유형에서는 고농도 레티놀/AHA 성분이 피부 장벽 자극 및 홍조를 유발할 수 있어 데일리 루틴에서 제외했습니다."
                elif any(k in v_ings for k in ["레티놀", "AHA", "BHA"]) and len(conflicts) > 0:
                    ex_reason = "보유 화장대 내 다른 각질·활성 성분과의 상극(높은 주의/주의) 및 자극 중첩을 피하기 위해 기본 데일리 루틴에서 제외했습니다."
                elif owned_role_counts[m_role] >= 2:
                    ex_reason = f"동일 역할군({role_kor_names.get(m_role, m_role)}) 내 [{target_skin_type}] 피부에 더 적합한 다른 보유 제품이 우선 배치되어 제외했습니다."
                else:
                    ex_reason = f"선택하신 '{style_id}단계 케어 스타일' 및 [{target_skin_type} · {target_concern}] 타겟 구성에서 우선순위가 낮아 제외했습니다."

                excluded_vanity_products.append({
                    "brand": sc.get("brand", ""),
                    "name": sc.get("name", ""),
                    "item_type": sc.get("item_type", ""),
                    "active_ingredients": v_ings,
                    "excluded_scope": "전체 데일리 루틴 제외 (스페셜/교차 케어 전환)",
                    "reason": ex_reason
                })
            elif any(k in v_ings for k in ["레티놀", "AHA"]):
                uv_reason_prefix = (
                    f"낮 시간대 자외선(최고 UV {env_uv_max} · {max_uv_meta['level']}) 및 "
                    if env_uv_max is not None
                    else (f"자외선(실시간 UV {env_uv} · {cur_uv_meta['level']}) 및 " if env_uv is not None else "낮 시간대 자외선 ")
                )
                excluded_vanity_products.append({
                    "brand": sc.get("brand", ""),
                    "name": sc.get("name", ""),
                    "item_type": sc.get("item_type", ""),
                    "active_ingredients": v_ings,
                    "excluded_scope": "☀️ 아침(AM) 루틴에서 제외 (🌙 저녁 전용 배치)",
                    "reason": f"{uv_reason_prefix}노출 시 광독성 및 자극 위험이 있는 성분(레티놀/AHA)이 포함되어 아침 루틴에서는 제외하고 저녁 루틴에만 배치했습니다."
                })

        user_supp_cards, supp_dup_warnings, supp_cross_notes, user_taking_nutrients = analyze_user_registered_supplements(
            req.user_supplements, set(mapped_ings), target_skin_type, target_concern
        )

        candidate_scores: Dict[str, Dict[str, Any]] = {}

        def register_nutrient_candidate(nut_name: str, add_score: int, tag: str, reason_txt: str, skin_rationale: str):
            if nut_name not in candidate_scores:
                candidate_scores[nut_name] = {
                    "nutrient": nut_name,
                    "score": add_score,
                    "priority_tag": tag,
                    "custom_reason": reason_txt,
                    "skin_type_rationale": skin_rationale
                }
            else:
                candidate_scores[nut_name]["score"] += add_score
                candidate_scores[nut_name]["priority_tag"] = f"{candidate_scores[nut_name]['priority_tag']} + {tag}"
                if reason_txt not in candidate_scores[nut_name]["custom_reason"]:
                    candidate_scores[nut_name]["custom_reason"] += f" 또한, {reason_txt}"

        skin_type_nut_rationales = {
            "건성": "피지 분비와 수분 보유력이 낮은 건성 피부의 진피 보습력과 지질 장벽을 채워줍니다.",
            "지성": "피지선 과다 활동을 조절하여 지성 피부의 번들거림과 모공 트러블 발생을 억제합니다.",
            "수부지": "속수분 부족과 겉유분 과다를 동시에 겪는 수부지 피부의 유수분 밸런스를 맞춰줍니다.",
            "복합성": "T존 피지 조절과 U존 수분·에너지 회복을 동시에 지원하는 복합성 맞춤 배합입니다.",
            "민감성": "장벽이 약해 쉽게 붉어지는 민감성 피부의 면역 과민 반응을 진정시키고 장벽을 튼튼히 합니다.",
            "여드름성": "면포·염증성 트러블이 잦은 여드름성 피부의 피지 억제와 상처 조직 재생을 돕습니다."
        }
        base_st_rationale = skin_type_nut_rationales.get(target_skin_type, st_profile["type_guide"])

        for idx_st, st_nut in enumerate(st_profile.get("inner_nutrients", [])):
            register_nutrient_candidate(
                st_nut,
                45 - idx_st * 2,
                f"[{target_skin_type}] 피부유형 핵심",
                f"선택하신 [{target_skin_type}] 피부 유형의 체질 개선을 위한 핵심 영양소입니다.",
                f"[{target_skin_type} 맞춤 근거] {base_st_rationale}"
            )

        for idx_c, c_nut in enumerate(concern_criteria.get("inner_nutrients", [])):
            register_nutrient_candidate(
                c_nut,
                38 - idx_c * 3,
                f"[{target_concern}] 고민 타겟",
                f"집중 타겟 고민인 [{target_concern}] 개선에 직접 작용하는 이너뷰티 원료입니다.",
                f"[{target_skin_type} × {target_concern}] 피부 유형과 주된 고민을 동시에 케어하기 위해 우선 선정되었습니다."
            )

        if sleep_val == 0:
            for idx_s, s_nut in enumerate(["마그네슘", "L-테아닌", "비타민B군"]):
                register_nutrient_candidate(
                    s_nut,
                    32 - idx_s * 3,
                    "수면 부족 보완",
                    "생활패턴 중 [수면 부족(6시간 미만)]으로 인한 야간 피부 재생 저하 및 스트레스 완화를 돕습니다.",
                    f"[{target_skin_type}] 피부의 수면 부족 시 떨어지는 야간 장벽 회복력을 보충합니다."
                )

        is_high_uv_env = (env_uv is not None and env_uv >= 6.0) or (env_uv_max is not None and env_uv_max >= 6.0)
        if outdoor_val == 2 or is_high_uv_env:
            uv_label_str = f"UV {env_uv_max if env_uv_max is not None else env_uv} · 높음" if is_high_uv_env else "야외활동 많음"
            for idx_o, o_nut in enumerate(["비타민C", "아스타잔틴", "글루타치온"]):
                register_nutrient_candidate(
                    o_nut,
                    34 - idx_o * 3,
                    f"자외선({uv_label_str}) 방어",
                    f"[{uv_label_str}] 환경에서 발생하는 활성산소 및 광노화·색소 침착 방어를 지원합니다.",
                    f"[{target_skin_type}] 피부가 자외선 노출 후 겪는 산화 스트레스와 색소 침착을 줄여줍니다."
                )

        if alcohol_val >= 1:
            alc_label = "주 3회 이상 음주" if alcohol_val == 2 else "주 1~2회 음주"
            for idx_a, a_nut in enumerate(["밀크씨슬", "히알루론산", "비타민B군"]):
                register_nutrient_candidate(
                    a_nut,
                    (33 if alcohol_val == 2 else 25) - idx_a * 3,
                    f"{alc_label} 해독·수분",
                    f"생활패턴 중 [{alc_label}]로 인한 간 해독 부담과 알코올성 수분 손실 보충을 돕습니다.",
                    f"[{target_skin_type}] 피부가 음주 후 급격히 건조해지고 트러블이 올라오는 것을 예방합니다."
                )

        if workout_val == 2:
            for idx_w, w_nut in enumerate(["코엔자임Q10", "비타민C"]):
                register_nutrient_candidate(
                    w_nut,
                    28 - idx_w * 3,
                    "고강도 운동 항산화",
                    "생활패턴 중 [고강도 운동] 후 증가하는 활성산소 제거와 피부 탄력 유지를 돕습니다.",
                    f"[{target_skin_type}] 피부의 운동 후 열감·산화 스트레스를 빠르게 진정시킵니다."
                )

        fallback_pool = [
            ("히알루론산", 18, "기초 수분 보충", "진피층 수분 보유력을 높여 속건조를 예방합니다."),
            ("비타민C", 17, "기초 항산화 케어", "콜라겐 합성과 맑은 피부 톤 유지를 돕습니다."),
            ("유산균(프로바이오틱스)", 16, "장-피부 면역 밸런스", "장내 환경 개선을 통해 피부 민감도를 낮춥니다."),
            ("비타민B군", 15, "기초 피부 대사", "피부 세포 에너지 대사와 컨디션 회복을 지원합니다."),
            ("오메가3", 14, "필수 지방산 보습", "세포막 보호와 피부 장벽 유지를 돕습니다."),
            ("저분자 콜라겐", 13, "진피 탄력 보강", "피부 탄력 섬유 유지에 도움을 줍니다.")
        ]
        for fb_nut, fb_score, fb_tag, fb_reason in fallback_pool:
            register_nutrient_candidate(
                fb_nut,
                fb_score,
                fb_tag,
                fb_reason,
                f"[{target_skin_type}] 피부의 전반적인 컨디션 유지를 위한 보조 영양소입니다."
            )

        ranked_candidates = sorted(candidate_scores.values(), key=lambda x: -x["score"])

        detailed_supplements = build_supplement_dosing_recommendations(
            conn=conn,
            ranked_nutrient_candidates=ranked_candidates,
            user_registered_nutrients=user_taking_nutrients,
            supplement_limit=supp_limit
        )
        conn.close()

        combined_good_ings = list(dict.fromkeys(st_profile["good_ings"] + concern_criteria["good_ings"]))
        combined_avoid_ings = list(dict.fromkeys(st_profile["avoid_ings"] + concern_criteria["avoid_ings"]))

        matched_good_cnt = len([g for g in combined_good_ings if any(g in m or m in g for m in mapped_ings)])
        matched_avoid_cnt = len([a for a in combined_avoid_ings if any(a.split("(")[0] in m for m in mapped_ings)])

        base_type_score = {
            "건성": 24, "지성": 23, "수부지": 25, "복합성": 24, "민감성": 22, "여드름성": 23
        }.get(target_skin_type, 24)
        if target_skin_type in ["건성", "탄력/주름"] and style_id >= 3:
            base_type_score += 2
        elif target_skin_type in ["민감성", "지성"] and style_id <= 3:
            base_type_score += 2
        concern_pt = min(30, max(16, base_type_score + min(matched_good_cnt * 2, 6) - (matched_avoid_cnt * 3)))

        req_total = max(1, len(required_types))
        covered_cnt = max(0, req_total - len(missing_steps))
        vanity_pt = round((covered_cnt / req_total) * 25) if len(selected_cosmetics) > 0 else 12

        harmony_pt = max(8, round((safety_score / 100.0) * 20))
        if len(supp_cross_notes) > 0 and any("높은 주의" in n for n in supp_cross_notes):
            harmony_pt = max(6, harmony_pt - 5)

        lifestyle_penalty = (2 if sleep_val == 0 else 0) + (2 if alcohol_val == 2 else (1 if alcohol_val == 1 else 0)) + (2 if outdoor_val == 2 and "SUNCARE" not in owned_macro_roles else 0)
        lifestyle_pt = max(8, 15 - lifestyle_penalty)

        top_priority_nuts = {c["nutrient"] for c in ranked_candidates[:4]}
        overlap_supp_cnt = len(user_taking_nutrients.intersection(top_priority_nuts))
        inner_pt = min(10, 6 + min(len(user_supp_cards), 2) + min(overlap_supp_cnt * 2, 4) - len(supp_dup_warnings) * 2)
        inner_pt = max(5, inner_pt)

        total_100_score = min(100, concern_pt + vanity_pt + harmony_pt + lifestyle_pt + inner_pt)
        utilization_rate = round((len(all_used_keys) / len(selected_cosmetics)) * 100) if len(selected_cosmetics) > 0 else 0

        elapsed_ms = round((time.time() - start_ts) * 1000, 1)

        lifestyle_desc = (
            f"수면·음주·야외활동 및 실제 날씨({env_temp}°C/습도{env_hum}%)"
            if (has_env_data and env_temp is not None and env_hum is not None)
            else "수면·음주·야외활동 생활패턴 대응 (기상 실측 미수신)"
        )

        return {
            "status": "success",
            "execution_time_ms": elapsed_ms,
            "env_reflected": {
                "is_available": has_env_data,
                "data_source": env_ctx.data_source or "UNAVAILABLE",
                "summary": env_summary_text if has_env_data else "기상 데이터 미수신 (생활패턴·피부유형 기준 분석)",
                "uv_index": env_uv,
                "uv_level": cur_uv_meta["level"],
                "uv_max_today": env_uv_max,
                "uv_max_level": max_uv_meta["level"],
                "humidity": env_hum,
                "temp": env_temp,
                "pm25": env_pm25,
                "rationale_items": env_rationale_list
            },
            "safety_assessment": {
                "safety_score": safety_score,
                "conflicts": conflicts
            },
            "score_evaluation": {
                "total_score": total_100_score,
                "utilization_rate": utilization_rate,
                "used_owned_count": len(all_used_keys),
                "total_owned_count": len(selected_cosmetics),
                "has_insufficient_vanity": len(selected_cosmetics) == 0,
                "disclaimer": "위 100점 만점 배점은 선택한 피부 유형·피부 고민·보유 화장대·생활패턴·복용 영양제 데이터를 종합한 알고리즘 적합도 지표입니다.",
                "breakdown": [
                    {"label": "피부 유형·고민 적합도", "score": concern_pt, "max_score": 30, "desc": f"[{target_skin_type} · {target_concern}] 맞춤 성분·스타일 부합도"},
                    {"label": "보유 화장품 적합도", "score": vanity_pt, "max_score": 25, "desc": f"필수 제형 충족률 ({covered_cnt}/{req_total}단계 보유)"},
                    {"label": "성분 궁합 및 안전성", "score": harmony_pt, "max_score": 20, "desc": "화장품·영양제 상극 및 피부유형 자극 안전도"},
                    {"label": "생활패턴·환경 방어력", "score": lifestyle_pt, "max_score": 15, "desc": lifestyle_desc},
                    {"label": "영양·이너뷰티 연계성", "score": inner_pt, "max_score": 10, "desc": f"복용 영양제({len(user_supp_cards)}종) 및 맞춤 영양소({len(detailed_supplements)}종) 시너지"}
                ]
            },
            "ai_recommendation": {
                "archetype_id": predicted_arch_id,
                "skin_type": target_skin_type,
                "skin_type_label": st_profile["label"],
                "target_concern": archetype_data.get("concern", target_concern),
                "care_style": archetype_data.get("style_name", f"{style_id}단계 맞춤 루틴"),
                "match_rate_percentage": match_rate,
                "coaching_tip": f"[{target_skin_type}] {st_profile['type_guide']}",
                "overall_ingredient_guide": {
                    "good_ingredients": combined_good_ings[:7],
                    "avoid_ingredients": combined_avoid_ings[:5],
                    "summary_guide": f"[{target_skin_type} × {target_concern}] 맞춤 가이드: {st_profile['type_guide']} {concern_criteria['summary_guide']}"
                }
            },
            "routine_diagnosis": {
                "missing_essential_steps": missing_steps,
                "missing_step_recommendations": missing_step_recommendations,
                "overused_steps": overused_steps,
                "excluded_vanity_products": excluded_vanity_products,
                "morning_sunscreen_missing": "SUNCARE" not in owned_macro_roles,
                "is_balanced": len(missing_steps) == 0 and len(overused_steps) == 0,
                "owned_products_count": len(selected_cosmetics)
            },
            "time_based_schedule": {
                "morning_routine": morning_steps,
                "night_routine": night_steps,
                "morning_detailed": morning_detailed,
                "night_detailed": night_detailed,
                "weekly_special_care": weekly_special
            },
            "inner_beauty_matching": {
                "supplement_limit": supp_limit,
                "user_registered_supplements": user_supp_cards,
                "duplicate_warnings": supp_dup_warnings,
                "cross_cosmetic_notes": supp_cross_notes,
                "recommended_nutrients": [d["nutrient"] for d in detailed_supplements],
                "detailed_supplements": detailed_supplements,
                "reason": f"선택하신 [{target_skin_type}] 피부 유형, [{target_concern}] 고민, 5대 생활패턴 및 현재 복용 중인 영양제({len(user_supp_cards)}종)를 통합 분석해 중복 없이 상위 {len(detailed_supplements)}종을 우선순위 추천했습니다."
            }
        }

    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"[ERROR] /api/v1/routine/recommend failed:\n{tb_str}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_code": "SERVER_ERROR",
                "user_message": "추천 알고리즘 실행 중 서버 내부 오류가 발생했습니다.",
                "dev_log": f"{type(e).__name__}: {str(e)}"
            }
        )


# 하위 호환용 이너뷰티 단독 분석 엔드포인트
class InnerBeautyAnalyzeRequest(BaseModel):
    skin_type: Optional[str] = "수부지"
    supplement_limit: Optional[int] = 3
    supplements: List[Dict[str, Any]] = Field(default_factory=list)
    vanity_items: List[Dict[str, Any]] = Field(default_factory=list)
    concern: str = Field("수분/속건조")
    sleep: int = Field(1)
    outdoor: int = Field(1)
    alcohol: int = Field(0)
    workout: int = Field(1)

@app.post("/api/v1/supplements/analyze")
def analyze_inner_beauty(req: InnerBeautyAnalyzeRequest):
    routine_req = RoutineRecommendRequest(
        skin_type=req.skin_type or "수부지",
        concern=req.concern,
        care_style_id=3,
        supplement_limit=req.supplement_limit or 3,
        lifestyle=LifestyleInput(makeup=1, sleep=req.sleep, outdoor=req.outdoor, alcohol=req.alcohol, workout=req.workout),
        user_supplements=req.supplements,
        vanity_items=[VanityCustomItem(**v) if isinstance(v, dict) else v for v in req.vanity_items]
    )
    full_res = recommend_routine(routine_req)
    if isinstance(full_res, dict) and "inner_beauty_matching" in full_res:
        return {"status": "success", **full_res["inner_beauty_matching"]}
    return full_res


# ==============================================================================
# [7] 화장품 508개 & 영양제 45,869건 검색 API & OCR 화장품 자동 인식 고도화 API
# ==============================================================================
@app.get("/api/cosmetics/search", summary="화장품 키워드 및 30개 세부 제형 검색 (더미 데이터 자동 제외)")
def search_cosmetics(
    q: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    item_type: Optional[str] = Query(None),
    limit: int = Query(25, ge=1, le=100)
):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT COUNT(*) as cnt FROM products
        WHERE category = 'cosmetic'
          AND product_id NOT IN (9001, 9002, 9003, 9004)
          AND product_name NOT IN ('샘플A', '샘플B', '샘플C', '샘플D')
    """)
    total_cosmetics_in_db = cur.fetchone()["cnt"]

    query = """
        SELECT product_id, report_no, company, product_name, sub_category, item_type, raw_ingredients_text
        FROM products
        WHERE category = 'cosmetic'
          AND product_id NOT IN (9001, 9002, 9003, 9004)
          AND product_name NOT IN ('샘플A', '샘플B', '샘플C', '샘플D')
    """
    params = []
    if q:
        query += " AND (company LIKE ? OR product_name LIKE ? OR raw_ingredients_text LIKE ?)"
        params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
    if category:
        query += " AND sub_category = ?"
        params.append(category)
    if item_type:
        query += " AND item_type = ?"
        params.append(item_type)

    query += " ORDER BY CASE WHEN report_no LIKE 'USER_OCR_%' THEN 0 ELSE 1 END ASC, product_id DESC LIMIT ?"
    params.append(limit)

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        raw_text = r["raw_ingredients_text"] or ""
        act_ings = list(dict.fromkeys([normalize_ing_name(i) for i in extract_standard_ingredients(raw_text)]))
        time_info = classify_product_time_slot(r["item_type"], act_ings, r["product_name"], r["sub_category"])
        is_user_ocr = str(r["report_no"] or "").startswith("USER_OCR_")
        results.append({
            "product_id": r["product_id"],
            "goods_no": r["report_no"],
            "brand": r["company"],
            "name": r["product_name"],
            "category": r["sub_category"],
            "item_type": r["item_type"],
            "sample_ingredients": make_sample_ings(raw_text),
            "active_ingredients": act_ings,
            "raw_ingredients_text": raw_text,
            "time_slot": time_info,
            "is_user_registered": is_user_ocr
        })
    return {
        "total": len(results),
        "total_db_cosmetics": total_cosmetics_in_db,
        "items": results
    }


@app.get("/api/supplements/search", summary="식약처 건강기능식품 실시간 검색")
def search_supplements(q: Optional[str] = Query(None), limit: int = Query(20, ge=1, le=100)):
    conn = get_db()
    cur = conn.cursor()
    query = """
        SELECT product_id, report_no, company, product_name, primary_function, caution, raw_ingredients_text
        FROM products WHERE category = 'Supplement'
    """
    params = []
    if q:
        query += " AND (company LIKE ? OR product_name LIKE ? OR raw_ingredients_text LIKE ?)"
        params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
    query += " ORDER BY product_id ASC LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    items = []
    for r in rows:
        raw_text = r["raw_ingredients_text"] or ""
        items.append({
            "product_id": r["product_id"],
            "report_no": r["report_no"],
            "brand": r["company"],
            "name": r["product_name"],
            "primary_function": r["primary_function"],
            "sample_ingredients": make_sample_ings(raw_text),
            "raw_ingredients_text": raw_text
        })
    return {"total": len(items), "items": items}


# ------------------------------------------------------------------------------
# [OCR 화장품 자동 인식 & DB 대조 매칭 헬퍼]
# ------------------------------------------------------------------------------
KNOWN_COSMETIC_BRANDS: List[str] = [
    "라운드랩", "토리든", "아누아", "메디힐", "닥터지", "에스트라", "일리윤", "아이소이",
    "구달", "마녀공장", "넘버즈인", "성분에디터", "바이오더마", "라로슈포제", "피지오겔",
    "브링그린", "비플레인", "코스알엑스", "스킨1004", "이니스프리", "설화수", "헤라",
    "라네즈", "아이오페", "센텔리안24", "파티온", "달바", "아비브", "바이오힐보",
    "차앤박", "CNP", "더마토리", "세타필", "아벤느", "유리아쥬", "가히", "식물나라",
    "미샤", "어퓨", "클리오", "페리페라", "롬앤", "에뛰드", "스킨푸드", "네이처리퍼블릭",
    "더페이스샵", "토니모리", "홀리카홀리카", "한율", "프리메라", "오휘", "숨37", "빌리프"
]

FORMULATION_KEYWORDS: List[Tuple[str, str]] = [
    ("토너패드", "토너패드"), ("패드", "토너패드"),
    ("스킨", "스킨/토너"), ("토너", "스킨/토너"), ("미스트", "스킨/토너"),
    ("앰플", "에센스/세럼/앰플"), ("세럼", "에센스/세럼/앰플"), ("에센스", "에센스/세럼/앰플"),
    ("로션", "로션/에멀젼"), ("에멀젼", "로션/에멀젼"), ("에멀전", "로션/에멀젼"), ("올인원", "로션/에멀젼"),
    ("수분크림", "크림/밤"), ("수딩크림", "크림/밤"), ("재생크림", "크림/밤"), ("크림", "크림/밤"), ("밤", "크림/밤"),
    ("클렌징오일", "클렌징 오일"), ("클렌징 오일", "클렌징 오일"), ("클렌징밤", "클렌징 오일"),
    ("클렌징폼", "클렌징 폼/젤"), ("클렌징 폼", "클렌징 폼/젤"), ("약산성폼", "클렌징 폼/젤"), ("클렌저", "클렌징 폼/젤"), ("폼", "클렌징 폼/젤"),
    ("선세럼", "선세럼/선젤/플루이드"), ("선젤", "선세럼/선젤/플루이드"), ("선플루이드", "선세럼/선젤/플루이드"),
    ("선크림", "선크림/로션"), ("선스틱", "선스틱"), ("자외선", "선크림/로션"), ("선케어", "선크림/로션"), ("SPF", "선크림/로션"),
    ("마스크팩", "시트마스크"), ("시트마스크", "시트마스크"), ("마스크", "시트마스크"), ("워시오프팩", "워시오프팩/슬리핑팩"), ("슬리핑팩", "워시오프팩/슬리핑팩")
]

def extract_and_match_cosmetic_from_ocr(
    conn: sqlite3.Connection,
    raw_ocr_tokens: List[str],
    filename: str = "",
    brand_hint: str = "",
    name_hint: str = "",
    type_hint: str = "AUTO"
) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute("""
        SELECT product_id, report_no, company, product_name, sub_category, item_type, raw_ingredients_text
        FROM products
        WHERE category = 'cosmetic'
          AND product_id NOT IN (9001, 9002, 9003, 9004)
          AND product_name NOT IN ('샘플A', '샘플B', '샘플C', '샘플D')
    """)
    db_cosmetics = cur.fetchall()

    db_brands = sorted(list({r["company"].strip() for r in db_cosmetics if r["company"] and len(r["company"].strip()) >= 2}), key=len, reverse=True)
    all_brand_candidates = list(dict.fromkeys(db_brands + KNOWN_COSMETIC_BRANDS))

    clean_tokens = [str(t).strip() for t in (raw_ocr_tokens or []) if str(t).strip()]
    base_fname = os.path.splitext(filename or "")[0]
    if (
        base_fname.startswith("camera_label_")
        or base_fname.startswith("image")
        or base_fname.startswith("IMG_")
        or base_fname.startswith("스크린샷")
        or base_fname.startswith("Screenshot")
    ):
        base_fname = ""
    base_fname = re.sub(r"[_\\-]+", " ", base_fname).strip()

    ingredient_tokens = []
    for tok in clean_tokens:
        tok_clean = re.sub(r"^\[.*?\]", "", tok).strip()
        if not tok_clean:
            continue
        if any(skip in tok_clean for skip in ["전성분", "제조업자", "책임판매업자", "사용기한", "용량", "화장품법"]):
            continue
        for sub_t in tok_clean.split(","):
            st = sub_t.strip()
            if st:
                ingredient_tokens.append(st)

    dedup_ingredients = list(dict.fromkeys(ingredient_tokens))
    raw_ing_text = ", ".join(dedup_ingredients)
    std_ings = list(dict.fromkeys([normalize_ing_name(i) for i in extract_standard_ingredients(raw_ing_text)]))
    if not std_ings and dedup_ingredients:
        std_ings = list(dict.fromkeys([normalize_ing_name(i) for i in dedup_ingredients[:6]]))

    search_pool_text = f"{brand_hint} {name_hint} {base_fname}".strip()

    extracted_brand = brand_hint.strip()
    if not extracted_brand and search_pool_text:
        for b_cand in all_brand_candidates:
            if b_cand and b_cand in search_pool_text:
                extracted_brand = b_cand
                break

    extracted_item_type = type_hint if (type_hint and type_hint != "AUTO") else ""
    if not extracted_item_type and search_pool_text:
        for kw, mapped_type in FORMULATION_KEYWORDS:
            if kw in search_pool_text:
                extracted_item_type = mapped_type
                break

    extracted_name = name_hint.strip() or base_fname
    norm_query_name = re.sub(r"\s+", "", extracted_name).lower()
    norm_pool = re.sub(r"\s+", "", search_pool_text).lower()

    ocr_ing_set = {x.replace(" ", "") for x in dedup_ingredients if len(x) >= 2}

    best_match_row = None
    best_match_score = 0

    for row in db_cosmetics:
        db_b = (row["company"] or "").strip()
        db_n = (row["product_name"] or "").strip()
        db_raw = (row["raw_ingredients_text"] or "").strip()
        if not db_n:
            continue

        norm_db_n = re.sub(r"\s+", "", db_n).lower()
        norm_db_b = re.sub(r"\s+", "", db_b).lower()

        score = 0
        if len(norm_db_n) >= 3 and (norm_db_n in norm_pool or (norm_query_name and len(norm_query_name) >= 3 and norm_query_name in norm_db_n)):
            score += 65

        db_words = [w for w in db_n.split() if len(w) >= 2]
        matched_words = [w for w in db_words if w in search_pool_text]
        if db_words:
            word_ratio = len(matched_words) / len(db_words)
            if word_ratio >= 0.6:
                score += int(word_ratio * 45)
            elif len(matched_words) >= 2:
                score += 25

        if extracted_brand and db_b and (extracted_brand == db_b or norm_db_b in norm_pool):
            score += 25

        if len(ocr_ing_set) >= 5 and db_raw:
            db_ing_set = {
                re.sub(r"\(.*?\)", "", p).replace(" ", "").strip()
                for p in db_raw.split(",")
                if len(p.strip()) >= 2
            }
            if db_ing_set:
                inter_cnt = len(ocr_ing_set.intersection(db_ing_set))
                union_cnt = len(ocr_ing_set.union(db_ing_set))
                jaccard = inter_cnt / max(1, union_cnt)
                overlap_ratio = inter_cnt / max(1, min(len(ocr_ing_set), len(db_ing_set)))
                if jaccard >= 0.55 or (inter_cnt >= 8 and overlap_ratio >= 0.70):
                    score += int(max(jaccard, overlap_ratio) * 85)

        if score > best_match_score:
            best_match_score = score
            best_match_row = row

    if best_match_row is not None and best_match_score >= 55:
        db_raw_txt = best_match_row["raw_ingredients_text"] or raw_ing_text
        db_std_ings = list(dict.fromkeys([normalize_ing_name(i) for i in extract_standard_ingredients(db_raw_txt)]))
        if not db_std_ings:
            db_std_ings = std_ings
        return {
            "db_matched": True,
            "match_score": best_match_score,
            "product_id": int(best_match_row["product_id"]),
            "brand": best_match_row["company"] or extracted_brand or "등록 브랜드",
            "name": best_match_row["product_name"],
            "sub_category": best_match_row["sub_category"] or "Skincare",
            "item_type": best_match_row["item_type"] or extracted_item_type or "에센스/세럼/앰플",
            "raw_ingredients_text": db_raw_txt,
            "active_ingredients": db_std_ings
        }

    if not extracted_brand:
        extracted_brand = ""
    if not extracted_name:
        extracted_name = ""

    sub_cat, resolved_item_type = infer_db_category_and_item_type(extracted_name or raw_ing_text[:40], extracted_item_type or "AUTO")

    return {
        "db_matched": False,
        "match_score": best_match_score,
        "product_id": 0,
        "brand": extracted_brand,
        "name": extracted_name,
        "sub_category": sub_cat,
        "item_type": resolved_item_type,
        "raw_ingredients_text": raw_ing_text,
        "active_ingredients": std_ings
    }


@app.post("/api/v1/ocr/analyze-image", summary="라벨 사진 OCR 비동기 자동 인식, DB 대조 및 확인/수동 등록")
async def analyze_label_image(
    file: Optional[UploadFile] = File(None),
    brand: Optional[str] = Form(""),
    product_name: Optional[str] = Form(""),
    item_type_hint: Optional[str] = Form("AUTO"),
    action: Optional[str] = Form("AUTO"),
    raw_ingredients_override: Optional[str] = Form(""),
    matched_product_id: Optional[int] = Form(0)
):
    act_mode = (action or "AUTO").strip().upper()
    conn = get_db()
    cur = conn.cursor()

    if act_mode == "SAVE":
        clean_brand = (brand or "").strip() or "사용자 직접 등록"
        clean_name = (product_name or "").strip()
        if not clean_name:
            conn.close()
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "제품명을 입력해 주세요."}
            )

        sub_cat, resolved_item_type = infer_db_category_and_item_type(clean_name, item_type_hint or "AUTO")
        raw_text_joined = (raw_ingredients_override or "").strip()

        existing_row = None
        if matched_product_id and int(matched_product_id) > 0:
            cur.execute("SELECT * FROM products WHERE product_id = ? AND category = 'cosmetic'", (int(matched_product_id),))
            existing_row = cur.fetchone()

        if not existing_row:
            cur.execute("""
                SELECT * FROM products
                WHERE category = 'cosmetic' AND (
                    (company = ? AND product_name = ?)
                    OR product_name = ?
                )
                LIMIT 1
            """, (clean_brand, clean_name, clean_name))
            existing_row = cur.fetchone()

        if existing_row and not raw_text_joined:
            raw_text_joined = existing_row["raw_ingredients_text"] or ""

        if not raw_text_joined:
            raw_text_joined = "정제수, 글리세린, 부틸렌글라이콜, 판테놀, 히알루론산"

        std_ings = list(dict.fromkeys([normalize_ing_name(i) for i in extract_standard_ingredients(raw_text_joined)]))
        if not std_ings:
            std_ings = list(dict.fromkeys([normalize_ing_name(s.strip()) for s in raw_text_joined.split(",")[:5] if s.strip()]))

        if existing_row and not str(existing_row["report_no"] or "").startswith("USER_OCR_"):
            final_pid = int(existing_row["product_id"])
            db_action = "DB_MATCHED"
            clean_brand = existing_row["company"] or clean_brand
            clean_name = existing_row["product_name"] or clean_name
            sub_cat = existing_row["sub_category"] or sub_cat
            resolved_item_type = existing_row["item_type"] or resolved_item_type
            raw_text_joined = existing_row["raw_ingredients_text"] or raw_text_joined
            std_ings = list(dict.fromkeys([normalize_ing_name(i) for i in extract_standard_ingredients(raw_text_joined)])) or std_ings
        else:
            final_pid, db_action = upsert_cosmetic_to_db(
                conn=conn,
                brand=clean_brand,
                product_name=clean_name,
                sub_category=sub_cat,
                item_type=resolved_item_type,
                raw_ingredients_text=raw_text_joined,
                active_ingredients=std_ings
            )

        cur.execute("SELECT COUNT(*) as cnt FROM products WHERE category = 'cosmetic'")
        total_cosmetics_count = cur.fetchone()["cnt"]
        conn.close()

        time_info = classify_product_time_slot(resolved_item_type, std_ings, clean_name, sub_cat)
        return {
            "status": "success",
            "ocr_recognized": True,
            "db_saved": True,
            "db_action": db_action,
            "db_matched": db_action == "DB_MATCHED",
            "total_db_cosmetics": total_cosmetics_count,
            "product": {
                "product_id": final_pid,
                "brand": clean_brand,
                "name": clean_name,
                "category": sub_cat,
                "item_type": resolved_item_type,
                "active_ingredients": std_ings,
                "sample_ingredients": make_sample_ings(raw_text_joined),
                "raw_ingredients_text": raw_text_joined,
                "time_slot": time_info,
                "is_user_registered": db_action != "DB_MATCHED"
            },
            "raw_ingredients": [s.strip() for s in raw_text_joined.split(",") if s.strip()],
            "matched_active_ingredients": std_ings
        }

    if not file or not file.filename:
        conn.close()
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "ocr_recognized": False,
                "message": "업로드된 이미지 파일이 없습니다. 수동 등록을 진행해 주세요."
            }
        )

    # 동시 업로드 충돌 방지를 위한 고유 임시 파일명 부여
    safe_orig_name = os.path.basename(file.filename)
    unique_tmp_name = f"{int(time.time() * 1000)}_{secrets.token_hex(4)}_{safe_orig_name}"
    save_path = os.path.join("uploads", unique_tmp_name)

    ocr_recognized = True
    ocr_error_msg = ""
    raw_ings: List[str] = []

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # CPU 집중 EasyOCR 연산을 별도 스레드에서 실행하여 서버 블로킹 방지
        raw_ings = await run_in_threadpool(process_label_image, save_path)
        if not raw_ings or len(raw_ings) == 0:
            ocr_recognized = False
            ocr_error_msg = "라벨 이미지에서 텍스트를 충분히 인식하지 못했습니다. 아래 입력칸에서 직접 확인·수동 입력하여 등록할 수 있습니다."
    except Exception as ocr_err:
        ocr_recognized = False
        ocr_error_msg = f"OCR 인식 중 오류가 발생했습니다({type(ocr_err).__name__}). 수동 등록 모드로 전환합니다."
        raw_ings = []
    finally:
        # 처리 완료 후 임시 업로드 파일 정리
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except Exception:
                pass

    extracted_info = extract_and_match_cosmetic_from_ocr(
        conn=conn,
        raw_ocr_tokens=raw_ings,
        filename=safe_orig_name,
        brand_hint=brand or "",
        name_hint=product_name or "",
        type_hint=item_type_hint or "AUTO"
    )

    if not ocr_recognized and not extracted_info["db_matched"]:
        conn.close()
        return {
            "status": "success",
            "ocr_recognized": False,
            "db_saved": False,
            "db_matched": False,
            "fallback_manual": True,
            "message": ocr_error_msg or "자동 인식에 실패했습니다. 제품 정보를 수동으로 입력해 주세요.",
            "product": {
                "product_id": 0,
                "brand": (brand or "").strip(),
                "name": (product_name or "").strip(),
                "category": extracted_info["sub_category"],
                "item_type": extracted_info["item_type"],
                "active_ingredients": [],
                "sample_ingredients": "",
                "raw_ingredients_text": "",
                "is_user_registered": True
            },
            "filename": safe_orig_name,
            "raw_ingredients": [],
            "matched_active_ingredients": []
        }

    if act_mode == "EXTRACT":
        conn.close()
        time_info = classify_product_time_slot(
            extracted_info["item_type"],
            extracted_info["active_ingredients"],
            extracted_info["name"] or "스킨케어",
            extracted_info["sub_category"]
        )
        return {
            "status": "success",
            "ocr_recognized": True,
            "db_saved": False,
            "db_matched": extracted_info["db_matched"],
            "match_score": extracted_info["match_score"],
            "message": (
                f"기존 DB에서 일치하는 제품([{extracted_info['brand']}] {extracted_info['name']})을 찾아 정보를 불러왔습니다."
                if extracted_info["db_matched"]
                else f"전성분 {len(raw_ings)}종을 추출 및 자동 교정했습니다. 브랜드·제품명을 확인(또는 입력)한 뒤 등록해 주세요."
            ),
            "product": {
                "product_id": extracted_info["product_id"],
                "brand": extracted_info["brand"],
                "name": extracted_info["name"],
                "category": extracted_info["sub_category"],
                "item_type": extracted_info["item_type"],
                "active_ingredients": extracted_info["active_ingredients"],
                "sample_ingredients": make_sample_ings(extracted_info["raw_ingredients_text"]),
                "raw_ingredients_text": extracted_info["raw_ingredients_text"],
                "time_slot": time_info,
                "is_user_registered": not extracted_info["db_matched"]
            },
            "filename": safe_orig_name,
            "raw_ingredients": raw_ings,
            "matched_active_ingredients": extracted_info["active_ingredients"]
        }

    if extracted_info["db_matched"] and extracted_info["product_id"] > 0:
        new_product_id = extracted_info["product_id"]
        db_action = "DB_MATCHED"
    else:
        save_brand = extracted_info["brand"] or "사용자 스캔 등록"
        save_name = extracted_info["name"] or f"라벨 스캔 화장품 ({datetime.now().strftime('%H:%M')})"
        raw_text_joined = extracted_info["raw_ingredients_text"] or "정제수, 글리세린, 부틸렌글라이콜, 판테놀, 히알루론산, 병풀추출물"
        std_ings = extracted_info["active_ingredients"] or ["히알루론산", "판테놀"]
        new_product_id, db_action = upsert_cosmetic_to_db(
            conn=conn,
            brand=save_brand,
            product_name=save_name,
            sub_category=extracted_info["sub_category"],
            item_type=extracted_info["item_type"],
            raw_ingredients_text=raw_text_joined,
            active_ingredients=std_ings
        )
        extracted_info["brand"] = save_brand
        extracted_info["name"] = save_name
        extracted_info["raw_ingredients_text"] = raw_text_joined
        extracted_info["active_ingredients"] = std_ings

    cur.execute("SELECT COUNT(*) as cnt FROM products WHERE category = 'cosmetic'")
    total_cosmetics_count = cur.fetchone()["cnt"]
    conn.close()

    time_info = classify_product_time_slot(
        extracted_info["item_type"],
        extracted_info["active_ingredients"],
        extracted_info["name"],
        extracted_info["sub_category"]
    )

    return {
        "status": "success",
        "ocr_recognized": True,
        "db_saved": True,
        "db_action": db_action,
        "db_matched": extracted_info["db_matched"],
        "total_db_cosmetics": total_cosmetics_count,
        "product": {
            "product_id": new_product_id,
            "brand": extracted_info["brand"],
            "name": extracted_info["name"],
            "category": extracted_info["sub_category"],
            "item_type": extracted_info["item_type"],
            "active_ingredients": extracted_info["active_ingredients"],
            "sample_ingredients": make_sample_ings(extracted_info["raw_ingredients_text"]),
            "raw_ingredients_text": extracted_info["raw_ingredients_text"],
            "time_slot": time_info,
            "is_user_registered": not extracted_info["db_matched"]
        },
        "filename": safe_orig_name,
        "raw_ingredients": raw_ings,
        "matched_active_ingredients": extracted_info["active_ingredients"]
    }