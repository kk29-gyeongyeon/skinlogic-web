"""
SkinLogic - 35 Multi-Tier Routine Archetypes Definition (Final Complete)
7대 피부 고민 x 5대 케어 스타일 = 총 35개 기준 표본 루틴 정의
"""
from typing import List, Dict, Any

# 1. 7대 피부 고민 축
SKIN_CONCERNS: List[str] = [
    "수분/속건조",
    "장벽/민감진정",
    "탄력/주름",
    "미백/잡티",
    "트러블/피지",
    "홍조/열감",
    "모공/각질"
]

# 2. 5대 사용자 케어 스타일 축
CARE_STYLES: List[Dict[str, Any]] = [
    {
        "style_id": 1,
        "name": "1단계: 원스텝 미니멀",
        "step_count": 1,
        "target_types": ["크림/밤", "로션/에멀젼", "올인원/세트"],
        "desc": "화장품 다이어트 또는 귀차니즘 유저를 위한 단일 고보습·복합 케어"
    },
    {
        "style_id": 2,
        "name": "2단계: 에센셜",
        "step_count": 2,
        "target_types": ["스킨/토너", "크림/밤"],
        "desc": "수분 충전과 유분 장벽 잠금에 집중한 실속형 2단계 루틴"
    },
    {
        "style_id": 3,
        "name": "3~4단계: 밸런스 정석",
        "step_count": 4,
        "target_types": ["클렌징 폼/젤", "스킨/토너", "에센스/세럼/앰플", "크림/밤", "선크림"],
        "desc": "클렌징부터 기능성 세럼과 자외선 차단까지 아우르는 정석 루틴"
    },
    {
        "style_id": 4,
        "name": "4~5단계: 패드·레이어링",
        "step_count": 5,
        "target_types": ["클렌징 폼/젤", "토너패드", "에센스/세럼/앰플", "크림/밤", "선세럼/선젤/플루이드"],
        "desc": "패드로 피부결을 정돈하고 목적별 기능성 앰플을 레이어링하는 트렌디 루틴"
    },
    {
        "style_id": 5,
        "name": "5단계+: 풀케어 맥시멀",
        "step_count": 6,
        "target_types": ["클렌징 오일", "클렌징 폼/젤", "토너패드", "에센스/세럼/앰플", "크림/밤", "시트 마스크", "선크림"],
        "desc": "2중 세안과 주간 스페셜 팩을 포함한 고영양 집중 홈에스테틱 루틴"
    }
]

def build_archetype_matrix() -> Dict[int, Dict[str, Any]]:
    """7대 고민 x 5대 스타일 = 35종 표본 루틴 딕셔너리 생성"""
    matrix = {}
    archetype_id = 0

    for concern in SKIN_CONCERNS:
        for style in CARE_STYLES:
            # 케어 스타일별 맞춤 스케줄링 구성
            if style["style_id"] == 1:
                morning_steps = ["물세안 또는 약산성 클렌저", f"고보습 {concern} 올인원/크림", "선케어"]
                night_steps = ["약산성 클렌저", f"고보습 {concern} 올인원/크림 도포"]
                special = "바르는 단계가 최소화되어 있으므로, 주 1회 수분 패드로 각질 정돈 권장"
            elif style["style_id"] == 2:
                morning_steps = ["약산성 클렌저", f"{concern} 스킨/토너", "가벼운 크림", "선크림"]
                night_steps = ["약산성 클렌저", f"{concern} 스킨/토너 흡수", "밀폐 보습 크림"]
                special = "건조감이 심할 경우 저녁에 토너를 2~3회 레이어링(N스킨법) 권장"
            elif style["style_id"] == 3:
                morning_steps = ["약산성 클렌저", "진정 토너", f"{concern} 데일리 세럼", "수분크림", "선크림 (SPF50+)"]
                night_steps = ["딥 클렌징 폼", "진정 토너", f"{concern} 고농축 앰플", "장벽 재생 크림"]
                special = "기능성 성분의 충돌을 피하기 위해 아침(항산화)과 저녁(재생) 앰플을 분리 도포"
            elif style["style_id"] == 4:
                morning_steps = ["클렌징 폼", f"{concern} 퀵 토너패드", "수분 세럼", "장벽 크림", "선세럼/선젤"]
                night_steps = ["클렌징 폼", "닦토/패드 결 정돈", f"{concern} 타깃 앰플 2종 교차", "고보습 크림"]
                special = "패드로 인한 물리적 자극을 줄이기 위해 아침에는 닦아내지 말고 팩처럼 3분간 얹어두기 권장"
            else:  # style_id == 5
                morning_steps = ["약산성 클렌저", "결 정돈 패드", f"{concern} 세럼", "아이크림", "수분크림", "선크림"]
                night_steps = ["클렌징 오일 (1차)", "클렌징 폼 (2차)", "토너", f"{concern} 앰플", "아이크림", "밀폐 수면크림"]
                special = "주 2~3회 시트 마스크팩 적용 시 성분 과부하 방지를 위해 나이트 앰플 단계를 생략(Skip)"

            matrix[archetype_id] = {
                "archetype_id": archetype_id,
                "concern": concern,
                "style_id": style["style_id"],
                "style_name": style["name"],
                "step_count": style["step_count"],
                "required_types": style["target_types"],
                "morning_steps": morning_steps,
                "night_steps": night_steps,
                "weekly_special": special,
                "coaching_tip": f"[{concern}] 관리를 위해 {style['name']} 기준에 맞춰 유효 성분의 중복과 누락을 방어하세요."
            }
            archetype_id += 1

    return matrix

ARCHETYPE_MATRIX = build_archetype_matrix()

def get_archetype_by_id(arch_id: int) -> Dict[str, Any]:
    """ID로 표본 루틴 단일 조회"""
    return ARCHETYPE_MATRIX.get(arch_id, ARCHETYPE_MATRIX[0])

def find_archetype_id(concern: str, style_id: int) -> int:
    """피부 고민과 스타일 ID로 표본 번호(0~34) 역추적"""
    for arch_id, data in ARCHETYPE_MATRIX.items():
        if data["concern"] == concern and data["style_id"] == style_id:
            return arch_id
    return 0