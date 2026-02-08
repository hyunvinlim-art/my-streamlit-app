import json
import requests
import streamlit as st
from openai import OpenAI

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"


def tmdb_get(endpoint, tmdb_key, params=None):
    if params is None:
        params = {}
    params["api_key"] = tmdb_key
    params["language"] = "ko-KR"
    url = f"{TMDB_BASE_URL}{endpoint}"
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()


def get_genre_map(content_type="movie", tmdb_key=None):
    data = tmdb_get(f"/genre/{content_type}/list", tmdb_key)
    genres = data.get("genres", [])
    return {g["name"]: g["id"] for g in genres}


def discover_candidates(content_type, tmdb_key, genre_id=None, page=1):
    params = {
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "page": page,
        "vote_count.gte": 200,
    }
    if genre_id:
        params["with_genres"] = genre_id

    endpoint = "/discover/movie" if content_type == "movie" else "/discover/tv"
    data = tmdb_get(endpoint, tmdb_key, params=params)
    return data.get("results", [])


def get_watch_providers(content_type, tmdb_key, tmdb_id):
    endpoint = f"/{content_type}/{tmdb_id}/watch/providers"
    data = tmdb_get(endpoint, tmdb_key)

    kr = data.get("results", {}).get("KR", {})
    flatrate = kr.get("flatrate", [])
    rent = kr.get("rent", [])
    buy = kr.get("buy", [])

    providers = []
    for group in [flatrate, rent, buy]:
        for p in group:
            providers.append(p["provider_name"])

    return list(sorted(set(providers)))


def get_trailer_youtube_url(content_type, tmdb_key, tmdb_id):
    endpoint = f"/{content_type}/{tmdb_id}/videos"
    data = tmdb_get(endpoint, tmdb_key)
    results = data.get("results", [])

    for v in results:
        if v.get("site") == "YouTube" and v.get("type") in ["Trailer", "Teaser"]:
            key = v.get("key")
            if key:
                return f"https://www.youtube.com/watch?v={key}"

    return None


def build_candidate_text(candidates, content_type):
    lines = []
    for c in candidates:
        title = c.get("title") if content_type == "movie" else c.get("name")
        overview = (c.get("overview", "") or "")[:250]
        vote = c.get("vote_average", 0)

        year = ""
        if content_type == "movie":
            year = (c.get("release_date") or "")[:4]
        else:
            year = (c.get("first_air_date") or "")[:4]

        lines.append(
            f"- id={c.get('id')} | 제목={title} | 연도={year} | 평점={vote:.1f} | 줄거리={overview}"
        )
    return "\n".join(lines)


def find_candidate_by_id(candidates, chosen_id):
    for c in candidates:
        if c.get("id") == chosen_id:
            return c
    return None


def openai_next_question(openai_key, messages):
    client = OpenAI(api_key=openai_key)

    system = """
너는 영화/드라마 추천 상담사다.
지금은 추천하지 말고, 사용자의 취향과 상태를 깊게 파악해야 한다.

목표:
- 4~5턴 안에 추천에 필요한 정보를 충분히 모은다.
- 마지막에는 "추천 시작해도 될까요?" 같은 멘트로 마무리한다.

규칙:
- 질문은 반드시 1개만 해라.
- 질문은 짧고 자연스럽게.
- 과하게 친절한 말투 말고, 친구처럼 가볍게.
- 아래 항목을 자연스럽게 수집해라:
  1) 오늘 기분/상태
  2) 원하는 분위기(힐링/스릴/웃김/감동 등)
  3) 피하고 싶은 요소(잔인함/우울한 결말/공포/로맨스 등)
  4) 시청 가능 시간(짧게/길게)
  5) 영화 vs 드라마 선호
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system.strip()}] + messages,
        temperature=0.8,
    )

    return resp.choices[0].message.content.strip()


def openai_extract_profile(openai_key, messages):
    client = OpenAI(api_key=openai_key)

    system = """
너는 대화 내용을 정리하는 도우미다.
사용자와의 대화에서 영화/드라마 추천에 필요한 조건을 추출해 JSON으로 만들어라.

규칙:
- JSON만 출력
- 모르면 null 또는 "상관없음"으로 처리
- avoid는 리스트로
"""

    user_prompt = f"""
[대화 기록]
{json.dumps(messages, ensure_ascii=False, indent=2)}

아래 JSON 형태로만 출력해라:

{{
  "content_type": "movie 또는 tv 또는 상관없음",
  "mood": "사용자 기분",
  "tone": "원하는 분위기",
  "time": "15~30분 / 30~60분 / 1~2시간 / 2시간 이상 / 상관없음",
  "genre": "상관없음 또는 장르 힌트",
  "avoid": ["피하고 싶은 요소1", "피하고 싶은 요소2"]
}}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
    )

    text = resp.choices[0].message.content.strip()
    start = text.find("{")
    end = text.rfind("}")
    text = text[start:end+1]
    return json.loads(text)


def openai_pick_best(openai_key, user_profile, candidate_text, reject_count=0, reviewer_style="친구처럼 수다"):
    client = OpenAI(api_key=openai_key)
    decision_mode = reject_count >= 3

    style_guide = {
        "침착한 평론가": "차분하고 논리적으로 추천한다. 감정 과장 없이 깔끔하게 말한다.",
        "친구처럼 수다": "친구한테 추천하듯이 가볍고 재밌게 말한다. 너무 과한 밈은 쓰지 않는다.",
        "냉정한 심사위원": "단호하고 짧게 말한다. 선택장애를 끊어준다.",
        "감성 충만": "감정과 분위기를 섬세하게 묘사한다. 여운을 강조한다.",
    }

    style_text = style_guide.get(reviewer_style, style_guide["친구처럼 수다"])

    system_prompt = f"""
너는 영화/드라마 추천 전문가이자 리뷰어다.
너의 목표는 사용자가 고민을 멈추고 바로 시청을 시작하게 만드는 것이다.

컨셉:
- 이 앱은 "추천"이 아니라 "결정"을 도와주는 앱이다.
- 사용자가 추천을 여러 번 거절했다면(거절 3회 이상) 더 단호하게 추천해라.
- 추천 과정 자체가 재미있어야 한다. 유튜브 리뷰 채널처럼 '짧은 대본'을 제공해라.

리뷰어 스타일:
- {reviewer_style}
- 스타일 지침: {style_text}

규칙:
- 반드시 후보 목록에 있는 id 중 하나만 선택해라.
- 추천 근거는 사용자의 입력과 직접 연결되게 작성해라.
- 피하고 싶은 요소(avoid)가 있으면 반드시 피해서 추천해라.
- 말투는 확신 있게.
- 출력은 JSON만.
"""

    user_prompt = f"""
[사용자 프로필]
{json.dumps(user_profile, ensure_ascii=False, indent=2)}

[후보 목록]
{candidate_text}

[현재 상태]
- 사용자가 추천을 거절한 횟수: {reject_count}
- 결정 모드(3회 이상이면 True): {decision_mode}

아래 JSON 형식으로만 답해라.

{{
  "chosen_id": 123,
  "mood_insight": "사용자의 상태를 한 줄로 분석한 문장",
  "one_line": "한 줄 추천 멘트",
  "review_script": "유튜브 리뷰 채널처럼 말하는 6~10줄 정도의 짧은 대본",
  "reasons": ["이유1", "이유2", "이유3"],
  "summary": "줄거리 요약 (2~3문장)",
  "confidence_push": "사용자가 바로 보게 만드는 마지막 한마디"
}}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.7,
    )

    text = resp.choices[0].message.content.strip()
    start = text.find("{")
    end = text.rfind("}")
    text = text[start:end+1]
    return json.loads(text)


st.set_page_config(page_title="무비메이트", page_icon="🎬", layout="wide")
st.title("🎬 무비메이트 (대화형 Streamlit MVP)")
st.caption("짧게 대화하고, 오늘 딱 맞는 영화/드라마를 1개로 결정해주는 앱")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "profile" not in st.session_state:
    st.session_state.profile = None

if "recommendation" not in st.session_state:
    st.session_state.recommendation = None

if "candidates" not in st.session_state:
    st.session_state.candidates = None

if "reject_count" not in st.session_state:
    st.session_state.reject_count = 0

if "last_chosen_id" not in st.session_state:
    st.session_state.last_chosen_id = None

st.sidebar.header("🔑 API 키 입력")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
tmdb_key = st.sidebar.text_input("TMDB API Key", type="password")

st.sidebar.divider()
st.sidebar.header("⚙️ 추천 옵션")

reviewer_style = st.sidebar.selectbox(
    "리뷰어 캐릭터",
    ["친구처럼 수다", "침착한 평론가", "냉정한 심사위원", "감성 충만"],
)

content_type_kor = st.sidebar.radio("기본 추천 타입", ["영화", "드라마", "상관없음"])
content_type_default = (
    "movie" if content_type_kor == "영화" else "tv" if content_type_kor == "드라마" else None
)

if tmdb_key:
    try:
        genre_map_movie = get_genre_map("movie", tmdb_key=tmdb_key)
        genre_list = ["상관없음"] + list(genre_map_movie.keys())
    except:
        genre_map_movie = {}
        genre_list = ["상관없음"]
else:
    genre_map_movie = {}
    genre_list = ["상관없음"]

genre_choice = st.sidebar.selectbox("장르(선택)", genre_list)

sidebar_time = st.sidebar.radio(
    "시청 가능 시간",
    ["상관없음", "15~30분", "30~60분", "1~2시간", "2시간 이상"],
)

otts = st.sidebar.multiselect(
    "보유 OTT (선택)",
    ["Netflix", "Disney Plus", "TVING", "Wavve", "Coupang Play", "Watcha", "Apple TV+", "상관없음"],
)

st.sidebar.divider()

if st.sidebar.button("🧹 대화 초기화", use_container_width=True):
    st.session_state.messages = []
    st.session_state.profile = None
    st.session_state.recommendation = None
    st.session_state.candidates = None
    st.session_state.reject_count = 0
    st.session_state.last_chosen_id = None
    st.rerun()

st.subheader("💬 무비메이트 상담")

if len(st.session_state.messages) == 0:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "오늘 뭐 보고 싶어? 그냥 한 줄로 말해줘. (예: 머리 비우고 웃긴 거 보고 싶어)",
        }
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_text = st.chat_input("여기에 입력...")

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})

    if not openai_key:
        st.session_state.messages.append(
            {"role": "assistant", "content": "OpenAI 키가 있어야 대화를 이어갈 수 있어! 사이드바에 입력해줘 🔑"}
        )
        st.rerun()

    try:
        next_q = openai_next_question(openai_key, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": next_q})
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"AI 질문 생성 실패 😢\n\n에러: {e}"})

    st.rerun()

st.divider()
st.subheader("🎯 추천")

turn_count = len([m for m in st.session_state.messages if m["role"] == "user"])

if turn_count < 2:
    st.info("대화를 조금만 더 해보자! (최소 2번은 답해줘야 추천이 더 정확해져)")
else:
    st.success("좋아. 이제 추천해도 될 것 같아 😎")

    if st.button("🎬 추천 시작하기", use_container_width=True):
        if not openai_key or not tmdb_key:
            st.error("추천하려면 OpenAI 키와 TMDB 키가 모두 필요해!")
            st.stop()

        st.info("대화 내용을 정리하는 중...")
        try:
            profile = openai_extract_profile(openai_key, st.session_state.messages)
        except Exception as e:
            st.error(f"프로필 추출 실패: {e}")
            st.stop()

        if sidebar_time != "상관없음":
            profile["time"] = sidebar_time

        content_type = profile.get("content_type")
        if content_type in ["movie", "tv"]:
            final_content_type = content_type
        else:
            final_content_type = content_type_default if content_type_default else "movie"

        final_genre = genre_choice

        st.info("TMDB에서 후보를 가져오는 중...")

        genre_id = None
        if final_genre != "상관없음":
            genre_id = genre_map_movie.get(final_genre)

        candidates = discover_candidates(final_content_type, tmdb_key, genre_id=genre_id, page=1)
        candidates = candidates[:20]

        candidate_text = build_candidate_text(candidates, final_content_type)

        st.info("AI가 최종 추천을 고르는 중...")

        profile["otts"] = otts
        profile["genre"] = final_genre
        profile["content_type_final"] = final_content_type

        try:
            rec = openai_pick_best(
                openai_key,
                profile,
                candidate_text,
                reject_count=st.session_state.reject_count,
                reviewer_style=reviewer_style,
            )
        except Exception as e:
            st.error(f"추천 생성 실패: {e}")
            st.stop()

        chosen_id = rec["chosen_id"]
        chosen = find_candidate_by_id(candidates, chosen_id)

        if not chosen:
            st.error("AI가 후보 목록에 없는 id를 골랐어. 다시 추천 시작하기를 눌러줘!")
            st.stop()

        st.session_state.profile = profile
        st.session_state.recommendation = rec
        st.session_state.candidates = candidates
        st.session_state.last_chosen_id = chosen_id

        st.rerun()

if st.session_state.recommendation and st.session_state.candidates:
    rec = st.session_state.recommendation
    candidates = st.session_state.candidates
    chosen_id = rec["chosen_id"]

    profile = st.session_state.profile
    final_content_type = profile.get("content_type_final", "movie")

    chosen = find_candidate_by_id(candidates, chosen_id)

    title = chosen.get("title") if final_content_type == "movie" else chosen.get("name")
    poster_path = chosen.get("poster_path")
    overview = chosen.get("overview", "")

    providers = []
    try:
        providers = get_watch_providers(final_content_type, tmdb_key, chosen_id)
    except:
        providers = []

    trailer_url = None
    try:
        trailer_url = get_trailer_youtube_url(final_content_type, tmdb_key, chosen_id)
    except:
        trailer_url = None

    st.markdown("## ✅ 오늘의 최종 추천")

    mood_insight = rec.get("mood_insight")
    if mood_insight:
        st.info(f"🧠 오늘의 상태 분석: {mood_insight}")

    col1, col2 = st.columns([1, 2])

    with col1:
        if poster_path:
            st.image(f"{TMDB_IMG_BASE}{poster_path}", use_container_width=True)
        else:
            st.write("포스터 없음")

        st.write("")
        st.markdown("### 🎞️ 예고편")
        if trailer_url:
            st.link_button("유튜브 예고편 보기", trailer_url, use_container_width=True)
        else:
            st.write("예고편 정보 없음")

    with col2:
        st.subheader(title)
        st.markdown(f"**{rec.get('one_line', '')}**")

        st.write("")
        st.markdown("### 🎙️ 리뷰 대본 (유튜브 채널 느낌)")
        st.write(rec.get("review_script", "리뷰 대본 없음"))

        st.write("")
        st.markdown("### 🔥 추천 이유")
        for r in rec.get("reasons", []):
            st.write(f"- {r}")

        st.write("")
        st.markdown("### 📖 줄거리")
        st.write(rec.get("summary", overview))

        st.write("")
        st.markdown("### 📺 시청 가능한 OTT (KR 기준)")
        if providers:
            st.write(", ".join(providers))
        else:
            st.write("정보 없음 (또는 한국에서 제공되지 않을 수 있어요)")

        st.write("")
        st.success(rec.get("confidence_push", "지금 이거 보자. 오늘은 이게 정답이야."))

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        if st.button("✅ 오케이, 이거 볼래", use_container_width=True):
            st.balloons()
            st.success("좋아. 오늘은 고민 끝. 재생 버튼만 누르면 돼 🎬")

    with c2:
        if st.button("❌ 별로야, 다른 거 줘", use_container_width=True):
            st.session_state.reject_count += 1

            candidate_text = build_candidate_text(candidates, final_content_type)

            profile["avoid"] = (profile.get("avoid") or []) + [f"id:{chosen_id}"]

            try:
                rec2 = openai_pick_best(
                    openai_key,
                    profile,
                    candidate_text,
                    reject_count=st.session_state.reject_count,
                    reviewer_style=reviewer_style,
                )
            except Exception as e:
                st.error(f"다른 추천 생성 실패: {e}")
                st.stop()

            st.session_state.recommendation = rec2
            st.rerun()
