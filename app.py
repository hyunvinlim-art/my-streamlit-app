 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/app.py b/app.py
index 8361eeba2c3eaa697385fcba1373626f171d0d46..7da7cf3711a5152b7b456537a702d4db89430e06 100644
--- a/app.py
+++ b/app.py
@@ -1,37 +1,43 @@
 import streamlit as st
 from openai import OpenAI
 
-st.title("🤖 나의 AI 챗봇")
-
-# 사이드바에서 API Key 입력
-api_key = st.sidebar.text_input("OpenAI API Key", type="password")
+st.title("🤖 나의 AI 챗봇")
+
+# 사이드바에서 API Key 입력
+api_key = st.sidebar.text_input("OpenAI API Key", type="password")
+st.sidebar.markdown("### 😊 오늘의 기분")
+mood = st.sidebar.selectbox(
+    "현재 기분을 선택하세요",
+    ["기쁨", "평온", "슬픔", "불안", "피곤", "설렘", "화남", "기타"],
+)
 
 # 대화 기록 초기화
 if "messages" not in st.session_state:
     st.session_state.messages = []
 
 # 이전 대화 표시
 for message in st.session_state.messages:
     with st.chat_message(message["role"]):
         st.markdown(message["content"])
 
-# 사용자 입력 처리
-if prompt := st.chat_input("메시지를 입력하세요"):
-    if not api_key:
-        st.error("⚠️ 사이드바에서 API Key를 입력해주세요!")
-    else:
-        # 사용자 메시지 저장 및 표시
-        st.session_state.messages.append({"role": "user", "content": prompt})
-        with st.chat_message("user"):
-            st.markdown(prompt)
+# 사용자 입력 처리
+if prompt := st.chat_input("메시지를 입력하세요"):
+    if not api_key:
+        st.error("⚠️ 사이드바에서 API Key를 입력해주세요!")
+    else:
+        # 사용자 메시지 저장 및 표시
+        mood_prompt = f"[기분: {mood}] {prompt}"
+        st.session_state.messages.append({"role": "user", "content": mood_prompt})
+        with st.chat_message("user"):
+            st.markdown(mood_prompt)
         
         # AI 응답 생성
         with st.chat_message("assistant"):
             client = OpenAI(api_key=api_key)
             response = client.chat.completions.create(
                 model="gpt-4o-mini",
                 messages=st.session_state.messages
             )
             reply = response.choices[0].message.content
             st.markdown(reply)
-            st.session_state.messages.append({"role": "assistant", "content": reply})
\ No newline at end of file
+            st.session_state.messages.append({"role": "assistant", "content": reply})
 
EOF
)
