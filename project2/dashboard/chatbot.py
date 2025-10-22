# Gemini API 설정
GEMINI_MODEL_NAME = 'gemini-2.5-flash'
try:
    API_KEY = "YOUR-API-KEY"
    if API_KEY == "YOUR_API_KEY_HERE":
        raise KeyError("API 키가 설정되지 않았습니다.")
    genai.configure(api_key=API_KEY)
except KeyError:
    startup_error = "GEMINI_API_KEY가 설정되지 않았습니다. 챗봇을 사용할 수 없습니다."
    print(f"ERROR: {startup_error}")
except Exception as e:
    startup_error = f"Gemini API 키 설정 오류: {e}"
    print(f"ERROR: {startup_error}")

# --------------------------------
# [서버]
    @output
    @render.ui
    def chatbot_popup():
        if not chatbot_visible.get():
            return None
    
        return ui.div(
            ui.div(
                style=(
                    "position: fixed; top: 0; left: 0; width: 100%; height: 100%; "
                    "background-color: rgba(0, 0, 0, 0.5); z-index: 1050;"
                )
            ),
            ui.div(
                ui.div("🤖 AI 챗봇", class_="fw-bold mb-2", style="font-size: 22px; text-align:center;"),
                ui.div(
                    ui.output_ui("chatbot_response"),
                    style=(
                        "height: 600px; overflow-y: auto; border: 1px solid #ddd; border-radius: 10px; "
                        "padding: 15px; background-color: #f0f4f8; margin-bottom: 12px; font-size: 14px; line-height: 1.4;"
                    )
                ),
                ui.div(
                    ui.input_text("chat_input", "", placeholder="메시지를 입력하세요...", width="80%"),
                    ui.input_action_button("send_chat", "전송", class_="btn btn-primary", style="width: 18%; margin-left: 2%;"),
                    style="display: flex; align-items: center;"
                ),
                ui.input_action_button("close_chatbot", "닫기 ✖", class_="btn btn-secondary mt-3 w-100"),
                style=(
                    "position: fixed; bottom: 90px; right: 20px; width: 800px; background-color: white; "
                    "border-radius: 15px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25); "
                    "z-index: 1100; padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;"
                )
            )
        )  
    
    @reactive.effect
    @reactive.event(input.toggle_chatbot)
    def _():
        chatbot_visible.set(not chatbot_visible.get())
    
    @reactive.Effect
    @reactive.event(input.close_chatbot)
    def _():
        chatbot_visible.set(False)
      
# --------------------------------------------------
    def get_dashboard_summary(current_data_df: pd.DataFrame):
        status_text = "🟢 공정 진행 중"
        if was_reset():
            status_text = "🟡 리셋됨"
        elif not is_streaming():
            status_text = "🔴 일시 정지됨"

        anomaly_label = {0: "양호", 1: "경고"}.get(latest_anomaly_status(), "N/A")
        defect_label = {0: "양품", 1: "불량"}.get(latest_defect_status(), "N/A")
        
        latest_time_str = "데이터 없음"
        if not current_data_df.empty and "registration_time" in current_data_df.columns:
             latest_time = pd.to_datetime(current_data_df["registration_time"], errors='coerce').max()
             if not pd.isna(latest_time):
                 latest_time_str = latest_time.strftime("%Y-%m-%d %H:%M:%S")

        stats = get_realtime_stats(current_data_df)
        total_count = stats.get("total", 0)
        anomaly_rate = stats.get("anomaly_rate", 0.0)
        anomaly_count = stats.get("anomaly_count", 0)
        defect_rate = stats.get("defect_rate", 0.0)
        today_defect_rate = stats.get("today_defect_rate", 0.0)
        accuracy = stats.get("defect_accuracy", 0.0)
        goal_progress = stats.get("goal_progress", 0.0)
        goal_target = stats.get("goal_target", 'N/A')

        cum_perf = cumulative_performance()
        cum_recall = f"{cum_perf['recall'] * 100:.2f}%"
        cum_precision = f"{cum_perf['precision'] * 100:.2f}%"
    
        latest_perf = latest_performance_metrics()
        latest_recall = f"{latest_perf['recall'] * 100:.2f}%"
        latest_precision = f"{latest_perf['precision'] * 100:.2f}%"

        perf_status = performance_degradation_status()
        perf_status_text = "🚨 성능 저하 감지" if perf_status["degraded"] else "✅ 성능 양호"

        drift_stat = data_drift_status()
        drift_status_text = f"🚨 드리프트 의심 ({drift_stat.get('feature', 'N/A')})" if drift_stat["degraded"] else "✅ 분포 양호"
        
        defect_log_count = len(defect_logs())
        feedback_count = len(r_feedback_data())

        summary = {
            "공정_상태": status_text, "최신_시간": latest_time_str,
            "최근_이상치_상태": anomaly_label, "최근_불량_상태": defect_label,
            "총_처리_건수": total_count, 
            "이상치_탐지율": f"{anomaly_rate:.2f}%",
            "이상치_탐지_건수": anomaly_count,
            "불량_탐지율_전체": f"{defect_rate:.2f}%", "오늘_불량률": f"{today_defect_rate:.2f}%",
            "모델_예측_정확도": f"{accuracy:.2f}%",
            "목표_달성률": f"{goal_progress:.2f}% (목표: {goal_target}개)",
            "누적_재현율": cum_recall, "누적_정밀도": cum_precision,
            "최근_청크_재현율": latest_recall, "최근_청크_정밀도": latest_precision,
            "모델_성능_상태": perf_status_text,
            "데이터_분포_상태": drift_status_text,
            "불량_로그_건수": defect_log_count, "피드백_총_건수": feedback_count,
        }
        return summary

    KEYWORD_TO_INFO = {
        "상태": ["공정_상태", "총_처리_건수", "최신_시간"],
        "현재": ["공정_상태", "불량_탐지율_전체", "최신_시간", "오늘_불량률"],
        "지금": ["공정_상태", "최신_시간"],
        "오늘": ["오늘_불량률", "총_처리_건수"],
        "멈췄": ["공정_상태"], "리셋": ["공정_상태"],
        "이상치": ["최근_이상치_상태", "이상치_탐지율", "이상치_탐지_건수"],
        "불량": ["최근_불량_상태", "불량_탐지율_전체", "오늘_불량률", "불량_로그_건수"],
        "불량률": ["불량_탐지율_전체", "오늘_불량률"],
        "정확도": ["모델_예측_정확도"],
        "재현율": ["누적_재현율", "최근_청크_재현율"],
        "정밀도": ["누적_정밀도", "최근_청크_정밀도"],
        "성능": ["모델_성능_상태", "누적_재현율", "누적_정밀도"],
        "드리프트": ["데이터_분포_상태"],
        "분포": ["데이터_분포_상태"],
        "목표": ["목표_달성률"],
        "피드백": ["피드백_총_건수"],
        "총": ["총_처리_건수"], "최신": ["최신_시간"],
    }

    def generate_summary_for_gemini(summary, query):
        query_lower = query.lower().replace(" ", "")
        required_keys = set()
        for keyword, keys in KEYWORD_TO_INFO.items():
            if keyword in query_lower:
                required_keys.update(keys)

        if not required_keys or any(k in query_lower for k in ["전체", "요약", "모든", "현황", "알려줘"]):
            required_keys = {
                "공정_상태", "총_처리_건수", "이상치_탐지율", "이상치_탐지_건수",
                "불량_탐지율_전체", "모델_예측_정확도",
                "누적_재현율", "누적_정밀도", "모델_성능_상태", "데이터_분포_상태"
            }

        info_parts = []
        base_keys = ["공정_상태", "총_처리_건수", "최신_시간"]
        for key in base_keys:
             if key in summary:
                 info_parts.append(f"[{key.replace('_', ' ')}]: {summary[key]}")
        
        for key, value in summary.items():
            if key in required_keys and key not in base_keys:
                info_parts.append(f"[{key.replace('_', ' ')}]: {value}")

        return "\n".join(info_parts)
# -------------------------------------------

    def filter_df_by_question(df, query):
        df_filtered = pd.DataFrame()
        analyze_type = "No Match" 

        if 'registration_time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['registration_time']):
                try:
                    df = df.copy()
                    df['registration_time'] = pd.to_datetime(df['registration_time'], errors='coerce')
                    df = df.dropna(subset=['registration_time'])
                except Exception as e:
                    print(f"시간 데이터 변환 오류: {e}")
                    return pd.DataFrame(), "시간 데이터 변환 오류"
        else:
             return pd.DataFrame(), "시간 컬럼('registration_time') 없음"

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        count_pattern = re.compile(r'(?:최근|가장 최근)\s*(\d+)\s*(?:개|건)')
        count_match = count_pattern.search(query)

        if count_match:
            count = int(count_match.group(1))
            df_sorted = df.sort_values('registration_time')
            df_filtered = df_sorted.tail(count).copy()
            analyze_type = f"필터링된 건수: 최근 {len(df_filtered)}건"
            return df_filtered, analyze_type

        start_date, end_date = None, None
        query_lower = query.lower().replace(" ", "")

        if '오늘' in query_lower or '당일' in query_lower:
            start_date = today
            end_date = today + timedelta(days=1)
            analyze_type = "필터링된 기간: 오늘"
        elif '어제' in query_lower or '전일' in query_lower:
            start_date = today - timedelta(days=1)
            end_date = today
            analyze_type = "필터링된 기간: 어제"
        elif '이번주' in query_lower or '금주' in query_lower:
            start_date = today - timedelta(days=today.weekday())
            end_date = start_date + timedelta(weeks=1)
            analyze_type = "필터링된 기간: 이번 주"
        elif '지난주' in query_lower or '전주' in query_lower:
            start_of_this_week = today - timedelta(days=today.weekday())
            start_date = start_of_this_week - timedelta(weeks=1)
            end_date = start_of_this_week
            analyze_type = "필터링된 기간: 지난 주"

        if start_date is not None:
            df_sorted = df.sort_values('registration_time')
            df_filtered = df_sorted[
                (df_sorted['registration_time'] >= start_date) &
                (df_sorted['registration_time'] < end_date)
            ].copy()

            if df_filtered.empty:
                return pd.DataFrame(), f"{analyze_type} (데이터 없음)"

            return df_filtered, f"{analyze_type} (총 {len(df_filtered)}건)"
        
        return pd.DataFrame(), "No Match"
# ---------------------------------------------


    @reactive.Effect
    @reactive.event(input.send_chat)
    async def handle_chat_send():
        query = input.chat_input().strip()
        if not query:
            ui.notification_show("질문을 입력해주세요.", duration=3, type="warning")
            return
        
        ui.update_text("chat_input", value="")
        process_chat_query(query)

    def process_chat_query(query: str):
        if not API_KEY:
            r_ai_answer.set("❌ Gemini API 키가 설정되지 않아 챗봇을 사용할 수 없습니다.")
            return

        r_is_loading.set(True)
        r_ai_answer.set("")

        df = current_data()
        if df.empty:
            r_ai_answer.set("❗ 데이터가 없습니다. 스트리밍을 시작해주세요.")
            r_is_loading.set(False)
            return

        dashboard_summary = get_dashboard_summary(df)
        df_filtered, analyze_type = filter_df_by_question(df, query)

        if df_filtered.empty and analyze_type != "No Match":
            r_ai_answer.set(f"❗ '{analyze_type}'에 대한 데이터는 찾을 수 없습니다.")
            r_is_loading.set(False)
            return

        date_range_info = dashboard_summary.get("최신_시간", "N/A")
        defect_count_info = "불량 예측 결과 없음"
        if not df_filtered.empty and 'defect_status' in df_filtered.columns:
            label_counts = df_filtered['defect_status'].value_counts()
            defect_count = label_counts.get(1, 0)
            good_count = label_counts.get(0, 0)
            total_count_filtered = label_counts.sum()
            defect_rate_filtered = (defect_count / total_count_filtered) * 100 if total_count_filtered > 0 else 0
            defect_count_info = f"필터링된 {total_count_filtered}건 분석 중 (불량: {defect_count}건, 양품: {good_count}건, 불량률: {defect_rate_filtered:.2f}%)"
            
            if 'registration_time' in df_filtered.columns:
                try:
                    min_date = df_filtered['registration_time'].min().strftime('%Y-%m-%d %H:%M')
                    max_date = df_filtered['registration_time'].max().strftime('%Y-%m-%d %H:%M')
                    date_range_info = f"기간: {min_date} ~ {max_date}"
                except Exception:
                    date_range_info = "기간 정보 오류"

        latest_defect_id_info = "불량 제품 ID 정보 없음."
        defect_log_df = defect_logs.get()
        if not defect_log_df.empty and 'ID' in defect_log_df.columns:
            latest_ids_raw = defect_log_df['ID'].tail(20).tolist()
            latest_ids = list(map(str, latest_ids_raw))
            latest_defect_id_info = f"최근 불량 제품 20건의 ID: {', '.join(latest_ids)}"

        summary_text = generate_summary_for_gemini(dashboard_summary, query)
        prompt = f"""
        당신은 공정 모니터링 대시보드의 AI 챗봇입니다.
        아래 [대시보드 핵심 정보]와 [데이터 분석 결과]를 참고하여, 사용자의 질문에 대해 명확하고 간결하게 답변해 주세요. 답변은 한국어로 작성해주세요.

        ---
        **[대시보드 핵심 정보 (탭 1 & 3)]**
        {summary_text}
        
        **[데이터 분석 결과 (질문 기반 필터링)]**
        - 분석 대상: {analyze_type}
        - 분석 대상 기간/시점: {date_range_info}
        - {defect_count_info}
        - {latest_defect_id_info}

        ---
        사용자의 질문: "{query}"

        **답변 가이드:**
        1. 질문의 핵심 키워드를 파악하세요.
        2. 질문에 해당하는 정보가 [대시보드 핵심 정보]에 있다면, 해당 정보를 중심으로 답변하세요.
        3. 질문이 특정 기간이나 건수를 명시했다면, [데이터 분석 결과]를 우선적으로 사용하세요.
        4. 수치에는 단위를 명확히 표시하고, 중요한 정보는 **굵게** 표시해 주세요.
        5. 답변은 친절하고 전문적인 톤을 유지하세요.
        """
        
        try:
            model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            # ✅ 타임아웃을 포함한 동기식 호출 (30초)
            import threading
            result = {"text": None, "error": None}
            
            def api_call():
                try:
                    response = model.generate_content(prompt)
                    result["text"] = response.text.strip()
                except Exception as e:
                    result["error"] = str(e)
            
            thread = threading.Thread(target=api_call, daemon=True)
            thread.start()
            thread.join(timeout=30)  # 30초 타임아웃
            
            if result["text"]:
                r_ai_answer.set(result["text"])
            elif result["error"]:
                error_msg = result["error"]
                if "API_KEY" in error_msg or "401" in error_msg:
                    r_ai_answer.set("❌ Gemini API 키가 유효하지 않습니다. 환경 변수를 확인하세요.")
                else:
                    r_ai_answer.set(f"❌ AI 응답 오류: {error_msg}")
            else:
                r_ai_answer.set("❌ AI 응답 타임아웃 (30초 초과). 나중에 다시 시도해주세요.")
                
        except Exception as e:
            r_ai_answer.set(f"❌ 예상치 못한 오류: {str(e)}")
            print(f"ERROR: {e}")
        
        finally:
            # ✅ 항상 로딩 상태 해제
            r_is_loading.set(False)

    @output
    @render.ui
    def chatbot_response():
        if r_is_loading.get():
            return ui.div(
                 ui.div({"class": "spinner-border text-primary", "role": "status"}, 
                        ui.span({"class": "visually-hidden"}, "Loading...")),
                 ui.p("AI가 답변을 생성 중입니다...", style="margin-left: 10px; color: #555;"),
                 style="display: flex; align-items: center; justify-content: center; height: 100%;"
            )

        return ui.markdown(r_ai_answer.get())


