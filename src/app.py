import streamlit as st
from model import ImageClassifier
from PIL import Image
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI 이미지 분석기",
    page_icon="🖼️"
)

# 모델 캐싱 (반복 로딩 방지)
@st.cache_resource
def get_model():
    return ImageClassifier()

# 모델 설정
model = get_model()

# 이미지 준비 함수
def load_image(uploaded_file):   
    '''
    업로드된 이미지를 열고,
    224x224 크기로 조정하여 모델에 적합하게 만듭니다.
    '''
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    image = image.resize((224, 224))
    
    return image

# 이미지 분류 함수
def classify_image(image):
    '''
    업로드된 이미지를 모델에 입력하여 분류 결과를 반환합니다.
    '''
    results = model(image)
    return results

# UI 레이아웃 구성
st.title("AI 이미지 분석 서비스")
st.info("이미지를 업로드하면 AI가 분석 결과를 알려드립니다.")

# 이미지 업로드 (사진 파일 / 카메라)
input_method = st.radio("이미지 입력 방법을 선택해주세요.",
                        ("파일 업로드", "카메라로 촬영")
                        )

uploaded_file = None

if input_method == "파일 업로드":
    uploaded_file = st.file_uploader("이미지 파일을 선택해주세요", type=["jpg", "jpeg", "png"])

elif input_method == "카메라로 촬영":
    uploaded_file = st.camera_input("사진을 찍어주세요")

if uploaded_file is not None:
    st.image(uploaded_file, caption='업로드된 이미지')

if st.button("분류 시작"):
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        
        # 분류 실행
        with st.spinner("이미지 분류 중..."):
            results = model(image)

        st.success("분류 완료!")    

        # 결과 출력
            # 상위 1개 결과를 강조하여 표시
            # `st.progress()`를 활용하여 확률을 시각적으로 표현
            # 퍼센트 형식으로 신뢰도 표시 (예: 92.3%)
            
        st.write("분류 결과:")
        # 가장 높은 스코어 강조
        top_result = results[0]

        top_label = top_result["label"]
        top_score = top_result["score"]
        
        st.subheader(f"🏆 예측 결과: 이 이미지는 {top_label}일 확률이 가장 높습니다.")
        st.metric(
            label="신뢰도",
            value=f"{top_score * 100:.1f}%"
        )
        st.progress(int(top_score * 100))
        st.write("---")

        # 나머지 스코어
        st.caption("그 외 분류 결과")
        for result in results[1:]:
            st.write(f"{result['label']}: {result['score']*100:.1f}%")
            st.progress(int(result['score'] * 100))

        # top5 차트 시각화
        # 데이터 준비 
        top5 = results[:5]
        df = pd.DataFrame({
            "label": [r["label"] for r in top5],
            "confidence": [r["score"] * 100 for r in top5]
        })

        # Ploytly 막대 차트
        fig = px.bar(
            df,
            x="label",
            y="confidence",
            text="confidence",
            labels={
                "label": "분류 라벨",
                "confidence": "신뢰도 (%)"
            },
            title="Top-5 이미지 분류 결과"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_range=[0, 100],
            uniformtext_minsize=10,
            uniformtext_mode="hide"
        )

        st.plotly_chart(fig, use_container_width=True)

        
    else:
        st.write("이미지를 업로드하고 버튼을 눌러주세요.")
