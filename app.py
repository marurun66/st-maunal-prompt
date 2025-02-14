import streamlit as st
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # 수정된 모듈
from llama_index.core import Settings, StorageContext, load_index_from_storage
import os
from huggingface_hub import snapshot_download



def get_huggingface_token():
    token = os.environ.get('HUGGINGFACE_API_TOKEN')

    # 토큰이 환경변수에 없으면, 로컬에서 동작하니까 로컬에서 읽어오도록한다.
    if token is None :
        token = st.secrets.get('HUGGINGFACE_API_TOKEN')

    return token

@st.cache_resource  # 캐시로 저장
def initialize_models():
    # 사용할 모델 가져오기 (허깅페이스에서)
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    token = get_huggingface_token()

    llm = HuggingFaceInferenceAPI(  # 기존 HuggingFaceInferenceAPI → HuggingFaceLLM으로 변경
        model_name=model_name,
        max_new_tokens=512,
        temperature=0,  # 문서에 충실하게 0, 자유도 높게 1
        system_prompt="당신은 한국어로 대답하는 AI 어시스턴트입니다. 주어진 질문에 대해서만 한국어로 명확하게 답변해주세요. 응답의 마지막 부분은 단어로 끝내지 말고 문장으로 끝내도록 해주세요.",  # JSON 등의 데이터 형식 또는 말투
        api_key=token  # 허깅페이스 토큰
    )
    
    embed_model_name = "sentence-transformers/all-mpnet-base-v2"
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)  # 올바른 클래스 사용

    # 라마인덱스에 필요한 인덱스 세팅 (디폴트 OpenAI 방지)
    Settings.llm = llm
    Settings.embed_model = embed_model


def get_index_from_huggingface():
    repo_id ="marurun66/manual-index" #레파지토리 데이터셋
    local_dir="./manual_index_storage" #로컬 디렉토리, 폴더 만들어 저장
    token=get_huggingface_token
     #허깅페이스에 있는 데이터를 로컬로 다운로드 
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="dataset",
        token=token,
    )
    #다운로드 한 폴더를 메모리에 올린다.
    storage_context=StorageContext.from_defaults(persist_dir=local_dir) #로컬에 저장한 데이터를 메모리에 올린다.
    index=load_index_from_storage(storage_context)
    #파일이 커지면 메모리에 못올라간다. >데이터베이스를 이용해서 올린다.
    #지금파일은 작으니까 메모리에 올린다.
    return index



def main():
    #1. 사용할 모델 가져온다.
    #2. 사용할 토크나이저 세팅 : embed_model
    # 1,2는 세트 한번에 처리하자 (함수로)
    initialize_models() #매번 모델 저장하면 오래걸리니 @st.cache_resource #캐시로 저장


    #3. RAG에 필요한 인덱스 세팅
    index=get_index_from_huggingface() #허깅페이스로부터 인덱스를 받아오는 함수

    #4. 사용자에게 프롬프트 입력 받아서 응답
    st.title('PDF 문서 기반 질의응답 시스템')
    st.text('선진기업 복지 업무메뉴얼을 기반으로 질의응답을 제공합니다.')
    query_engine=index.as_query_engine()
    print(query_engine)
    #라마인덱스는 인덱스 기반으로 질문에 대답한다.
    #인덱스를 쓰기 위해서는 쿼리엔진이 필요하다.
    prompt = st.text_input('질문을 입력하세요.')
    if prompt :
        with st.spinner('답변 생성중...'):
            response = query_engine.query(prompt)
            st.text('답변: ')
            st.info(response.response)

    
if __name__ == '__main__':
    main()
