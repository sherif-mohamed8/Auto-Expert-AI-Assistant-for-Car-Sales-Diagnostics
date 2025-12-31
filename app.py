import streamlit as st
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# ==========================================
# 1. إعدادات الصفحة والمفتاح
# ==========================================
st.set_page_config(page_title="Auto-Expert AI", page_icon="🚗", layout="wide")
st.title("🚗 Auto-Expert: مساعدك الذكي للسيارات")
st.markdown("---")

# مفتاح الـ API (تأكد إنه شغال)
api_key = "AIzaSyB6Jc9UUaYexpV6L-n0ZJKRz9TxVjskYls" 
os.environ["GOOGLE_API_KEY"] = api_key

# ==========================================
# 2. تجهيز الميكانيكي (Mechanic Brain)
# ==========================================
@st.cache_resource 
def load_mechanic():
    print("🔧 Loading Mechanic Database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("faiss_index_mechanic"):
        vectorstore = FAISS.load_local(
            "faiss_index_mechanic", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vectorstore
    return None

def get_mechanic_response(query):
    vectorstore = load_mechanic()
    if not vectorstore:
        return "❌ Error: Mechanic Database (FAISS) not found."
    
    # البحث
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])
    
    # الموديل
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.3)
    
    prompt = f"""
    Role: Expert Egyptian Mechanic.
    Context: {context}
    User Complaint: {query}
    Task: Explain cause and give 3 solution steps in Egyptian Arabic.
    """
    try:
        return llm.invoke(prompt).content
    except Exception as e:
        return f"⚠️ Mechanic Error: {str(e)}"

# ==========================================
# 3. تجهيز المبيعات (Sales Agent)
# ==========================================
@st.cache_resource
def load_sales_agent():
    print("💰 Loading Sales Agent...")
    if os.path.exists("data/cleaned_car_data.csv"):
        df = pd.read_csv("data/cleaned_car_data.csv")
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
        return create_pandas_dataframe_agent(
            llm, df, verbose=True, allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )
    return None

def get_sales_response(query):
    agent = load_sales_agent()
    if not agent:
        return "❌ Error: Car CSV Data not found."
    
    prompt = f"Query: {query}. Answer in Egyptian Arabic. Format prices clearly."
    try:
        return agent.invoke(prompt)['output']
    except Exception as e:
        return f"⚠️ Sales Error: {str(e)}"

# ==========================================
# 4. المايسترو (Router) والواجهة
# ==========================================

# ذاكرة الشات
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض الرسائل السابقة
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# استقبال السؤال
if prompt := st.chat_input("اكتب مشكلتك أو استفسارك هنا..."):
    # عرض سؤال المستخدم
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # التحليل والرد
    with st.chat_message("assistant"):
        with st.spinner("جاري التفكير... 🧠"):
            
            router_llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
            router_prompt = f"Classify: SALES (price/buy/sell) or MECHANIC (repair/issue). Query: {prompt}. Output 1 word."
            
            try:
                intent = router_llm.invoke(router_prompt).content.strip().upper()
            except:
                intent = "MECHANIC" 
            
            # التوجيه
            if intent == "SALES":
                response = get_sales_response(prompt)
                st.caption("💰 (Sales Agent)")
            else:
                response = get_mechanic_response(prompt)
                st.caption("🔧 (Mechanic Agent)")
            
            st.markdown(response)
            
    # حفظ الرد
    st.session_state.messages.append({"role": "assistant", "content": response})