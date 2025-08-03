import streamlit as st
import fitz
import torch

st.set_page_config(page_title="Resume Coach AI", page_icon="🧠")
st.title(" Resume Coach Chatbot")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file is not None:
    #  Extract text from uploaded PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    resume_text = ""
    for page in doc:
        resume_text += page.get_text()

    st.subheader(" Extracted Resume Text")
    st.write(resume_text[:1500])

    #  Chunk + Embed
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(resume_text)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embedding_model)

    #  LLM + Retrieval Pipeline
    from transformers import pipeline
    from langchain.llms import HuggingFacePipeline
    from langchain.chains import RetrievalQA

    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    #  Chat Interface
    st.subheader(" Ask Your Resume a Question")

    example_qs = [
        "What are my top technical skills?",
        "Do I have experience with machine learning?",
        "Is my resume suitable for a Data Scientist role?",
        "How can I improve my resume for FAANG?",
        "Do I have cloud or big data experience?"
    ]

    st.markdown(" **Examples:**")
    for q in example_qs:
        st.markdown(f"- {q}")

    user_query = st.text_input("🔍 Ask a question about your resume:")

    if user_query:
        with st.spinner("Analyzing your resume..."):
            response = qa_chain.invoke({"query": user_query})
            result = response["result"].strip()

        st.success(" Here's what I found:")
        st.markdown(f"**Answer:** {result[:700]}{'...' if len(result) > 700 else ''}")
