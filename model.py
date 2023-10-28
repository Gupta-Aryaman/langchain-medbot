from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.llms import LlamaCpp
from huggingface_hub import hf_hub_download

from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorstores/db_faiss"
# model = "./model/llama-2-7b-chat.ggmlv3.q8_0.bin"

model_path = hf_hub_download(
  repo_id="TheBloke/Llama-2-7b-Chat-GGUF",
  filename="llama-2-7b-chat.Q4_K_M.gguf",
  resume_download=True,
  cache_dir="./models/",  #custom path for save the model
)
# If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.
custom_prompt_template = """Use the following pieces of information to answer the user's question. 

Context: {context}
Question: {question}

Only returns the detailed and helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
    prompt = PromptTemplate(template = custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    # llm = CTransformers(
    #     model = "llama-2-7b-chat.ggmlv3.q8_0.bin", 
    #     model_type = "llama",
    #     max_new_tokens = 512,
    #     temperature = 0.5,
    #     device="cpu"
    # )
    llm = LlamaCpp(
        model_path = model_path,
        n_gpu_layers=100, #According to your GPU if you have
        n_batch=2048,
        verbose=True,
        f16_kv=True,
        n_ctx=4096
    )
    return llm

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(seearch_kwargs = {'k': 2}),
        return_source_documents = True, #explaining the answer
        chain_type_kwargs= {'prompt': prompt},
    )

    return qa_chain

def qa_bot():
    print("Hi, in qa_bot")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    print("Hi, in final_result")
    response = qa_result({'query': query, 'max_length': 5000})
    return response

## CHAINLIT
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hello, I am a Med bot. Ask me anything!"
    await msg.update()
    cl.user_session.set("chain", chain) 

@cl.on_message
async def main(message):
    print(message.content)
    chain = cl.user_session.get("chain")
    print("chain: ", type(chain))
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True, 
        answer_prefix_tokens = ["FINAL", "ANSWER"]
    )
    cb.answers_reached = True
    res = await chain.acall(message.content, callbacks = [cb])
    res = final_result(message.content)
    answer = res["result"]
    sources = res["source_documents"]
    print(res)
    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += f"\nNo sources found"
    # print(answer)
    await cl.Message(content=answer).send()