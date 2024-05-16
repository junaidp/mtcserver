import os

from flask import Flask
from flask import request
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

app = Flask(__name__)
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
from langchain_community.chat_models import ChatOpenAI
import json
from pathlib import Path
from langchain.agents import create_json_agent
# from langchain_community.agent_toolkits import JsonToolkit
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.document_loaders import JSONLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma

from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory

import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter

store = {}


@app.route('/withHistory1')
def withhistory1():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    query = request.json.get('query')
    session = request.json.get('session_id')
    ### Construct retriever ###
    loader = JSONLoader(file_path="./experiencesFull.json", jq_schema=".trips[]", text_content=False)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history. Return your response in Parsable JSON format"""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    You are an expert Travel agent and your task is to get the Customer requirements and then based on those. \
    requirements and wishes, Suggest some trips from the {context}, Before Suggesting trips, Ask in total 3 questions 
    One By One to user, Means Ask a question and then wait for their answer and so on ,\
    and after you get the answer for the 3rd question , Suggest user, the best suitable trips as per their wishes from \
    {context}, In your response only send 2 Json fields about the trip . 1) A 2 line overview about the trips and how they \
    best fits with the customer. \
    2) the '_id's of the suggested trips. Do not send any other information from that trip other than the 2 mentioned. \

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Statefully manage chat history ###

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    var = conversational_rag_chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": session}
        },  # constructs a key "abc123" in `store`.
    )["answer"]

    return var


@app.route('/withHistory')
def withhistory():
    embedding_function = OpenAIEmbeddings()

    loader = JSONLoader(file_path="./experiencesFull.json", jq_schema=".trips[]", text_content=False)
    documents = loader.load()

    db = Chroma.from_documents(documents, embedding_function)
    retriever = db.as_retriever()

    print("RETRIEVER")
    print(retriever)

    query = request.json.get('query')
    session_id = request.json.get('session_id')
    model = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Answer the question based only on the following context:
    {context}."""

            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    response = with_message_history.invoke(
        {"context": retriever, "input": query, "question": RunnablePassthrough()},
        config={"configurable": {"session_id": session_id}},
    )

    return str(response)


@app.route('/withouthistory')
def hello1():
    query = request.json.get('query')
    chat_history = str(request.json.get('previousChat'))
    embedding_function = OpenAIEmbeddings()

    loader = JSONLoader(file_path="./experiencesFull.json", jq_schema=".trips[]", text_content=False)
    documents = loader.load()

    db = Chroma.from_documents(documents, embedding_function)
    retriever = db.as_retriever()
    print("RETRIEVER")
    print(retriever)

    template = """Answer the question based only on the following context:
    {context}. You are an expert Travel agent and your task is to get the Customer requirements and then based on those 
    requirements and wishes, Suggest a trip from the {context}, Before Suggesting trips Ask in total 3 questions One 
    By One to user, Means Ask a question and then wait for their answer and so on ,
    and after you get the answer for the 3rd question , Suggest user the best suitable trip as per their wishes from 
    {context} and keep in mind the chat history 

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI()
    memory = ConversationBufferMemory()
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
    )

    response = chain.invoke(chat_history + query)
    # history = ChatMessageHistory()
    # history.add_user_message(query)
    # history.add_ai_message(response)
    # print(history)
    return response


@app.route('/hello')
def hello():
    # with open("experiencesSample.json") as f:
    #    data = yaml.load(f, Loader=yaml.FullLoader)
    # json_spec = JsonSpec(dict_=data, max_value_length=4000)
    # json_toolkit = JsonToolkit(spec=json_spec)

    # json_agent_executor = create_json_agent(
    #   llm=BaseLanguageModel(temperature=0), toolkit=json_toolkit, verbose=True
    # )
    # print(json_agent_executor.run("which is the most expensive experience"))

    file_path = 'experiencesFull.json'
    data = json.loads(Path(file_path).read_text())

    spec = JsonSpec(dict_=data, max_value_length=4000)
    toolkit = JsonToolkit(spec=spec)
    agent = create_json_agent(llm=ChatOpenAI(temperature=1, model="gpt-3.5-turbo"), toolkit=toolkit,
                              max_iterations=1000,
                              verbose=True)
    print(agent.run("Go through all the trips in the file ,and Only return the '_id' "
                    "of the all the trips"))
    return "d"


@app.route('/qa')
def qa():
    query = request.json.get('query')
    customer = request.json.get('customer')
    previous = request.json.get('previous')
    prompt = prompts(query, customer, previous)
    response = model(prompt)
    return response


def prompts(customer_query, customer_data, previous_chat):
    prompt = f"""
    don't ask already answered questions, also dont ask any question
    for which the answer could also be found
    in 'customer_data', Always keep in mind the 'previousChat' before asking a question or suggesting the trip 
here is user {customer_query}
here is the customer file {customer_data}
here is the user previous chat data {previous_chat}


    """
    return prompt


def model(prompt):
    file_path = 'experiencesSample.json'
    data = json.loads(Path(file_path).read_text())

    messages = [
        {"role": "system", "content": f"""

  you are an helpful ,experienced traveler chatbot ,
  based on the user query ask Question to the user following their query, Once you get the answer of that question, 
  Ask another question to further narrow down the requirements,
  After you get the answer for the second question, Ask a last question.
  Make sure you do not Ask for the information which you already have from previous user's answers or in 'customer_data'

  and should recommend trips as per their interests given in the {data} data follow the instructions and provide relevant  
  experiences id, id`s separated by comma without any descriptive text
  """},
        {'role': 'user', 'content': prompt}]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return str(response.choices[0].message.content)
