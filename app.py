import os

from flask import Flask
from flask import request
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser

app = Flask(__name__)
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")
mongoUrl = os.getenv("MONGO_URL")
client = OpenAI(api_key=openai_api_key)
#anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
#client = OpenAI(api_key=anthropic_api_key)
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.mongodb import MongodbLoader
store = {}


@app.route('/getTrips', methods=['POST'])
def getTrips():
    #   llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    llm = ChatAnthropic(model="claude-2.1", temperature=1)
    query = request.json.get('query')
    session = request.json.get('session_id')
    customer_data = request.json.get('customer_data')
    customer_hypothesis = request.json.get('customer_hypothesis')
    # print("INSIDE: query:" + query + ", session:" + session + ", customer_data:" + customer_data)
    loader = JSONLoader(file_path="experiencesFull.jsonl", jq_schema=".trips[]", text_content=False)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is..    
 """
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
    qa_system_prompt = """You are an expert Travel agent and your task is to understand the Customer and their \
    requirements  by 2 Ways :\
    1) The customer data provided in customer_data  i.e Customer's sex, Age, passions , \
    main Interests, lifestyles, type of travel, travel span, travelBucketList Of the Customer and Also their Dependents\
    Considering the customer's Profession, Age , Family, upcoming Birthday and other provided information \
    Go through customer_hypothesis on customers and their dependents and consider them when suggesting a trip .      
             
    2) The customer input / questions. \
    
    Do Not Ask Questions which Answers can be found from input or customer_data or from above customer_hypothesis
    For example Do not ask Customer name , Age, their dependents, Customer and Dependent"s interests,passions, 
    lifestyles etc If they are available in customer_data \
    and any other information provided in customer_data or customer_hypothesis .\
    Also, Please Consider that The customers and All their dependents Will be joining the Trip .\
    Before providing any Suggestions to the user,You must greet the user Only First time,\
    with their First Name from the First Customer 
    and with that you will ask the user Further questions One By one In a very polite and welcoming tone
    to gather more information. Question must not be more than 2 lines, Wait for the answer And then Then Ask the other\
    question , SO JUST 1 Question at a time Please\
    
    With every question Give user some Possible Examples of the Answer to that question  \
    Ask questions keeping in mind the available Experiences we have in context \
    Ask Few Questions one by one to narrow down customer wishes and understand customer requirements more. \
    As soon you understand customer's requirements ,\
    Suggest customer, the best suitable trips from context with _id's of the suggested trips,\
    as per their input, customer_data and customer_hypothesis. \
    
    If Customer asking for some ideas from you , Suggest them some available trips from context with '_id's, which \
    matches their interests from customer_hypothesis OR customer_data or input.\
    Always Provide _id of the trip you are suggesting or referring to \
    Do Not suggest any trip which is not available in context\
    
    If there are more than one trip aligns with customer requirements , Suggest all of those trips .\
    IF the suggested Trip is not present in context , Do not show any _id for that trip and Tell Customer \
    that we do not have this trip with us at the moment ,If you like we can Prepare this trip for you. \
    Your Response must be in Parsable json format . \
    1) A Three line overview about the trips where you \
    explain the customer in detail how the suggested trips best fits with the customer from their chat, customer_data \
    and the customer_hypothesis \
    2) the '_id's of the suggested trips as given in context \
    
    Display All suggested Trip _id's separately with space separated \
    
    Use three sentences maximum and keep the answer concise. \

    {input} {context} {customer_data} {customer_hypothesis}"""
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
        {"input": query, "customer_data": customer_data, "customer_hypothesis": customer_hypothesis},
        config={
            "configurable": {"session_id": session}
        },
    )["answer"]

    return var


@app.route('/getTripsTest', methods=['POST'])
def getTripsTest():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
 #   llm = ChatAnthropic(model="claude-2.1", temperature=1)
    query = request.json.get('query')
    session = request.json.get('session_id')
    customer_data = request.json.get('customer_data')
    customer_hypothesis = request.json.get('customer_hypothesis')
    # print("INSIDE: query:" + query + ", session:" + session + ", customer_data:" + customer_data)
    loader = JSONLoader(file_path="experiencesFullBk.json", jq_schema=".trips[]", content_key=".attributes.title",
                        is_content_key_jq_parsable=True)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is..    
 """
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
    qa_system_prompt = """Your Job is to Answer customer input Only from the context

    {input} {context} {customer_data} {customer_hypothesis}"""
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
        {"input": query, "customer_data": customer_data, "customer_hypothesis": customer_hypothesis},
        config={
            "configurable": {"session_id": session}
        },
    )["answer"]

    return var



@app.route('/getTripsMongo', methods=['POST'])
def getTripsMongo():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
 #   llm = ChatAnthropic(model="claude-2.1", temperature=1)
    query = request.json.get('query')
    session = request.json.get('session_id')
    customer_data = request.json.get('customer_data')
    customer_hypothesis = request.json.get('customer_hypothesis')
    # print("INSIDE: query:" + query + ", session:" + session + ", customer_data:" + customer_data)
    loader = MongodbLoader(
        connection_string="mongodb+srv://junaidp:@cluster0-wxkrw.mongodb.net/data-entry?retryWrites=true&w=majority",
        db_name="data-entry",
        collection_name="experience",
#        filter_criteria={"borough": "Bronx", "cuisine": "Bakery"},
        field_names=["_id", "title", "description"],
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is..    
 """
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
    qa_system_prompt = """You are an expert Travel Agent,\
    Your Job is to Answer customer input Only from the context.\
    Greet the user with their name, in a polite way on the first message \
    Go through the whole context and then answer the input.\
    Do Not Ask Questions , for which the answer could be found in input, customer_data, Or customer_hypothesis. \
    Also Consider that , All the dependents which are in customer_data , will be Going for the Trip. \
    Before Suggesting some experience, First ask few questions to the user to understand better their needs, \
    Go through the input , customer_data and customer_hypothesis, To understand the customer more and considering \
    all that information , Suggest some experience/s from context.\
    And explain why you think these experiences could be a good fit for the customer, or their dependents \
    keeping in mind , customer Age , their Kids Age (if any) , interests of the customer and their dependents \
    input, customer_data, customer_hypothesis .\
    
    With Every Suggested Experience , Return the '_id's of the suggested experiences .\
   
 
    {input} {context} {customer_data} {customer_hypothesis}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ],

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
        {"input": query, "customer_data": customer_data, "customer_hypothesis": customer_hypothesis},
        config={
            "configurable": {"session_id": session}
        },
    )["answer"]

    return var


