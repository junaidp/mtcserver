import os

from flask import Flask
from flask import request

app = Flask(__name__)
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
from langchain_community.chat_models import ChatOpenAI
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

store = {}


@app.route('/getTrips', methods=['POST'])
def getTrips():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)
    query = request.json.get('query')
    session = request.json.get('session_id')
    customer_data = request.json.get('customer_data')
   # print("INSIDE: query:" + query + ", session:" + session + ", customer_data:" + customer_data)
    loader = JSONLoader(file_path="./experiencesFull.json", jq_schema=".trips[]", text_content=False)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history. 
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
    Do below Hypothesis on customers and their dependents and consider these factors when suggesting a trip: \
            1. Age gap between the youngest and oldest person \
            2. Closest combinations of birthday milestones among all persons \
            3. Suggest combined celebration experience for close birthdays among persons \
                consider physical limitation based on age and their lifestyle preferences  \
            4. Use each person's age to infer their current life milestones (e.g., school, university, retirement). \
            5. Upcoming milestone for each person" \
            6. Infer the timeframe to the next supposed life milestone in months and years. \
            7. Find out where each person works If this information could be found out from their email Domain , \
                also try to find the domain on internet. \
            9. evaluate if a person is a frequent traveler and what could be the nature of their travel and \
                city/country on the basis of any loyalty programs they Might have and nature of their job \
            10.Determine the Grades/Classes of each child on the basis of their age and the city and  country they \
                living in. \
            11.To determine the most likely travel months/weeks for the family, we need to identify the children's \
                school holidays based on their grades/classes and the country/city they live in. \
                This information can be obtained from the school calendar. \
             
    2) The customer Query/questions. \
    
    Do Not Ask Question which Answers can be found from customer chat or customer_data or the from above hypothesis
    e.g Customer age , Age, name of \
    their dependents and any other information provided in customer_data \
    Before providing the final answer,You must greet the user with their First Name from the First Customer 
    and with that you will ask the user 3 clarifying questions One By one In a very polite and welcoming tone
    to gather more information. Question must not be more than 2 lines, Wait for the answer And then Then Ask the other\
     question , SO JUST 1 Question at a time Please\
    First clarifying question:\
    Get further Information on the basis of their Query and try to find more information on what kind of \
    trip they want.\
    Second clarifying question: \
    Get further Information on the basis of their Answer to the First  Question \
    and try find more information on what kind of trip they want. \
    Third clarifying question: \
    Get further Information regarding user on the basis of their Answer to the Second  Question \
    and try to find more information on what kind of trip they want. \
    With every question Give user some Possible Examples of the Answer to that question  \
    Final answer to the user's original question: \
    Suggest user, the best suitable trips as per their query and customer_data \
    
    If there are more than one trip aligns with customer requirements , Suggest all of those trips .\
    In your response only send 2 Json fields about the trip . \
    1) A Three line overview about the trips and \
    explain the customer in detail how the suggested trips best fits with the customer from their chat, customer_data 
    and the hypothesis \
    Also write down some hypothesis data from the customer_data which helped you to suggest this trip. \
    2) the '_id's of the suggested trips. \
    

    {context} {customer_data}"""
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
        {"input": query, "customer_data": customer_data},
        config={
            "configurable": {"session_id": session}
        },
    )["answer"]

    return var


@app.route('/hello')
def hello():
    return "hee"
