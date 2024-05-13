import os

from flask import Flask, g, request, jsonify

from flask import Flask

app = Flask(__name__)
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


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
    messages = [
        {"role": "system", "content": """

  you are an helpful ,experienced traveler chatbot ,
  based on the user query ask Question to the user following their query, Once you get the answer of that question, 
  Ask another question to further narrow down the requirements,
  After you get the answer for the second question, Ask a last question.
  Make sure you do not Ask for the information which you already have from previous user's answers or in 'customer_data'
  
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
