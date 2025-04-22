
import json
import os
import warnings
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel
from typing import List

from config import USER_INFORAMTION

warnings.filterwarnings('ignore')
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("API key not found in environment variables")

app = Flask(__name__)

# Define Pydantic Models
class Exercise(BaseModel):
    day: str
    exercise: str
    sets: str
    reps: str
    weight: str

class UserInfo(BaseModel):
    name: str
    age: int
    height: float
    weight: float
    BMI: float
    BMI_case: str
    fitness_goal: str
    fitness_level: str

class FitnessResponse(BaseModel):
    message: str
    user_info: UserInfo
    exercise_plan: List[Exercise]

# Setup LangChain
parser = PydanticOutputParser(pydantic_object=FitnessResponse)

style = """polite tone that speaks in English , 
keep the questions direct and concise, asking only for the required details without adding unnecessary conversation."""
system_message = """
You are a helpful fitness assistant.

Move to the next section only when the user has provided all the required details in the current section.

Make sure the response is one big JSON object containing three main keys: 
1. "message" (a summary)
2. "user_info" (with name, age, height, weight, BMI, BMI_case, fitness_goal, and fitness_level)
3. "exercise_plan" (a list of structured workouts)

Use this format: 
{format_instructions}

Use this flow to build a personalized plan based on the following user input:
User input: {text}
In Style: {style}
"""

prompt = PromptTemplate.from_template(
    template=system_message,
    partial_variables={"format_instructions": parser.get_format_instructions(), "style": style}
)

chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    api_key=groq_api_key
)

chain = LLMChain(llm=chat, prompt=prompt)

# Save JSON
def create_or_update_json(user_info, exercise_plan):
    combined_data = {
        "user_info": user_info,
        "exercise_plan": exercise_plan
    }
    with open("user_history.json", "w") as json_file:
        json.dump(combined_data, json_file, indent=4)

# Save TXT
def create_or_update_txt(user_input, assistant_response):       
    with open("user_conversation.txt", "a") as f:
        f.write(f"User: {user_input}\n")
        f.write(f"Assistant: {assistant_response}\n\n")

@app.route('/chat', methods=['POST'])
def chat_api():
    try:
        user_input = request.json.get("user_input", "")
        if not user_input:
            return jsonify({"error": "No user input provided"}), 400

        raw_response = chain.run(text=user_input)
        cleaned_response = re.sub(r'```(?:json)?\n?', '', raw_response).strip()

        parsed_response = parser.parse(cleaned_response)

        create_or_update_txt(user_input, cleaned_response)
        create_or_update_json(parsed_response.user_info.dict(), [e.dict() for e in parsed_response.exercise_plan])

        return jsonify({"assistant_response": parsed_response.dict()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
