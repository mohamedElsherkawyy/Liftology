import json
import os
import warnings
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, ValidationError
from typing import List, Union

from config import USER_INFORAMTION

warnings.filterwarnings('ignore')
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("API key not found in environment variables")

app = Flask(__name__)
memory = ConversationBufferMemory()

chat = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key=groq_api_key
)

conversation = ConversationChain(llm=chat, memory=memory)

style = """polite tone that speaks in English, 
keep the questions direct and concise, asking only for the required details without adding unnecessary conversation."""

# --- Pydantic Models ---

class UserInfo(BaseModel):
    name: str
    age: Union[str, int]
    height: Union[str, int, float]
    weight: Union[str, int, float]
    BMI: Union[str, int, float]
    BMI_case: str
    fitness_goal: str
    fitness_level: str

class ExerciseDay(BaseModel):
    day: str
    exercise: str
    sets: str
    reps: str
    weight: str

class NutritionTip(BaseModel):
    nutrition_tip: str

class ResponseModel(BaseModel):
    message: str
    user_info: UserInfo
    exercise_plan: List[Union[ExerciseDay, NutritionTip]]

# --- Prompt Template ---
system_message = """
You are a helpful fitness assistant. 
Ask about each section separately.

Move to the next section only when the user has provided all the required details in the current section.
Make sure the response is a valid JSON object.

Ask the users about their name, age, height, weight and calculate the BMI, then predict BMI_case.
Ask the users about fitness goal and fitness level and save them in user_info.

Use this flow to build a personalized plan based on the following user input:
user input : {text}
style : {style}
use this instructions to format your response:
{format_instructions}
"""

conversation_prompt_template = ChatPromptTemplate.from_template(system_message)
response_parser = PydanticOutputParser(pydantic_object=ResponseModel)

# --- Helpers ---
def create_or_update_json(user_info, exercise_plan):
    combined_data = {
        "user_info": user_info,
        "exercise_plan": exercise_plan
    }
    with open("user_history.json", "w") as json_file:
        json.dump(combined_data, json_file, indent=4)

def create_or_update_txt(user_input, assistant_response):
    with open("user_conversation.txt", "a+") as f:
        f.write(f"User: {user_input}\n")
        f.write(f"Assistant: {assistant_response}\n\n")

# --- Flask Route ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("user_input", "")
        if not user_input:
            return jsonify({"error": "No user input provided"}), 400

        format_instructions = response_parser.get_format_instructions()
        user_messages = conversation_prompt_template.format_messages(
            style=style,
            text=user_input,
            format_instructions=format_instructions
        )

        response = conversation.run(input=user_messages[0].content)

        # Clean up the JSON response
        cleaned_response = re.sub(r'```(?:json)?\n?', '', response).strip()
        cleaned_response = cleaned_response.rstrip("```")

        response_json = json.loads(cleaned_response)

        # Validate with Pydantic
        parsed_response = ResponseModel.parse_obj(response_json)

        user_info = parsed_response.user_info.dict()
        exercise_plan = [item.dict() for item in parsed_response.exercise_plan]

        create_or_update_txt(user_input, cleaned_response)
        create_or_update_json(user_info, exercise_plan)

        return jsonify({"assistant_response": parsed_response.dict()})

    except ValidationError as ve:
        return jsonify({"error": "Response validation failed", "details": ve.errors()}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
