import json
import os
import random
import requests
import re
import warnings
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

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

conversation = ConversationChain(
    llm=chat,
    memory=memory
)

style = """polite tone that speaks in English , 
keep the questions direct and concise, asking only for the required details without adding unnecessary conversation."""

system_message = """
You are a helpful fitness assistant. 
Ask about each section separately.

Move to the next section only when the user has provided all the required details in the current section.
make sure the response is one big JSON object contains three main keys: the first key is the "message" and the second key is the "user_info" to extract user_info and the third key is "exercise_plan" to store in it the exercises.

Ask the users about their name, age, height, weight and calculate the BMI based on BMI predict BMI_case.
Ask the users about fitness goal , fitness level, and save it in user_info.

use this as a reference :

Chatbot: "Hi there! What is your main fitness goal? (e.g., weight loss, muscle gain, endurance)"
User: "Muscle gain"

Chatbot: "Great! What is your current fitness level? (e.g., beginner, intermediate, advanced)"
User: "Intermediate"

### Instructions
- when he answers all the questions make to him a reasonable customized Exercise plan with a nutrition tip based on his BMI, fitness goal , fitness level.
- use this object as a reference to fill the keys {object}.
- this reference {object} you can adjust the days name and you can make some of the days rest day.

Use this flow to build a personalized plan based on the following user input:
user input : {text}
in Style : {style}
"""

conversation_prompt_template = ChatPromptTemplate.from_template(system_message)

# -------- API Utilities --------

def fetch_exercises_from_api(category_name=None, limit=5):
    url = "https://wger.de/api/v2/exerciseinfo/?language=2&limit=100"
    response = requests.get(url)
    data = response.json()["results"]

    if category_name:
        data = [ex for ex in data if ex['category']['name'].lower() == category_name.lower()]
    return random.sample(data, min(limit, len(data)))

def generate_nutrition_tip(bmi_case):
    tips = {
        "underweight": "Include more healthy calories: nuts, avocados, whole milk, and strength training to build muscle.",
        "normal": "Maintain a balanced diet with fruits, vegetables, lean protein, and regular exercise.",
        "overweight": "Reduce sugary and high-fat foods, increase fiber and water intake, and do more cardio.",
        "obese": "Consult a specialist, cut refined carbs, eat more vegetables and lean protein, and increase daily movement."
    }
    return tips.get(bmi_case.lower(), "Eat a balanced diet and stay active.")

# -------- File Writers --------

def create_or_update_json(user_info, exercise_plan):
    with open("user_history.json", "w") as json_file:
        json.dump({
            "user_info": user_info,
            "exercise_plan": exercise_plan
        }, json_file, indent=4)

def create_or_update_txt(user_input, assistant_response):       
    with open("user_conversation.txt", "a") as f:
        f.write(f"User: {user_input}\n")
        f.write(f"Assistant: {json.dumps(assistant_response, indent=4)}\n\n")

# -------- Chat Endpoint --------

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        user_input = request.json.get("user_input", "")
        if not user_input:
            return jsonify({"error": "No user input provided"}), 400

        user_messages = conversation_prompt_template.format_messages(style=style, text=user_input, object=USER_INFORAMTION)
        response = conversation.run(input=user_messages[0].content)

        if isinstance(response, str):
            cleaned_response = re.sub(r'json|```|\\n', '', response).strip()
            response = json.loads(cleaned_response)

        user_info = response.get("user_info", {})
        exercise_plan = response.get("exercise_plan", {})

        # ✨ Enhance plan with real API data
        goal = user_info.get("fitness_goal", "")
        bmi_case = user_info.get("BMI_case", "")
        category = "Strength" if "muscle" in goal.lower() else "Cardio"
        api_exercises = fetch_exercises_from_api(category_name=category, limit=10)

        for day in exercise_plan:
            chosen = random.choice(api_exercises)
            exercise_plan[day]["exercise"] = chosen["name"]
            exercise_plan[day]["description"] = re.sub("<[^<]+?>", "", chosen["description"]) or "General workout"

        # Add nutrition tip
        exercise_plan["Nutrition Tip"] = generate_nutrition_tip(bmi_case)

        # Save to files
        create_or_update_txt(user_input, {
            "user_info": user_info,
            "exercise_plan": exercise_plan
        })
        create_or_update_json(user_info, exercise_plan)

        return jsonify({
            "assistant_response": {
                "user_info": user_info,
                "exercise_plan": exercise_plan
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------- Run App --------

if __name__ == '__main__':
    app.run(debug=True)