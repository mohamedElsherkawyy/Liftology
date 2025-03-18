import json
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
import warnings
import re

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

Ask the users about their name, age, height, weight and calculate the BMI based on BMI predict BMI_case ,fitness goal , fitness level,  and save it in user_info.

use this as a reference :

Chatbot: "Hi there! What is your main fitness goal? (e.g., weight loss, muscle gain, endurance)"
User: "Muscle gain"

Chatbot: "Great! What is your current fitness level? (e.g., beginner, intermediate, advanced)"
User: "Intermediate"

### Instructions
- when he answers all the questions make to him a reasonable customized Exercise plan with a nutrition tip based on his BMI, fitness goal , fitness level.
- use this object as a reference to fill the keys {object}.
- this reference {object} you can adjust the days name and you can make some of the days rest day.

Text : {text}
in Style : {style}
"""

conversation_prompt_template = ChatPromptTemplate.from_template(system_message)


def create_or_update_json(user_info, exercise_plan):
    # Combine user_info and exercise_plan into one dictionary
    combined_data = {
        "user_info": user_info,
        "exercise_plan": exercise_plan
    }
    
    # Write the combined data to the JSON file
    with open("user_history.json", "w") as json_file:
        json.dump(combined_data, json_file, indent=4)
# Function to create or update the text file

def create_or_update_txt(user_input, assistant_response):       
    with open("user_conversation.txt", "r+") as f:
        lines = f.readlines()
        
        last_non_empty_line_index = None
        for i in range(len(lines)-1, -1, -1):
            if lines[i].strip():  
                last_non_empty_line_index = i
                break

        if last_non_empty_line_index is not None:
            f.seek(0)
            f.writelines(lines[:last_non_empty_line_index+1])
            f.write("\n")  
        else:
            f.seek(0)

        f.write(f"User: {user_input}\n")
        f.write(f"Assistant: {assistant_response}\n\n")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("user_input","")
        
        if not user_input:
            return jsonify({"error": "No user input provided"}), 400

        user_messages = conversation_prompt_template.format_messages(style=style, text=user_input , object = USER_INFORAMTION)

        response = conversation.run(input=user_messages[0].content)
        if isinstance(response, str):

            cleaned_response = re.sub(r'json|', '', response).strip()
            response = json.loads(cleaned_response)
        
        user_info = response.get("user_info", "") 
        exercise_plan = response.get("exercise_plan","")
        

        create_or_update_txt(user_input, response)  
        create_or_update_json(user_info,exercise_plan) 

        return jsonify({"assistant_response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
