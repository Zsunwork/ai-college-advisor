import os
from dotenv import load_dotenv
import json
from datetime import datetime, timezone
import uuid
from typing import Optional

from modules.Agent import Agent
from modules.Career import CareerAdvisor
from modules.Academic import AcademicAdvisor
from modules.General import GeneralAdvisor
from modules.UpdateMemory import UpdateMemory

import openai

# Load environment variables
load_dotenv()

# Constants and configurations
DATABASE_PATH = "database/user_backgrounds.json"
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    raise ValueError(
        "No OpenAI API key found. Please set the OPENAI_API_KEY environment variable. "
        "You can get an API key from https://platform.openai.com/api-keys"
    )

class LocalDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conversation_path = "database/conversation_history.json"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize user background if not exists
        if not os.path.exists(db_path):
            default_data = {
                "default_user": {
                    "general_background": None,
                    "preference": None,
                    "education_background": None,
                    "education_plan": None,
                    "courses_taken": None,
                    "research_experience": None,
                    "career_background": None,
                    "career_plan": None,
                    "summary_plans": None,
                    "summary_advice": None,
                    "milestones_achieved": None
                }
            }
            self._write_data(default_data)
        
        # Initialize conversation history if not exists
        if not os.path.exists(self.conversation_path):
            default_conversations = {
                "default_user": {
                    "conversations": [],
                    "summary_conversation": "",
                    "clarifying_question_count": 0
                }
            }
            self._write_conversation_data(default_conversations)

    def _read_data(self):
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _write_data(self, data):
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_user_background(self, user_id):
        data = self._read_data()
        return data.get(user_id, {})

    def update_user_background(self, user_id, updates):
        data = self._read_data()
        if user_id not in data:
            data[user_id] = {}
        data[user_id].update(updates)
        self._write_data(data)

    def _read_conversation_data(self):
        try:
            with open(self.conversation_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _write_conversation_data(self, data):
        with open(self.conversation_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_conversation_history(self, user_id):
        data = self._read_conversation_data()
        return data.get(user_id, {
            "conversations": [],
            "summary_conversation": "",
            "clarifying_question_count": 0
        })

    def update_conversation_history(self, user_id, conversation_item, summary_conversation, clarifying_question_count):
        data = self._read_conversation_data()
        if user_id not in data:
            data[user_id] = {
                "conversations": [],
                "summary_conversation": "",
                "clarifying_question_count": 0
            }
        data[user_id]["conversations"].append(conversation_item)
        data[user_id]["summary_conversation"] = summary_conversation
        data[user_id]["clarifying_question_count"] = clarifying_question_count
        self._write_conversation_data(data)

def main():
    # Initialize components
    client = openai.Client(api_key=api_key)
    db = LocalDatabase(DATABASE_PATH)
    agent = Agent(client)
    careerAdvisor = CareerAdvisor(client)
    academicAdvisor = AcademicAdvisor(client)
    generalAdvisor = GeneralAdvisor(client)
    update_memory = UpdateMemory(client)

    # Get or create user profile
    user_id = "default_user"
    user_data = db.get_user_background(user_id)
    
    # Load conversation history
    conversation_data = db.get_conversation_history(user_id)
    conversation_history = conversation_data["conversations"]
    summary_conversation = conversation_data["summary_conversation"]
    clarifying_question_count = conversation_data["clarifying_question_count"]

    print("Welcome to AI Consultant! Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            break

        # Format conversation history
        if len(conversation_history) >= 2:
            last_two_items = conversation_history[-2:]
        else:
            last_two_items = conversation_history
        past_response = ";".join([f'{item["from"]}:"{item["text"]}"' for item in last_two_items])

        # Get agent selection
        selection = agent.get_agent_selection(user_input, past_response)
        
        selected_memory = selection["selected_memory"]
        selected_tools = selection["selected_tools"]

        # Extract relevant user data
        extracted_data = {key: user_data.get(key, None) for key in selected_memory}

        # Generate response based on selected tool
        if selected_tools == "career":
            response, _ = careerAdvisor.get_response(user_input, summary_conversation, past_response, 
                                                   selected_memory, extracted_data, clarifying_question_count)
        elif selected_tools == "academic":
            response, _ = academicAdvisor.get_response(user_input, summary_conversation, past_response, 
                                                     selected_memory, extracted_data, clarifying_question_count)
        else:  # general
            response, _ = generalAdvisor.get_response(user_input, summary_conversation, past_response, 
                                                    selected_memory, extracted_data)

        # Update memory
        updated_memory, token_usage = update_memory.get_updated_memory(user_input, selected_memory, extracted_data)
        if updated_memory and isinstance(updated_memory, dict) and 'updated_memory' in updated_memory:
            db.update_user_background(user_id, updated_memory['updated_memory'])

        # Update conversation history in memory and persistent storage
        new_conversation_items = [
            {"text": user_input, "from": "user"},
            {"text": response["answer_to_customer"], "from": "assistant"}
        ]
        conversation_history.extend(new_conversation_items)
        
        for item in new_conversation_items:
            db.update_conversation_history(
                user_id,
                item,
                response.get("update_summarized_conversation", summary_conversation),
                clarifying_question_count
            )

        # Print response
        print("\nAssistant:", response["answer_to_customer"])

        # Update clarifying question count
        if response.get('clarifying_question', False):
            clarifying_question_count += 1
        else:
            clarifying_question_count = 0

if __name__ == '__main__':
    main()
