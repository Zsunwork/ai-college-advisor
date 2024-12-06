from pydantic import BaseModel, create_model
from typing import Dict, Any, List
import openai
import json

class UpdateMemory:


    def __init__(self, client):
        self.client = client

    def generate_prompt(self, input_text: str,memory_item:Dict[str,str],memory_descriptions:Dict[str,str]) -> list[dict]:
        return [
            {
                "role": "system",
                "content": f"""
                
                Your task is to update the user's memory item based on the latest input from users.

                The memory item needed to be updated with description:
                {memory_descriptions}
                And the memory item with content is:
                {memory_item}
                """
            },
            {
                "role": "user",
                "content": f"User's input message: {input_text}"
            }
        ]

    def get_updated_memory(self, input_text: str, selected_memory: List[str], memory_item: Dict[str, str]):
        if not selected_memory:  # If selected_memory is empty
            return {"updated_memory": {}}, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        
        memory_items = {
            "general_background": "General Information about users, excluding education, and working experience",
            "preference": "The user's preference between academic and career (pursuing a higher degree or finding a job)",
            "education_background": "User's educational background including current and past college names, majors, degrees, current year of study, GPAs",
            "education_plan": "User's plan about future degree, major and research area",
            "courses_taken": "The user's current finished courses",
            "research_experience": "The user's past and current research experience",
            "career_background": "User's career background including past position and company, working content, level",
            "career_plan": "User's plan about future career path, industry of interest, timeline",
            "summary_plans": "Summary of schedules and plans made by users and AI together",
            "summary_advice": "Summary of provided advice to user",
            "milestones_achieved": "Previous milestones the user has achieved"
        }
        
        memory_descriptions = {item: memory_items[item] for item in selected_memory if item in memory_items}
        
        if not memory_descriptions:  # If no valid memory items found
            return {"updated_memory": {}}, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
       
        messages = self.generate_prompt(input_text, memory_item, memory_descriptions)

        fields: Dict[str, (Any, ...)] = {name: (str, ...) for name in selected_memory}
        UpdateUserBackgroundModel = create_model('UpdateUserBackgroundModel', **fields)
        
        class OutputFormat(BaseModel):
            updated_memory: UpdateUserBackgroundModel
         
        output_format_schema = OutputFormat.model_json_schema()
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "Memory_Update",
                    "description": "Return the updated memory content based on user input",
                    "schema": output_format_schema
                }},
            )
            
            content = json.loads(response.choices[0].message.content)
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return content, token_usage
        except Exception as e:
            print(f"Error in memory update: {e}")
            return {"updated_memory": {}}, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
       
