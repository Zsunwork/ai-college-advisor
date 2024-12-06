from pydantic import BaseModel
import openai
import json

class Agent:
    class OutputFormat(BaseModel):
        selected_tools: str
        selected_memory: list[str]

    def __init__(self, client):
        self.client = client

    def generate_prompt(self, input_text: str, past_response: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": """
                Your task is to select the most appropriate tool and relevant memory items based on the user's input and their interaction history. **Select only one tool** and choose the most relevant memory items from the list. 
    
                **Tools**:
                - academic: Use this when the user's query involves academic consulting, such as advice on degree programs, research, or coursework.
                - career: Use this when the user's query involves professional or career-related consulting, such as job advice, career planning, or industry-specific guidance.
                - general: Use this when the user's query does not directly relate to academic or career advice and requires a general response.
    
                **Memory Items** (choose relevant items):
                - general_background: General information about the user, excluding education and work experience.
                - preference: The user's preference between academia and career (e.g., pursuing a higher degree or finding a job).
                - education_background: User’s educational background including current and past college names, majors, degrees, current year of study, GPAs.
                - education_plan: User’s plan for future degree, major, or research area.
                - courses_taken: Courses the user has completed.
                - research_experience: User’s past and current research experience.
                - career_background: User’s career background, including past positions, companies, job responsibilities, and job level.
                - career_plan: User’s plan for their future career path, industry of interest, and timeline.
                - summary_plans: Summary of schedules and plans made by the user and AI together.
                - summary_advice: Summary of advice previously provided to the user.
                - milestones_achieved: Significant milestones the user has reached.
    
                **Instructions**:
                1. Analyze the user's input and past responses.
                2. Select the most suitable tool based on the type of query.
                3. Choose **3** memory items that are most relevant to the context of the query and past interactions.
                4. Ensure the selected tool is the best fit for the user's needs and that the memory items provide relevant context for further interaction.
                """
            },
            {
                "role": "user",
                "content": f"User's input message: {input_text} | Past AI response: {past_response}"
            }
        ]

    def get_agent_selection(self, input_text: str, past_response: str):
        messages = self.generate_prompt(input_text, past_response)
        output_format_schema = self.OutputFormat.model_json_schema()
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": {
                "name": "Agent_Selection",
                "description": "Return the selected tools and selected memory by Agent",
                "schema": output_format_schema
            }}
        )

        content = response.choices[0].message.content
        parsed_content = json.loads(content)
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
        
        memory_descriptions = {item: memory_items[item] for item in parsed_content['selected_memory'] if item in memory_items}
        
        # Add token usage to the return value
        return {
            "selected_tools": parsed_content['selected_tools'],
            "selected_memory": parsed_content['selected_memory'],
            "memory_descriptions": memory_descriptions,
            "token_usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
