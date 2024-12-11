from pydantic import BaseModel, create_model
from typing import Dict, Any, List
import openai
import json

class AcademicAdvisor:
    
    def __init__(self, client):
        self.client = client

    def generate_prompt(self, input_text: str, summary_conversation: str, conversation_history: str, select_memory_content: Dict[str, str], clarifying_question_count: int) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": f"""
                You are an academic consultant AI, dedicated to assisting users with academic queries. Follow these steps carefully:
    
                **Current Clarifying Question Round**: {clarifying_question_count}
                
                **Inputs to consider**:
                    a. User’s question: {input_text}
                    b. Latest conversation history: {conversation_history}
                    c. Summary of long-run conversation: {summary_conversation}
                    d. User profile: {select_memory_content}
                
                1. **Check if sufficient information is provided**:
                    - Consider the provided information in {select_memory_content}. If any of the keys in this dictionary contain blank values (`None` or empty string ""), this is defined as **insufficient information**. In this case, do **not** provide an answer immediately.
                    - If **fewer than 2 clarifying questions** have already been asked (i.e., {clarifying_question_count} < 2), ask **1 clarifying question** to gather more relevant information. Return clarifying questions in the `answer_to_customer` field and set `clarifying_question=True`.**Skip step 2** and proceed to step 3.
                    - If **2 or more clarifying questions** have already been asked (i.e., {clarifying_question_count} >= 2), proceed to step 2.
    
                2. **Respond to the User's Query**:
                    - If sufficient information has been provided or after **2 rounds of clarifying questions**, provide a concise answer (3 bullet points or fewer, max 100 words).
                    - **Formatting Requirements**: Use numbered bullet points for each item (e.g., "1.", "2.", "3.") and include **bold formatting** for key terms in each bullet (e.g., "1. **Recommendation**: Explanation here.").
                    - Offer specific, actionable advice related to the user's goals, including **recommendations for relevant exams, certifications, or professional pathways** (e.g., CPA for accounting, PMP for project management).
                    - Focus on expert-level advice by mentioning industry-standard qualifications where relevant.
                    - Return the answer in the `answer_to_customer` field and set `clarifying_question=False`.

    
                3. **Summarize the Conversation**:
                    - Summarize the user’s question and your answer (or clarifying questions, if applicable) in one sentence (100 words or fewer).
                    - Return this summary as `update_summarized_conversation`.
    
                4. **Return the `clarifying_question` Feature**:
                    - If clarifying questions are returned, set `clarifying_question=True`.
                    - If an answer is returned, set `clarifying_question=False`.
                """
            }
        ]


    def get_response(self, input_text: str, summary_conversation: str, conversation_history: str, selected_memory: List[str], select_memory_content: Dict[str, str], clarifying_question_count: int):
        # Dynamically create the Pydantic model based on the selected memory
        fields: Dict[str, (Any, ...)] = {name: (str, ...) for name in selected_memory}
        UpdateUserBackgroundModel = create_model('UpdateUserBackgroundModel', **fields)

        class OutputFormat(BaseModel):
            answer_to_customer: str
            update_summarized_conversation: str
            clarifying_question: bool

        output_format_schema = OutputFormat.model_json_schema()

        messages = self.generate_prompt(input_text, summary_conversation, conversation_history, select_memory_content, clarifying_question_count)

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": {
                "name": "Output",
                "description": "Return the answer, updated conversation history, updated background and a bool for whether AI returns clarifying questions",
                "schema": output_format_schema
            }},
        )

        # Extract the content and token usage
        content = json.loads(response.choices[0].message.content)
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        # Return both the content and token usage
        return content, token_usage
