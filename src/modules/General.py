from pydantic import BaseModel, create_model
from typing import Dict, Any, List
import openai
import json

class GeneralAdvisor:

    def __init__(self, client):
        self.client = client

    def generate_prompt(self, input_text: str, summary_conversation: str, conversation_history: str,
                        select_memory_content: Dict[str, str]) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": f"""
                You are a general-purpose AI named Stairway AI, dedicated to answering questions **not related to academia or career advice**, but gently steering users towards those topics. Your role involves the following tasks:

                1. **Respond to User's Input**:
                    - If the input includes a greeting (e.g., "Hello", "How are you?"), respond with a friendly and polite greeting of your own, then invite them to ask questions about academia or career.
                    - If the input includes a question unrelated to academia or career, provide a brief, helpful answer (within 100 words), using a neutral and friendly tone.
                    - Add this response to 'answer_to_customer' field.

                2. **Encourage Academia or Career-Related Questions**:
                    - At the end of each response, regardless of the input, include a polite, friendly invitation encouraging questions about academia or career advice. (e.g., "If you have any questions related to academia or career advice, feel free to ask!").

                **Inputs to consider**:
                    a. User’s input: "{input_text}"
                    b. Conversation history: "{conversation_history}"
                """
            }
        ]

    def get_response(self, input_text: str, summary_conversation: str, conversation_history: str, selected_memory: List[str], select_memory_content: Dict[str, str]):
        # Dynamically create the Pydantic model based on the selected memory
        fields: Dict[str, (Any, ...)] = {name: (str, ...) for name in selected_memory}
        UpdateUserBackgroundModel = create_model('UpdateUserBackgroundModel', **fields)

        class OutputFormat(BaseModel):
            answer_to_customer: str
            update_summarized_conversation: str

        output_format_schema = OutputFormat.model_json_schema()

        messages = self.generate_prompt(input_text, summary_conversation,conversation_history, select_memory_content)

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": {
                "name": "Output",
                "description": "Return the answer, updated conversation history",
                "schema": output_format_schema
            }},
        )

        content = json.loads(response.choices[0].message.content)
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        # Return both the content and token usage
        return content, token_usage
