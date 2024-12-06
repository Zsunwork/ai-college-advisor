# AI Advisor for College Students

Welcome to the AI Advisor for College Students project! This open-source project provides AI-powered assistance to help college students plan their academic and career futures.

## Features

- **Academic Planning**: Get personalized advice on course selection, major choices, and academic pathways
- **Career Guidance**: Receive insights on career options, industry trends, and professional development
- **Personalized Memory**: The system remembers your background and preferences to provide tailored advice
- **Interactive Conversations**: Natural dialogue with follow-up questions for better understanding

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Zsunwork/ai-college-advisor.git
cd ai-college-advisor
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

1. Create a `.env` file in the root directory
2. Add your OpenAI API key to the file:

```env
OPENAI_API_KEY=your_api_key_here
```

> 📝 **Note**: You can obtain an API key from the [OpenAI Platform](https://platform.openai.com/api-keys)

### 4. Run the Application

Launch the advisor from the src directory:

```bash
python src/main.py
```

## Usage

1. Start the application
2. Type your questions or concerns about academic or career planning
3. The AI will:
   - Analyze your query
   - Access relevant background information
   - Provide personalized advice
   - Ask clarifying questions when needed
4. Type 'exit' to end the conversation

## Project Structure

```
ai-college-advisor/
├── src/
│   ├── main.py
│   ├── modules/
│   │   ├── Agent.py
│   │   ├── Academic.py
│   │   ├── Career.py
│   │   ├── General.py
│   │   └── UpdateMemory.py
│   └── database/
│       ├── user_backgrounds.json
│       └── conversation_history.json
├── requirements.txt
├── .env
└── README.md
```

## Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions:
1. Check the existing issues in the GitHub repository
2. Create a new issue if needed
3. Provide detailed information about your problem

---

Made with ❤️ for college students