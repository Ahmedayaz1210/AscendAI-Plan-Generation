from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from openai import OpenAI,OpenAIError
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class QuizAnswer(BaseModel):
    id: int
    question: str
    answer: Union[str, List[str]]

class QuizSubmission(BaseModel):
    answers: List[QuizAnswer]

@app.post("/quiz-answers-simple")
async def process_quiz_answers_simple(submission: QuizSubmission):
    try:
        # Format quiz answers for the LLM
        formatted_answers = "\n".join([f"Q{answer.id}: {answer.question}\nA: {answer.answer}" for answer in submission.answers])
        
        # Extract key information from specific questions
        habits_to_implement = next((answer.answer for answer in submission.answers if answer.id == 11), "N/A")
        habits_to_remove = next((answer.answer for answer in submission.answers if answer.id == 12), "N/A")
        ultimate_goal = next((answer.answer for answer in submission.answers if answer.id == 13), "N/A")
        
        # Generate the habit plan without expanded understanding
        habit_plan = generate_habit_plan_simple(formatted_answers, habits_to_implement, habits_to_remove, ultimate_goal)
        
        return {
            "habit_plan": habit_plan
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

def load_chroma_db():
    persist_directory = "chroma_db3"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectordb


vectordb = load_chroma_db()

@app.get("/test-chroma")
def test_chroma():
    try:
        # Perform a simple query to test the connection
        results = vectordb.similarity_search("What are the four laws of behavior change?", k=1)
        if results:
            return {"message": "ChromaDB connection successful", "sample_result": results[0].page_content}
        else:
            return {"message": "ChromaDB connection successful, but no results found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB error: {str(e)}")
    

def expand_user_goals_with_llm(habits_to_implement: str, habits_to_remove: str, ultimate_goal: str) -> str:
    prompt = """
    You are an AI assistant specializing in habit formation and personal development, with extensive knowledge of James Clear's "Atomic Habits". Your task is to expand on the user's stated habits and goals to provide a more comprehensive understanding for a habit-tracking application.

    Context:
    - The application helps users build better habits based on principles from "Atomic Habits".
    - Key concepts include the four laws of behavior change: make it obvious, attractive, easy, and satisfying.
    - Other important ideas: habit stacking, environment design, identity-based habits, and the two-minute rule.

    Your task:
    1. Analyze the given habits to implement, habits to remove, and ultimate goal.
    2. Expand on these by including:
       - Related habits or sub-habits that might support the main goals
       - Relevant concepts from "Atomic Habits" that apply to these specific habits and goals
       - Potential challenges or obstacles related to these habits
       - Measurable outcomes or benefits of achieving these habits and goals

    3. Ensure the expanded understanding covers multiple aspects of habit formation, including:
       - Habit implementation (cue, craving, response, reward)
       - Habit tracking and measurement
       - Overcoming obstacles and maintaining consistency
       - Long-term behavior change and identity shifts

    4. Format the expanded understanding as a series of related points or phrases, separated by semicolons.

    5. Keep the expansion focused and relevant to the user's stated intentions.

    Habits to implement: {habits_to_implement}
    Habits to remove: {habits_to_remove}
    Ultimate goal: {ultimate_goal}

    Expanded understanding:
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Expand on the following habits and goal:\n\nHabits to implement: {habits_to_implement}\nHabits to remove: {habits_to_remove}\nUltimate goal: {ultimate_goal}"}
            ],
            max_tokens=200,
            n=1,
            temperature=0.7,
        )
        expanded_understanding = completion.choices[0].message.content.strip()
        return expanded_understanding
    except Exception as e:
        print(f"Error in expanding user goals: {str(e)}")
        return f"Habits to implement: {habits_to_implement}; Habits to remove: {habits_to_remove}; Ultimate goal: {ultimate_goal}" 
    

def generate_habit_plan_simple(quiz_answers: str, habits_to_implement: str, habits_to_remove: str, ultimate_goal: str) -> str:
    prompt = f"""
    You are an expert habit formation coach and behavioral psychologist for Ascend AI, a cutting-edge habit-tracking application based on the principles from James Clear's "Atomic Habits". Your role is to provide insightful, practical, and motivating guidance to users seeking to build lasting habits and achieve personal growth.

    Given the user's quiz answers, habits they want to implement and remove, and their ultimate goal, your task is to synthesize this information into a comprehensive, coherent, and actionable habit plan. Your response should be in two parts:

    Part 1: Detailed Narrative Response
    1. Brief introduction and overview of the habit-building process
    2. Detailed habit plan with at least 7 actionable steps, each including:
    - Clear action to take
    - Specific implementation details (when, where, how)
    - How to use Ascend AI features to support this step
    - Potential obstacles and how to overcome them
    3. Conclusion with words of encouragement and reminder of long-term benefits

    Ensure that this part:
    - Directly addresses the user's habits to implement, habits to remove, and ultimate goal
    - Incorporates key insights from "Atomic Habits", including the four laws of behavior change (make it obvious, attractive, easy, and satisfying)
    - Provides specific, measurable actions for each step of the plan
    - Offers strategies to overcome potential obstacles and maintain consistency
    - Suggests how Ascend AI's features (like habit tracking, reminders, or progress visualization) can support each step of the plan
    - Provides encouragement and motivation to help users stay committed to their habit-building journey
    - Tailors the advice based on the user's quiz answers

    Part 2: JSON Summary
    After the narrative response, provide a JSON summary of the plan with the following structure:

    {{
    "overview": "Brief overview of the habit-building process (max 150 words)",
    "habitPlan": [
        {{
        "step": "Clear action to take",
        "implementation": "Specific details on when, where, and how to perform the action",
        "ascendAISupport": "How to use Ascend AI features to support this step",
        "potentialObstacles": ["Obstacle 1", "Obstacle 2"],
        "overcomingObstacles": ["Strategy for Obstacle 1", "Strategy for Obstacle 2"]
        }}
        // Repeat for each step (5-7 steps total)
    ],
    "motivation": "Brief motivational message (max 50 words)",
    "longTermBenefits": ["Benefit 1", "Benefit 2", "Benefit 3"]
    }}

    User's Quiz Answers:
    {quiz_answers}

    Habits to Implement:
    {habits_to_implement}

    Habits to Remove:
    {habits_to_remove}

    Ultimate Goal:
    {ultimate_goal}

    Please provide both the detailed narrative response and the JSON summary, ensuring they are well-structured, engaging, and highly detailed. Tailor the plan to the user's specific needs, habits, and goals based on their quiz answers and the principles of "Atomic Habits".
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Please generate a comprehensive habit plan based on the given information."}
            ],
            max_tokens=1000,
            n=1,
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)