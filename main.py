import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
import litellm

# ==========================================
# TERMINAL COLORS (ANSI ESCAPE CODES)
# ==========================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_header(title, color=Colors.HEADER):
    print(f"\n{color}{Colors.BOLD}{'='*50}")
    print(f" {title}")
    print(f"{'='*50}{Colors.RESET}")

# ==========================================
# BUG FIX: Intercept & Clean Groq Payload
# ==========================================
original_completion = litellm.completion

def patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        clean_messages = []
        for msg in kwargs["messages"]:
            clean_msg = dict(msg) 
            clean_msg.pop("cache_breakpoint", None)
            clean_messages.append(clean_msg)
        kwargs["messages"] = clean_messages
    return original_completion(*args, **kwargs)

litellm.completion = patched_completion

# ==========================================
# INITIALIZATION
# ==========================================
load_dotenv()

groq_llm = LLM(
    model="groq/openai/gpt-oss-20b",
    temperature=0.6,
    max_tokens=4000
)

# ==========================================
# AGENT DEFINITIONS
# ==========================================
coordinator = Agent(
    role='Lead Educational Strategist',
    goal='Analyze the student topic "{topic}", determine the lesson scope, and create a structured outline.',
    backstory='You are an expert curriculum designer. If a topic is vague, you narrow it down to fundamental principles.',
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

explainer = Agent(
    role='Subject Matter Expert & Teacher',
    goal='Teach concepts clearly and re-explain confusing parts when asked.',
    backstory='You are a beloved professor known for making the most difficult subjects easy to understand using great analogies. You strictly format all mathematical, physics, and chemistry equations using plain text and standard keyboard symbols (e.g., use "x^2" or "3/2"). NEVER use LaTeX (like \\frac or \\pmatrix) because the terminal cannot render it. You should cover ALL the topics from the lesson plan.',
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

quiz_master = Agent(
    role='Assessment Creator',
    goal='Create a strict 3-question multiple-choice quiz based ONLY on the provided lesson text.',
    backstory='You are a strict but fair examiner. You format your quizzes beautifully and clearly. You must produce questions from the most important portions of the lesson. You strictly format all equations using plain text and standard keyboard symbols. NEVER use LaTeX formatting.',
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

evaluator = Agent(
    role='Feedback Coach & Grader',
    goal='Check the student\'s answers against the quiz, provide a score, and give constructive feedback.',
    backstory='You are a supportive teaching assistant. You help students learn from their mistakes. You take their answers and clarify their mistakes and re-explain the topic in an easy-to-understand way. You strictly format all equations using plain text and standard keyboard symbols. NEVER use LaTeX formatting.',
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

# ==========================================
# TERMINAL WORKFLOW
# ==========================================
def main():
    print_header("🎓 Welcome to Leo: The Multi-Agent AI Tutor", Colors.CYAN)

    while True:
        print(f"{Colors.CYAN}{Colors.BOLD}")
        topic = input("What would you like to learn today? (or type 'quit' to exit)\n> ")
        print(f"{Colors.RESET}", end="")
        
        if topic.lower() in ['quit', 'exit']:
            print(f"\n{Colors.GREEN}Thanks for studying with Leo! Goodbye.{Colors.RESET}")
            break
            
        # ------------------------------------------
        # STEP 1: GENERATE LESSON
        # ------------------------------------------
        print(f"\n{Colors.YELLOW}[Coordinator and Explainer are building your lesson...]{Colors.RESET}\n")
        
        plan_task = Task(
            description=f'Create a brief 3-point lesson plan for the topic: {topic}.',
            expected_output='A 3-point bulleted lesson plan.',
            agent=coordinator
        )
        
        teach_task = Task(
            description='Write a short, engaging lesson covering the exact 3 points from the lesson plan. Use plain-text math formatting.',
            expected_output='A clear educational text with analogies.',
            agent=explainer,
            context=[plan_task]
        )
        
        generation_crew = Crew(
            agents=[coordinator, explainer],
            tasks=[plan_task, teach_task],
            process=Process.sequential,
            memory=False, 
            verbose=False 
        )
        generation_crew.kickoff()
        
        lesson_output = teach_task.output.raw
        
        print_header("📋 LESSON PLAN", Colors.BLUE)
        print(plan_task.output.raw)
        
        print_header("📖 YOUR LESSON", Colors.GREEN)
        print(lesson_output)
        
        # ------------------------------------------
        # STEP 2: CLARIFICATION LOOP
        # ------------------------------------------
        while True:
            print_header("🤔 CLARIFICATION", Colors.YELLOW)
            print(f"{Colors.CYAN}{Colors.BOLD}")
            feedback = input("Did you understand the lesson? (Type 'yes' to take the quiz, or ask a question if you are confused)\n> ")
            print(f"{Colors.RESET}", end="")
            
            if feedback.lower() in ['yes', 'y', 'yep', 'ready', 'yeah']:
                break
            else:
                print(f"\n{Colors.YELLOW}[Explainer is reviewing your question...]{Colors.RESET}\n")
                clarify_task = Task(
                    description=f"""
                    The student read this lesson:
                    {lesson_output}
                    
                    They just said: "{feedback}"
                    
                    RULES:
                    1. If they mention specific topics, re-explain ONLY those topics clearly using simple language.
                    2. If they just say they don't understand but don't mention specifics, ask them directly what problem they are facing so you can help.
                    3. Use plain-text math formatting. NEVER use LaTeX.
                    """,
                    expected_output='A targeted re-explanation or a clarifying question.',
                    agent=explainer
                )
                
                clarify_result = Crew(agents=[explainer], tasks=[clarify_task], verbose=False).kickoff()
                
                print_header("💡 CLARIFICATION", Colors.MAGENTA)
                print(clarify_result.raw)
                
                lesson_output += f"\n\n---\n\n**Clarification regarding '{feedback}':**\n\n{clarify_result.raw}"

        # ------------------------------------------
        # STEP 3: QUIZ
        # ------------------------------------------
        print(f"\n{Colors.YELLOW}[Quiz Master is reading your lesson and generating questions...]{Colors.RESET}\n")
        
        quiz_task = Task(
            description=f'''Read this text:
{lesson_output}

Create EXACTLY 3 multiple choice questions based ON THIS TEXT. 

CRITICAL FORMATTING RULES:
1. You are running in a terminal. NEVER use LaTeX formatting (like \\frac or \\pmatrix).
2. Use standard plain-text for all math (e.g., write "3/2" or "x^2").

YOU MUST FORMAT EVERY QUESTION EXACTLY LIKE THIS:
**Question 1**
[Question text goes here]

A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
''',
            expected_output='Formatted text containing 3 multiple choice questions using plain-text math.',
            agent=quiz_master
        )
        
        quiz_result = Crew(agents=[quiz_master], tasks=[quiz_task], verbose=False).kickoff()
        quiz_output = quiz_result.raw
        
        print_header("📝 YOUR QUIZ", Colors.BLUE)
        print(quiz_output)
        
        # ------------------------------------------
        # STEP 4: EVALUATION
        # ------------------------------------------
        print(f"{Colors.CYAN}{Colors.BOLD}")
        student_answers = input("\nSubmit Your Answers (e.g., 1-A, 2-C, 3-B):\n> ")
        print(f"{Colors.RESET}", end="")
        
        print(f"\n{Colors.YELLOW}[Evaluator is grading...]{Colors.RESET}\n")
        
        eval_task = Task(
            description=f'''Here is the Quiz:
{quiz_output}

Here are the Student's Answers:
{student_answers}

Grade them out of 3. Provide brief feedback for each question.

CRITICAL FORMATTING RULES:
1. You are running in a terminal. NEVER use LaTeX formatting (like \\frac or \\pmatrix).
2. Use standard plain-text for all math (e.g., write "3/2" or "x^2").

YOU MUST FORMAT YOUR EVALUATION EXACTLY LIKE THIS:
**Grade:** [X]/3

**Feedback:**
1. **[Correct/Incorrect]** - [Brief explanation using plain-text math]
2. **[Correct/Incorrect]** - [Brief explanation using plain-text math]
3. **[Correct/Incorrect]** - [Brief explanation using plain-text math]
''',
            expected_output='A grade and feedback summary using plain-text math.',
            agent=evaluator
        )
        
        eval_result = Crew(agents=[evaluator], tasks=[eval_task], verbose=False).kickoff()
        
        print_header("📋 FINAL EVALUATION", Colors.GREEN)
        print(eval_result.raw)
        print(f"\n{Colors.YELLOW}🔄 Restarting the tutor loop...{Colors.RESET}")

if __name__ == "__main__":
    main()