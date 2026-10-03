import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
import litellm
import re

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

st.set_page_config(page_title="Leo: AI Tutor", page_icon="🎓", layout="centered")

# Initialize Session State
if "step" not in st.session_state:
    st.session_state.step = 1  # 1: Topic, 2: Lesson, 3: Quiz, 4: Evaluation

def reset_app():
    """Clears the session state to start a new lesson (Fix 3)"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

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
    backstory='You are a beloved professor known for making the most difficult subjects easy to understand using great analogies. You strictly format all mathematical, physics, and chemistry equations using standard LaTeX delimiters (use $ for inline math and $$ for display math). And you should cover ALL the topics from the lesson plan. CRITICAL RULE: You must ensure every single LaTeX delimiter is properly closed, otherwise the UI rendering will crash.',
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

quiz_master = Agent(
    role='Assessment Creator',
    goal='Create a strict 3-question multiple-choice quiz based ONLY on the provided lesson text.',
    backstory='You are a strict but fair examiner. You format your quizzes beautifully and clearly. You must produce questions from the most important portions of the lesson. You strictly format all mathematical, physics, and chemistry equations using standard LaTeX delimiters (use $ for inline math and $$ for display math). CRITICAL RULE: You must ensure every single LaTeX delimiter is properly closed, otherwise the UI rendering will crash.',
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

evaluator = Agent(
    role='Feedback Coach & Grader',
    goal='Check the student\'s answers against the quiz, provide a score, and give constructive feedback.',
    backstory='You are a supportive teaching assistant. You help students learn from their mistakes. You take their answers and clarify their mistakes and re-explain the topic in an easy-to-understand way. You strictly format all mathematical, physics, and chemistry equations using standard LaTeX delimiters (use $ for inline math and $$ for display math). CRITICAL RULE: You must ensure every single LaTeX delimiter is properly closed, otherwise the UI rendering will crash. NEVER use \\( or \\) or \\[ or \\].',
    verbose=True,
    allow_delegation=False,
    llm=groq_llm
)

# ==========================================
# UI WORKFLOW
# ==========================================
def fix_math_rendering(text):
    """Safely forces Streamlit to render math without breaking matrix formatting."""
    if not isinstance(text, str):
        return text
        
    # Safely replace inline math
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    
    # Safely replace display math \[ \] ONLY if not preceded by another backslash
    text = re.sub(r"(?<!\\)\\\[", "$$", text)
    text = re.sub(r"(?<!\\)\\\]", "$$", text)
    
    return text
st.title("🎓 Leo: Multi-Agent AI Tutor")

# ------------------------------------------
# STEP 1: TOPIC INPUT
# ------------------------------------------
if st.session_state.step == 1:
    st.write("Enter a topic to get a custom lesson plan, a detailed explanation, and a quiz!")
    topic = st.text_input("What would you like to learn today?", placeholder="e.g., Simple Harmonic Motion")
    
    if st.button("Generate Lesson"):
        if topic:
            with st.spinner("Coordinator and Explainer are building your lesson..."):
                plan_task = Task(
                    description=f'Create a brief 3-point lesson plan for the topic: {topic}.',
                    expected_output='A 3-point bulleted lesson plan.',
                    agent=coordinator
                )
                teach_task = Task(
                    description='Write a short, engaging lesson covering the exact 3 points from the lesson plan.',
                    expected_output='A clear educational text with analogies.',
                    agent=explainer,
                    context=[plan_task]
                )
                
                generation_crew = Crew(
                    agents=[coordinator, explainer],
                    tasks=[plan_task, teach_task],
                    process=Process.sequential,
                    memory=False, 
                    verbose=True
                )
                
                generation_crew.kickoff()
                
                # Save to state and advance to Step 2
                st.session_state.plan_output = plan_task.output.raw
                st.session_state.lesson_output = teach_task.output.raw
                st.session_state.step = 2
                st.rerun()

# ------------------------------------------
# STEP 2: LESSON REVIEW & CLARIFICATION (Fix 2)
# ------------------------------------------
elif st.session_state.step == 2:
    with st.expander("📋 Lesson Plan (Coordinator)", expanded=False):
        st.write(fix_math_rendering(st.session_state.plan_output))
        
    st.markdown("### 📖 Your Lesson")
    st.write(fix_math_rendering(st.session_state.lesson_output))
    
    st.divider()
    st.subheader("Did you understand the lesson?")
    
    # Layout for Clarification vs Proceeding
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Ready to test your knowledge?**")
        if st.button("Yes! Give me the Quiz 🚀", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
            
    with col2:
        st.write("**Need help?**")
        feedback = st.text_input("What is confusing?", placeholder="e.g., I didn't understand the formula...")
        if st.button("Clarify Please", use_container_width=True):
            if feedback:
                with st.spinner("Explainer is reviewing your question..."):
                    clarify_task = Task(
                        description=f"""
                        The student read this lesson:
                        {st.session_state.lesson_output}
                        
                        They just said: "{feedback}"
                        
                        RULES:
                        1. If they mention specific topics, re-explain ONLY those topics clearly using simple language.
                        2. If they just say they don't understand but don't mention specifics, ask them directly what problem they are facing so you can help.
                        """,
                        expected_output='A targeted re-explanation or a clarifying question.',
                        agent=explainer
                    )
                    
                    clarify_crew = Crew(agents=[explainer], tasks=[clarify_task], verbose=True)
                    clarify_result = clarify_crew.kickoff()
                    
                    # Append the clarification to the ongoing lesson state
                    st.session_state.lesson_output += f"\n\n---\n\n**Clarification regarding '{feedback}':**\n\n{clarify_result.raw}"
                    st.rerun()
            else:
                st.warning("Please type what you need help with first!")

# ------------------------------------------
# STEP 3: QUIZ TIME
# ------------------------------------------
elif st.session_state.step == 3:
    # Generate the quiz only once based on the FINAL lesson content (including clarifications)
    if "quiz_output" not in st.session_state:
        with st.spinner("Quiz Master is reading your lesson and generating questions..."):
            quiz_task = Task(
                description=f'''Read this text:
{st.session_state.lesson_output}

Create EXACTLY 3 multiple choice questions based ON THIS TEXT. 

CRITICAL LATEX RULES:
1. ALL matrices and standalone equations MUST be wrapped in double dollar signs. 
   RIGHT: $$ F = \\begin{{pmatrix}} 0 & 1 \\\\ 1 & 0 \\end{{pmatrix}} $$
   WRONG: F = \\begin{{pmatrix}} 0 & 1 \\\\ 1 & 0 \\end{{pmatrix}} $$
2. All inline variables must be wrapped in single dollar signs. 
   RIGHT: Let $F$ be the matrix.

YOU MUST FORMAT EVERY QUESTION EXACTLY LIKE THIS:
**Question 1**
[Question text goes here]

A) [Option 1]
B) [Option 2]
C) [Option 3]
D) [Option 4]
''',
                expected_output='Formatted text containing 3 multiple choice questions.',
                agent=quiz_master
            )
            quiz_crew = Crew(agents=[quiz_master], tasks=[quiz_task], verbose=True)
            quiz_result = quiz_crew.kickoff()
            st.session_state.quiz_output = quiz_result.raw
            st.rerun()
            
    st.markdown("### 📝 Your Quiz")
    st.write(fix_math_rendering(st.session_state.quiz_output))
    
    st.divider()
    st.subheader("Submit Your Answers")
    student_answers = st.text_input("Enter your answers (e.g., 1-A, 2-C, 3-B):")
    
    if st.button("Grade Me!"):
        if student_answers:
            with st.spinner("Evaluator is grading..."):
                eval_task = Task(
                    description=f'''Here is the Quiz:
{st.session_state.quiz_output}

Here are the Student's Answers:
{student_answers}

Grade them out of 3. Provide brief feedback for each question.

CRITICAL LATEX RULES:
1. ALL matrices and standalone equations MUST be wrapped in double dollar signs. 
   RIGHT: $$ F = \\begin{{pmatrix}} 0 & 1 \\\\ 1 & 0 \\end{{pmatrix}} $$
   WRONG: F = \\begin{{pmatrix}} 0 & 1 \\\\ 1 & 0 \\end{{pmatrix}} $$
2. All inline variables must be wrapped in single dollar signs. 

YOU MUST FORMAT YOUR EVALUATION EXACTLY LIKE THIS:
**Grade:** [X]/3

**Feedback:**
1. **[Correct/Incorrect]** - [Brief explanation with properly formatted LaTeX]
2. **[Correct/Incorrect]** - [Brief explanation with properly formatted LaTeX]
3. **[Correct/Incorrect]** - [Brief explanation with properly formatted LaTeX]
''',
                    expected_output='A grade and feedback summary.',
                    agent=evaluator
                )
                eval_crew = Crew(agents=[evaluator], tasks=[eval_task], verbose=True)
                eval_result = eval_crew.kickoff()
                
                st.session_state.eval_output = eval_result.raw
                st.session_state.step = 4
                st.rerun()
        else:
            st.warning("Don't leave it blank!")

# ------------------------------------------
# STEP 4: FINAL GRADE & RESTART (Fix 3)
# ------------------------------------------
elif st.session_state.step == 4:
    st.markdown("### 📋 Final Evaluation")
    st.write(fix_math_rendering(st.session_state.eval_output))
    
    st.divider()
    # Fix 3: Reset button to clear state and return to Step 1
    if st.button("Learn Another Lesson 🔄", use_container_width=True):
        reset_app()