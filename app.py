import streamlit as st
import sys
from llm_scratch import smaug

# Function to simulate task generation based on user input
def generate_tasks(user_input):
    # Placeholder for actual task generation logic
    response = smaug(f"""
{user_input}

Create a newline delimited list of subtasks to accomplish the above goal
    """)
    print(response)
    tasks = response.split("\n")
    st.session_state.tasks = [{"id": idx, "description": x} for idx, x in enumerate(tasks)]

def execute_task(task_id):
    # Placeholder for actual task execution logic
    st.write(f"Task {task_id} executed successfully")

# Function to display tasks
def display_tasks():
    for task in st.session_state.tasks:
        with st.container():
            # Editable text box for task description
            task_description = st.text_input("Task Description", value=task["description"], key=f"description_{task['id']}")
            col1, col2 = st.columns(2)
            with col1:
                # Run button for each task
                if st.button("Run", key=f"run_{task['id']}"):
                    execute_task(task["id"])
            with col2:
                # Cancel button for each task
                if st.button("Cancel", key=f"cancel_{task['id']}"):
                    # Remove the task from the list
                    st.session_state.tasks = [t for t in st.session_state.tasks if t['id'] != task['id']]
                    # Rerun the app to update the UI
                    st.experimental_rerun()

# Main app
def main():
    if st.session_state.get("tasks") is None:
        st.session_state.tasks = []
    
    st.title("Task Generator")

    # Textbox for user to enter a prompt
    user_input = st.text_input("Enter a prompt:", "")

    # Button to submit the prompt
    submit_button = st.button("Submit")

    if submit_button and user_input:
        # Generate tasks based on the user input
        generate_tasks(user_input)

    # Display tasks
    display_tasks()

if __name__ == "__main__":
    main()

