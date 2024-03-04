from dotenv import load_dotenv
import streamlit as st
import os
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI 

# Load environment variables
load_dotenv()

# Read data
data = pd.read_csv('cbo.csv')

# Initialize OpenAI LLM
llm = ChatOpenAI(openai_api_key=os.getenv("OPEN_API_KEY"), temperature=0.5)

# Create agent executer
agent_executer = create_csv_agent(llm, 'cbo.csv', verbose=True)

# Initialize session state for chat history if it doesn't exist
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

# Display Jivika logo in the sidebar
def display_sidebar():
    with st.sidebar:
        st.image("jivikanew.jpg")

# Function to process user input and generate response
def process_input(input_text, data, agent_executer):
    if input_text:
        st.session_state['chat_history'].append(("You", input_text))
        # Check if the question is data-related
        if any(word in input_text.lower() for word in data.columns):
            # If data-related, invoke the agent with the data
            response = agent_executer.invoke(input_text)
        else:
            # If not data-related, invoke the agent without the data
            response = agent_executer.invoke(input_text, use_data=False)

        return response

# Function to display chat history and current interaction
def display_chat():
    st.subheader("The Chat History is")
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")

# Function to display input box at the bottom
def display_input_box():
    input_text = st.text_input("Input and press Enter to ask the question:", key="input_text")
    return input_text

# Main function to run the app
def main():
    initialize_session_state()
    display_sidebar()
    display_chat()  # Display chat history above the top
    input_text = display_input_box()  # Display input box at the bottom
    if input_text:
        response = process_input(input_text, data, agent_executer)
        if response:
            st.subheader("Your Question is:")
            st.write(input_text)
            st.subheader("The Response is:")
            if 'output' in response and response['output'] != 'N/A':
                st.write(response['output'])
            else:
                st.write("I'm sorry, I don't have an answer for that question.")

if __name__ == "__main__":
    main()
