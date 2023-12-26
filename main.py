import os
from key import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key



st.title('BOOK_GPT')
input_text=st.text_input("Search the book u want")


first_input=PromptTemplate(
    input_variables=['name'],
    template="Tell me about the  Book {name}"
)


Book_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
author_memory = ConversationBufferMemory(input_key='Book', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='author', memory_key='description_history')

llm=OpenAI(temperature=0.8)
chain_1=LLMChain(
    llm=llm,prompt=first_input,verbose=True,output_key='Book',memory=Book_memory)


second_input=PromptTemplate(
    input_variables=['Book'],
    template="who is the author of this {Book}?"
)

chain_2=LLMChain(
    llm=llm,prompt=second_input,verbose=True,output_key='author',memory=author_memory)


third_input_prompt=PromptTemplate(
    input_variables=['author'],
    template="Mention 5 most famous books written by this {author}."
)
chain_3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)
Total_chain=SequentialChain(
    chains=[chain_1,chain_2,chain_3],input_variables=['name'],output_variables=['name','Book','author'],verbose=True)



if input_text:
    st.write(Total_chain({'name':input_text}))

    with st.expander('Author Name'): 
        st.info(Book_memory.buffer)

    with st.expander('Major Books Published'): 
        st.info(descr_memory.buffer)