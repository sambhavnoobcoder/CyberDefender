import os
import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False


def main():
    st.title("Chatbot")

    # Initialize the chat history
    chat_history = []

    # Load or create the model index
    if PERSIST and os.path.exists("persist"):
        st.write("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("data/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    # Create the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    user_input = st.text_input("User Input:", "")

    if user_input:
        if user_input.lower() in ['quit', 'q', 'exit']:
            st.stop()
        result = chain({"question": user_input, "chat_history": chat_history})
        st.write("Chatbot:", result['answer'])
        chat_history.append((user_input, result['answer']))


if __name__ == "__main__":
    main()
