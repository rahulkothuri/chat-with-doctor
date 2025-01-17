# Chat with Doctor

This project is a **Retrieval-Augmented Generation (RAG) application** designed to help users interact with medical case datasets in a conversational manner. The application uses advanced language models to analyze and extract insights from medical clinical summaries and doctor-patient conversations. Users can upload PDF files containing these datasets, ask questions, and receive accurate and contextually relevant responses.

## Features
- Upload multiple PDF files containing medical datasets.
- Automatically processes and indexes the text from uploaded PDFs.
- Conversational interface for asking questions and receiving detailed answers.
- Displays a clean and interactive chat history UI.
- Utilizes FAISS for vector storage and Google Generative AI for embeddings and responses.

## Prerequisites
Make sure you have the following installed on your system:
- Python 3.8 or later
- A Google Gemini API key (required for accessing the Google Generative AI models)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/chat-with-doctor.git
   cd chat-with-doctor
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google Gemini API key:
   - Create a `.env` file in the root of your project directory.
   - Add the following line to the file:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Launch the application using the command above. The app will open in your default web browser.

2. In the sidebar, upload one or more PDF files containing medical case data.

3. Click the **Submit & Process** button to preprocess and index the uploaded documents.

4. Ask your questions in the main input box, and the application will provide contextually accurate responses.

5. The chat history is displayed in a user-friendly format, showing both your questions and the bot's responses.

## File Structure
- `app.py`: Main application script.
- `requirements.txt`: List of Python dependencies.
- `.env`: Environment file to store your API key (not included in the repository).

## Key Libraries Used
- **Streamlit**: For building the web interface.
- **PyPDF2**: For extracting text from PDF files.
- **FAISS**: For vector storage and similarity search.
- **Google Generative AI**: For embeddings and conversational responses.
- **LangChain**: For managing the conversational chain.

## Notes
- Ensure your `.env` file is correctly configured with your Google Gemini API key.
- Avoid uploading sensitive or personal medical data to maintain privacy and security.

## License
This project is licensed under the [MIT License](LICENSE).

---

