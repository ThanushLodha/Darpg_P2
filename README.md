# Grievance Classification

---

## Description
This project has UI in jsx to take grievance as input and display the output as the category of the department to which the grievance should be send to. We have trained on the given data and fine tuned the Gemini Pro model by Google and facebook/bart-large-mnli (zero shot classifier) for translation, summarization and classification.

## Setup Instructions

### Prerequisites
- Node.js and npm installed
- Python and pip installed

### Frontend (React with Vite)
1. Navigate to the `darpg-main` directory:
    ```
    cd darpg-main
    ```

2. Install dependencies:
    ```
    npm install
    ```

3. Run the development server:
    ```
    npm run dev
    ```

### Backend (Flask)
1. Run the Parents.ipynb file

4. Install Flask and other dependencies:
    ```
    pip install -r requirements.txt
    ```

5. Run the Flask server:
    ```
    python model.py
    ```

### Full Application
To run the full application, you need to run both the frontend and backend servers simultaneously.

Follow the instructions above to start both the frontend and backend servers.
