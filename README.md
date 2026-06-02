live link : https://atulpandey5678-job-recomendation--app-yrmrpf.streamlit.app/
Yt code explanation link :

 what i have built : 

 1. The Core Purpose
 
The application takes a user's resume and automatically finds the best matching jobs from a dataset of over 2,200 job listings.
It acts as an intelligent matchmaker between candidate skills and job descriptions.

2. The Technologies Used

Streamlit: Powers the interactive, web-based user interface.
Sentence-Transformers (Hugging Face): Uses the all-MiniLM-L6-v2 AI model to read and "understand" the context of both the resumes and the job descriptions.
Scikit-Learn (Cosine Similarity): The math engine that calculates exactly how close the resume's meaning matches the job description's meaning.
Pandas & NumPy: Handles loading and managing the backend job dataset efficiently.
PyPDF2: Allows the app to natively read and extract text from uploaded PDF resumes.

3. How the Pipeline Works (Step-by-Step)

Data Loading: When the app starts, it reads your job_title_des.csv dataset and converts all the job descriptions into numerical vectors (embeddings) using the AI model.
User Input: A user either pastes their resume text or uploads a PDF/TXT file. The app reads and truncates the text to ensure it isn't too long for the AI to process.
AI Embedding: The app runs the resume text through the same NLP model to convert it into its own numerical vector.
Similarity Matching: It compares the resume vector against every single job vector in your database using Cosine Similarity, which outputs a score between 0.0 and 1.0 (where 1.0 is a perfect match).
Filtering & Display: It filters out jobs below the "Min Similarity" threshold (controlled via the sidebar) and sorts the rest to present the "Top K" best-matching jobs to the user.
In short, you have built a semantic search engine that doesn't just look for exact keyword matches, but actually understands the contextual meaning behind a candidate's 
experience to match them with the right jobs!

