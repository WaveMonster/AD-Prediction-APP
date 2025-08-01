Alzheimer's Disease (AD) Prediction Web Application
Overview

This is a Flask-based web application designed to provide a user-friendly graphical interface for predicting the probability of Alzheimer's Disease (AD). The application integrates a pre-trained machine learning model and supports both manual input of feature data and batch uploading of datasets in CSV or Excel format for predictions.
Features

    Graphical User Interface (GUI): Provides an intuitive web interface for easy user interaction.

    Manual Data Input: Users can input feature data one by one through a form for single-sample predictions. Each input field includes detailed descriptions and examples.

    Dataset Upload: Supports uploading .csv or .xlsx files for batch predictions on multiple samples.

    Real-time Prediction Results Display: After prediction, the diagnostic probability for each sample is displayed directly on the webpage.

    Model Integration: The backend seamlessly integrates pre-trained Scikit-learn models (AD_Prediction.joblib) and a data imputer (imputer.joblib).

    Responsive Design: Built with Bootstrap framework, the interface displays well across various devices (desktop, tablet, mobile).

Technology Stack

    Backend:

        Python: Primary programming language.

        Flask: Lightweight web framework.

        Pandas: Data manipulation and analysis library.

        NumPy: Numerical computing library.

        Scikit-learn: Machine learning library (for loading models).

        Joblib: For saving and loading Python objects (models and imputer).

        Gunicorn: Production-ready WSGI HTTP server (for deployment).

        Openpyxl: For handling Excel files.

    Frontend:

        HTML5: Page structure.

        CSS3: Page styling.

        JavaScript: Client-side interactivity (currently mainly for form submission, extensible).

        Bootstrap 5: Responsive front-end framework.

    Deployment/Containerization (Optional):

        Docker: Containerization platform, providing environment consistency.

        Nginx: Web server/reverse proxy (for production deployment).

        Systemd: Linux service management tool (for production deployment).

Project Structure

ADPredictionWeb/
├── app.py                     # Main Flask application file, containing backend logic and routes
├── static/
│   ├── css/
│   │   └── style.css          # Custom CSS styles
│   └── js/
│       └── script.js          # (Optional) Frontend JavaScript file
├── templates/
│   └── index.html             # Main HTML template, including data input and results display
├── model/
│   ├── AD_Prediction.joblib   # Pre-trained classifier model file
│   └── imputer.joblib         # Pre-trained data imputer file
├── requirements.txt           # Python dependencies list
├── Procfile                   # (For Heroku deployment) Tells Heroku how to start the app
├── Dockerfile                 # (For Docker deployment) Docker image build file
├── .dockerignore              # (For Docker deployment) Files to ignore during Docker build
└── README.md                  # Project README file

Local Setup and Running
1. Prerequisites

    Python 3.8+

    pip (Python package installer)

    Git (for cloning repositories)

    (Optional) Docker Desktop (if choosing to run with Docker)

2. Clone the Repository

First, clone the project repository to your local machine:

git clone https://github.com/YourUsername/ADPredictionWeb.git # Replace with your GitHub username and repository name
cd ADPredictionWeb

3. Set Up Python Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies:

python -m venv venv
source venv/bin/activate  # macOS/Linux
# Or on Windows: .\venv\Scripts\activate

4. Install Dependencies

With the virtual environment activated, install all necessary Python packages:

pip install -r requirements.txt

5. Place Model Files

Ensure that you have placed the pre-trained AD_Prediction.joblib and imputer.joblib files in the model/ directory of your project. If these files are missing, the application will not be able to load the model and will raise an error.
6. Run the Application
6.1 Running in Development Mode (Flask's Built-in Server)

From the project root directory, with the virtual environment activated, run:

python app.py

The application will be accessible at http://127.0.0.1:5000/.
6.2 Running with Gunicorn (Production Mode Simulation)

For a better simulation of a production environment, you can run with Gunicorn:

gunicorn -b 0.0.0.0:5000 app:app

The application will be accessible at http://0.0.0.0:5000/ (can be accessed via http://localhost:5000/).
7. Running with Docker (Recommended for Local Environment Consistency)

If you have Docker Desktop installed, you can build and run the application using Docker to ensure environment consistency.

    Build the Docker Image: From the project root directory, run:

    docker build -t ad-predictor-app .

    Run the Docker Container:

    docker run -p 5000:5000 ad-predictor-app

    The application will be accessible at http://localhost:5000/.

Deployment

This Flask application can be deployed to various platforms:

    PaaS Platforms (e.g., Heroku, Render): The simplest deployment method, requiring only Procfile and requirements.txt configuration, followed by code push.

    VPS (Virtual Private Server): Offers full control, requiring manual configuration of Nginx (reverse proxy) and Gunicorn (WSGI server).

    Container Orchestration Platforms (e.g., Google Cloud Run, AWS ECS, Azure Container Instances): Push the Docker image to a container registry, then run it on a cloud platform.

Model Information

The prediction model and data imputer used in this project are pre-trained joblib files. They are loaded using joblib.load() when the application starts.

    AD_Prediction.joblib: The core classification model used to predict the probability of Alzheimer's Disease.

    imputer.joblib: The data imputer used to handle missing values in the input data.

Contributing

Contributions to this project are welcome! If you have any suggestions for improvements, bug reports, or new feature requests, please feel free to submit an Issue or a Pull Request.
