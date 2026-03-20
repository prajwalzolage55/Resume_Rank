import os
import re
import datetime
import PyPDF2
import spacy
import nltk
import numpy as np
from flask import Flask, jsonify, render_template, request, session, redirect, url_for, flash, Blueprint
from flask_pymongo import PyMongo
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'default_secret_key')
    MONGO_URI = os.environ.get('MONGO_URI')
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    UPLOAD_FOLDER = 'uploads'

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig if os.environ.get('FLASK_ENV') != 'production' else ProductionConfig
}

# --- EXTENSIONS ---
mongo = PyMongo()

# --- UTILS: NLP & DOWNLOADS ---
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    # Using sys.executable to ensure we use the same python environment
    import sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

COMMON_SKILLS = [
    'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'html', 'css', 
    'react', 'angular', 'vue.js', 'node.js', 'express', 'django', 'flask', 
    'ruby on rails', 'spring boot', 'sql', 'mysql', 'postgresql', 'oracle',
    'mongodb', 'cassandra', 'redis', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 
    'terraform', 'machine learning', 'deep learning', 'nlp', 'computer vision', 
    'data science', 'pandas', 'numpy', 'scipy', 'pytorch', 'tensorflow', 'keras',
    'linux', 'git', 'github', 'gitlab', 'ci/cd', 'agile', 'scrum', 'jira'
]

def clean_text(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.\+]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text):
    cleaned = clean_text(text)
    found_skills = set()
    for skill in COMMON_SKILLS:
        escaped_skill = re.escape(skill)
        if re.search(r'\b' + escaped_skill + r'\b', cleaned):
            found_skills.add(skill)
    return list(found_skills)

def extract_text_from_pdf(file_path):
    if not os.path.exists(file_path): return ""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + " "
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text.strip()

# --- UTILS: EMBEDDINGS & SIMILARITY ---
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_embedding(text):
    if not text: return []
    model = get_model()
    return model.encode(text).tolist()

def calculate_similarity(embedding1, embedding2):
    if not embedding1 or not embedding2: return 0.0
    try:
        emb1 = np.array(embedding1).reshape(1, -1)
        emb2 = np.array(embedding2).reshape(1, -1)
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return round(max(0.0, min(100.0, similarity * 100)), 2)
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

# --- UTILS: LLM ---
def analyze_skill_gap(job_description, resume_text):
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key: return "LLM integration is disabled because GROQ_API_KEY is missing."
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    prompt = PromptTemplate(
        template="""You are an expert technical recruiter analyzing a candidate's resume against a job description.
        Job Description: {job_description}
        Candidate Resume: {resume_text}
        Please provide a concise analysis containing:
        1. Key matching skills (bullet points)
        2. Missing skills / Skill Gap (bullet points)
        3. Overall recommendation (1-2 sentences)
        Output directly without any introductory pleasantries.""",
        input_variables=["job_description", "resume_text"]
    )
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"job_description": job_description[:3000], "resume_text": resume_text[:3000]})
    except Exception as e:
        return f"Error during AI analysis: {str(e)}"

# --- MODELS ---
class User:
    @staticmethod
    def create_user(name, email, password, role='candidate'):
        status = 'approved' if role in ['candidate', 'super_admin'] else 'pending'
        user_data = {"name": name, "email": email, "password": generate_password_hash(password), "role": role, "status": status}
        result = mongo.db.users.insert_one(user_data)
        return str(result.inserted_id)
    @staticmethod
    def get_user_by_email(email): return mongo.db.users.find_one({"email": email})
    @staticmethod
    def get_user_by_id(user_id): return mongo.db.users.find_one({"_id": ObjectId(user_id)})
    @staticmethod
    def verify_password(stored_password, provided_password): return check_password_hash(stored_password, provided_password)
    @staticmethod
    def update_user_status(user_id, status):
        mongo.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"status": status}})
        return True

class Job:
    @staticmethod
    def create_job(recruiter_id, title, description, skills_required):
        job_data = {
            "recruiter_id": ObjectId(recruiter_id) if isinstance(recruiter_id, str) else recruiter_id,
            "job_title": title, "job_description": description, "skills_required": skills_required,
            "created_at": datetime.datetime.utcnow()
        }
        result = mongo.db.jobs.insert_one(job_data)
        return str(result.inserted_id)
    @staticmethod
    def get_all_jobs(): return list(mongo.db.jobs.find().sort("created_at", -1))
    @staticmethod
    def get_job_by_id(job_id): return mongo.db.jobs.find_one({"_id": ObjectId(job_id)})
    @staticmethod
    def get_jobs_by_recruiter(recruiter_id): return list(mongo.db.jobs.find({"recruiter_id": ObjectId(recruiter_id)}).sort("created_at", -1))
    @staticmethod
    def delete_job(job_id):
        mongo.db.jobs.delete_one({"_id": ObjectId(job_id)})
        mongo.db.matches.delete_many({"job_id": ObjectId(job_id)})
        return True

class Resume:
    @staticmethod
    def save_resume(candidate_id, file_path, extracted_text, skills, experience):
        resume_data = {
            "candidate_id": ObjectId(candidate_id) if isinstance(candidate_id, str) else candidate_id,
            "file_path": file_path, "extracted_text": extracted_text, "skills": skills, "experience": experience
        }
        mongo.db.resumes.update_one({"candidate_id": resume_data["candidate_id"]}, {"$set": resume_data}, upsert=True)
        return True
    @staticmethod
    def get_by_candidate_id(candidate_id):
        return mongo.db.resumes.find_one({"candidate_id": ObjectId(candidate_id) if isinstance(candidate_id, str) else candidate_id})
    @staticmethod
    def get_all_resumes(): return list(mongo.db.resumes.find())

class Match:
    @staticmethod
    def save_match(job_id, candidate_id, similarity_score, skill_gap=None):
        match_data = {
            "job_id": ObjectId(job_id) if isinstance(job_id, str) else job_id,
            "candidate_id": ObjectId(candidate_id) if isinstance(candidate_id, str) else candidate_id,
            "similarity_score": similarity_score, "skill_gap": skill_gap
        }
        mongo.db.matches.update_one({"job_id": match_data["job_id"], "candidate_id": match_data["candidate_id"]}, {"$set": match_data}, upsert=True)
        return True
    @staticmethod
    def get_matches_for_job(job_id):
        pipeline = [
            {"$match": {"job_id": ObjectId(job_id)}}, {"$sort": {"similarity_score": -1}},
            {"$lookup": {"from": "users", "localField": "candidate_id", "foreignField": "_id", "as": "candidate_info"}},
            {"$unwind": "$candidate_info"},
            {"$lookup": {"from": "resumes", "localField": "candidate_id", "foreignField": "candidate_id", "as": "resume_info"}},
            {"$unwind": {"path": "$resume_info", "preserveNullAndEmptyArrays": True}}
        ]
        return list(mongo.db.matches.aggregate(pipeline))
    @staticmethod
    def get_matches_for_candidate(candidate_id):
        pipeline = [
            {"$match": {"candidate_id": ObjectId(candidate_id)}},
            {"$lookup": {"from": "jobs", "localField": "job_id", "foreignField": "_id", "as": "job_info"}},
            {"$unwind": "$job_info"}, {"$sort": {"similarity_score": -1}}
        ]
        return list(mongo.db.matches.aggregate(pipeline))

# --- BLUEPRINTS & ROUTES ---
auth_bp = Blueprint('auth', __name__)
admin_bp = Blueprint('admin', __name__)
recruiter_bp = Blueprint('recruiter', __name__)
candidate_bp = Blueprint('candidate', __name__)

# Auth Routes
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET': return render_template('auth/login.html')
    data = request.form
    user = User.get_user_by_email(data.get('email'))
    if user and User.verify_password(user['password'], data.get('password')):
        if user['status'] == 'pending':
            flash('Your account is pending admin approval.', 'warning')
            return redirect(url_for('auth.login'))
        session.update({'user_id': str(user['_id']), 'role': user['role'], 'name': user['name']})
        if user['role'] == 'super_admin': return redirect(url_for('admin.dashboard'))
        elif user['role'] == 'recruiter': return redirect(url_for('recruiter.dashboard'))
        return redirect(url_for('candidate.dashboard'))
    flash('Invalid email or password.', 'danger')
    return redirect(url_for('auth.login'))

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET': return render_template('auth/register.html')
    data = request.form
    if User.get_user_by_email(data.get('email')):
        flash('Email already registered.', 'danger')
        return redirect(url_for('auth.register'))
    User.create_user(data.get('name'), data.get('email'), data.get('password'), data.get('role', 'candidate'))
    flash('Registration successful! Please log in.', 'success')
    return redirect(url_for('auth.login'))

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))

# Admin Routes
@admin_bp.before_request
def require_admin():
    if session.get('role') != 'super_admin':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('auth.login'))

@admin_bp.route('/dashboard')
def dashboard():
    users = list(mongo.db.users.find())
    return render_template('admin/dashboard.html', users=users, total_users=len(users), 
                           total_jobs=mongo.db.jobs.count_documents({}), total_resumes=mongo.db.resumes.count_documents({}))

@admin_bp.route('/approve/<user_id>', methods=['POST'])
def approve_recruiter(user_id):
    User.update_user_status(user_id, 'approved')
    flash('User approved successfully.', 'success')
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/delete/<user_id>', methods=['POST'])
def delete_user(user_id):
    mongo.db.users.delete_one({"_id": ObjectId(user_id)})
    flash('User deleted successfully.', 'success')
    return redirect(url_for('admin.dashboard'))

# Recruiter Routes
@recruiter_bp.before_request
def require_recruiter():
    if session.get('role') not in ['recruiter', 'super_admin']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('auth.login'))

@recruiter_bp.route('/dashboard')
def dashboard():
    jobs = Job.get_jobs_by_recruiter(session['user_id'])
    return render_template('recruiter/dashboard.html', jobs=jobs)

@recruiter_bp.route('/post_job', methods=['GET', 'POST'])
def post_job():
    if request.method == 'GET': return render_template('recruiter/post_job.html')
    title, desc = request.form.get('title'), request.form.get('description')
    skills = [s.strip() for s in request.form.get('skills_required', '').split(',') if s.strip()]
    job_id = Job.create_job(session['user_id'], title, desc, skills)
    job_emb = get_embedding(desc + " " + " ".join(skills))
    for res in Resume.get_all_resumes():
        score = calculate_similarity(job_emb, get_embedding(res.get('extracted_text', '')))
        Match.save_match(job_id, str(res['candidate_id']), score)
    flash('Job posted and candidates ranked successfully!', 'success')
    return redirect(url_for('recruiter.dashboard'))

@recruiter_bp.route('/job/<job_id>')
def view_job(job_id):
    return render_template('recruiter/view_job.html', job=Job.get_job_by_id(job_id), matches=Match.get_matches_for_job(job_id))

@recruiter_bp.route('/analyze/<job_id>/<candidate_id>', methods=['POST'])
def analyze_candidate(job_id, candidate_id):
    job, resume = Job.get_job_by_id(job_id), Resume.get_by_candidate_id(candidate_id)
    if job and resume:
        job_text = f"{job['job_description']} required skills: {', '.join(job['skills_required'])}"
        analysis = analyze_skill_gap(job_text, resume.get('extracted_text', ''))
        mongo.db.matches.update_one({"job_id": job['_id'], "candidate_id": resume['candidate_id']}, {"$set": {"skill_gap": analysis}})
        flash('AI Analysis complete.', 'success')
    else: flash('Data not found.', 'danger')
    return redirect(url_for('recruiter.view_job', job_id=job_id))

@recruiter_bp.route('/delete_job/<job_id>', methods=['POST'])
def delete_job(job_id):
    Job.delete_job(job_id)
    flash('Job deleted.', 'success')
    return redirect(url_for('recruiter.dashboard'))

# Candidate Routes
@candidate_bp.before_request
def require_candidate():
    if session.get('role') != 'candidate':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('auth.login'))

@candidate_bp.route('/dashboard')
def dashboard():
    return render_template('candidate/dashboard.html', resume=Resume.get_by_candidate_id(session['user_id']), 
                           matches=Match.get_matches_for_candidate(session['user_id']))

@candidate_bp.route('/upload', methods=['POST'])
def upload_resume():
    file = request.files.get('resume')
    if file and file.filename.lower().endswith('.pdf'):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        path = os.path.join(Config.UPLOAD_FOLDER, f"{session['user_id']}_{secure_filename(file.filename)}")
        file.save(path)
        text = extract_text_from_pdf(path)
        Resume.save_resume(session['user_id'], path, text, extract_skills(text), "")
        res_emb = get_embedding(text)
        for job in Job.get_all_jobs():
             job_text = f"{job['job_description']} {' '.join(job['skills_required'])}"
             Match.save_match(job['_id'], session['user_id'], calculate_similarity(get_embedding(job_text), res_emb))
        flash('Resume uploaded and processed successfully!', 'success')
    else: flash('Invalid file. Please upload a PDF.', 'danger')
    return redirect(url_for('candidate.dashboard'))

# --- APP FACTORY ---
def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    mongo.init_app(app)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(recruiter_bp, url_prefix='/recruiter')
    app.register_blueprint(candidate_bp, url_prefix='/candidate')

    @app.route('/health')
    def health_check(): return jsonify({"status": "healthy"}), 200
    @app.route('/')
    def index(): return render_template('index.html')
    return app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # In production, we should use a production server like gunicorn,
    # but for local testing or simple deployment, this handles the port.
    create_app().run(host='0.0.0.0', port=port)
