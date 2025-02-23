# 📝 AI Resume Screening System 🚀

A **Machine Learning-powered Resume Screening System** that classifies resumes into job categories using NLP and AI models.

---

## 🏆 Features
👉 **Automated Resume Classification** - Predicts job category from resume text  
👉 **BERT & Machine Learning Models** - Uses TF-IDF, Random Forest, XGBoost, and BERT  
👉 **FastAPI Integration** - REST API for easy deployment  
👉 **Preprocessing Pipeline** - Cleans, tokenizes, and vectorizes text  

---

## 🛠 Installation & Setup

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/sohrabbahari/AI-Resume-Screening.git
cd AI-Resume-Screening
```

### 2️⃣ **Create & Activate a Virtual Environment**
```bash
python -m venv env
```
- **Windows:**  
  ```bash
  env\Scripts\activate
  ```
- **Mac/Linux:**  
  ```bash
  source env/bin/activate
  ```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Run the API**
```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```
The API will run on: **`http://localhost:10000`**

---

## 🏗 How It Works
1. **User uploads a resume (text format)**
2. **The model processes & predicts the job category**
3. **The API returns the predicted category**

Example API Response:
```json
{
  "Predicted Job Category": "CONSULTANT"
}
```

---

## 🤝 Contributing
🚀 Want to improve the project? Feel free to submit **pull requests** or open **issues**!

### 🔹 **Ways to Contribute**
- 🛠 Improve model accuracy
- 🌍 Add multi-language support
- ⚡ Optimize API response time
- 📝 Improve documentation

---

## 💡 Future Improvements
📌 **Multi-Label Classification** – Detect multiple skills/jobs  
📌 **Resume Parsing Enhancements** – Handle different file formats  
📌 **Deploy on Cloud** – Host on AWS/GCP for public access  

---

## 👨‍💻 Author
**[Sohrab Bahari](https://github.com/sohrabbahari)**  
📧 Email: sohrab.bahari7@gmail.com
📌 **GitHub:** [AI Resume Screening](https://github.com/sohrabbahari/AI-Resume-Screening)  

---

## ⭐ Show Some Love
If you like this project, **please give it a ⭐ star** on GitHub!

