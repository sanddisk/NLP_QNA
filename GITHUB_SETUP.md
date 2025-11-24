# GitHub Setup Instructions

Your code is now ready to be pushed to GitHub! Follow these steps:

## Step 1: Create a New Repository on GitHub

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `NLP_QNA_CURSOR` (or any name you prefer)
   - **Description**: "Question-Answering and Summarization System with Multi-LLM Support"
   - **Visibility**: Choose **Public** (to share with colleague) or **Private**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

## Step 2: Push Your Code to GitHub

After creating the repository, GitHub will show you commands. Use these commands in your terminal:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/NLP_QNA_CURSOR.git

# Rename branch to main (if needed)
git branch -M main

# Push your code
git push -u origin main
```

## Step 3: Share the Repository Link

Once pushed, share this link with your colleague:
```
https://github.com/YOUR_USERNAME/NLP_QNA_CURSOR
```

## Important Notes for Your Colleague

When your colleague clones the repository, they need to:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/NLP_QNA_CURSOR.git
   cd NLP_QNA_CURSOR
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or: source venv/bin/activate  # Mac/Linux
   pip install -r requirements.txt
   ```

3. **Configure API keys:**
   ```bash
   # Copy the example secrets file
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   # Then edit .streamlit/secrets.toml with their own API keys
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Security Checklist ✅

- ✅ `.gitignore` is configured to exclude sensitive files
- ✅ `secrets.toml` is NOT in the repository (your API keys are safe)
- ✅ `secrets.toml.example` is included as a template
- ✅ `venv/` folder is excluded
- ✅ `__pycache__/` folders are excluded

Your API keys are safe and will NOT be shared on GitHub!

