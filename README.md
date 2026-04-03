# SLR Analyser — Streamlit Cloud Setup

## Repo structure (ALL files must be in root)
```
your-repo/
├── app.py
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## Step-by-step fix

### 1. Clear your GitHub repo
Delete ALL existing files in the repo (or create a fresh repo).

### 2. Upload these exact files
- `app.py`  
- `requirements.txt`  
- `.streamlit/config.toml`  

Make sure `requirements.txt` is in the **root**, not in a subfolder.

### 3. Streamlit Cloud settings
- **Main file path:** `app.py`  
- **Python version:** 3.11  

### 4. Reboot the app
In Streamlit Cloud → Manage app → Reboot

### 5. Verify installation worked
Click "Manage app" → Logs tab. You should see:
```
Installing collected packages: openai, pypdf, python-docx ...
```
If you see `anthropic` being installed instead, the old requirements.txt is still there.

## requirements.txt contents (exact)
```
openai==1.51.0
streamlit==1.39.0
pypdf==4.3.1
python-docx==1.1.2
pandas==2.2.3
```
