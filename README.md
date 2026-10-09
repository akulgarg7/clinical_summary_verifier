# Clinical Summary Verifier

An experimental Streamlit application for automated factual verification of AI-generated clinical discharge summaries against synthetic ground-truth patient records.


## Requirements

* Python 3.10 or a compatible version supported by the project's dependencies
* Git
* A Gemini API key for LLM-based claim extraction

## 1. Clone the repository

```bash
git clone https://github.com/akulgarg7/clinical_summary_verifier.git
cd clinical_summary_verifier
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configure environment variables

Create a file named `.env` in the project root.

Add your own Gemini API key and model configuration:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=your_configured_gemini_model
```

## 5. Run the Streamlit application

From the project root, run:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```
