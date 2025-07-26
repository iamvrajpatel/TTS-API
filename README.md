# TTS API

This project provides a Text-to-Speech (TTS) API using Python.

## Requirements

- **Python version:** 3.11.9  
  Make sure you have Python 3.11.9 installed. You can download it from [python.org](https://www.python.org/downloads/release/python-3119/).

## Installation

1. **Set up a virtual environment** (recommended):

   ```sh
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Clone the repository** (if applicable):

   ```sh
   git clone https://github.com/iamvrajpatel/TTS-API.git
   cd tts-api
   ```

3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

   If you need development or extra dependencies, check for additional requirements files or documentation.

## Usage

- Place your main application code in `main.py`.
- Run the API server or scripts as needed:

   ```sh
   python main.py
   ```

- Example API call using `curl`:

   ```sh
   curl --location 'http://localhost:8000/tts/' \
   --header 'Content-Type: application/json' \
   --data '{
       "text": "howw youu doinnnn???",
       "language": "en",
       "gender": "male"
   }'
   ```

## Notes

- Ensure you are using Python 3.11.9 to avoid compatibility issues.
- If you add new libraries, update `requirements.txt` accordingly.

## License

This project is provided as-is. See individual library licenses for details.
