import os
import google.generativeai as genai
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db
from sqlalchemy import text
import traceback

chat_bp = Blueprint("chat", __name__)

# System Prompt describing the database schema
SYSTEM_PROMPT = """
You are a helpful, senior SQL developer and AI Assistant for an ANPR (Automatic Number Plate Recognition) system.
You have access to a MySQL database with the following schema:

TABLE users:
- id (Integer, Primary Key)
- username (String)
- email (String)
- role (String) -- Can be 'normal_user', 'operational_user', 'admin'
- is_active (Boolean)
- created_at (DateTime)
- updated_at (DateTime)

TABLE plate_records:
- id (Integer, Primary Key)
- user_email (String, Foreign Key to users.email)
- image_path (String)
- plate_text (String)
- yolo_confidence (Float)
- ocr_confidence (Float)
- timestamp (DateTime)

YOUR TASK:
1. The user will ask a question.
2. If the user's question CANNOT be answered without querying the database (e.g., "how many plates...", "who is the admin", "show me recent detections"), you MUST MUST MUST respond with ONLY a raw SQL query. 
IMPORTANT: DO NOT include markup (like ```sql ... ```), DO NOT include any text before or after the query. ONLY the raw SQL query.
Make sure to use standard MySQL syntax. Only write SELECT queries.
3. If the user's question does NOT require a database query (e.g., "what is ANPR?", "who are you?", "how does this app work?"), answer them directly in plain text in a helpful manner. Do NOT return a SQL query in this case.

RULES FOR SQL:
- Only return SELECT commands. No INSERT/UPDATE/DELETE.
- LIMIT results to 10 unless specifically asked for more to avoid massive payloads.
- If asking about detection records, join with users using `plate_records.user_email = users.email`.
"""

# Configure Gemini once globally for the app
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    # Using gemini-2.5-flash as this API key natively supports it
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    model = None


@chat_bp.route("/chat", methods=["POST"])
@jwt_required()
def chat_with_db():
    if not model:
        return jsonify({"success": False, "message": "GEMINI_API_KEY is not configured on the backend."}), 500

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"success": False, "message": "Missing 'message' field."}), 400

    user_message = data["message"]
    history = data.get("history", []) # Optional history for context

    # 1. Ask Gemini to generate SQL or a direct answer
    prompt = f"{SYSTEM_PROMPT}\n\nUser Question: {user_message}"
    
    try:
        response = model.generate_content(prompt)
        ai_output = response.text.strip()
    except Exception as e:
        return jsonify({"success": False, "message": f"Error contacting AI: {str(e)}"}), 500

    # 2. Check if the AI returned a SQL query
    # A simple heuristic: if it starts with SELECT, treat it as a SQL query run.
    if ai_output.upper().startswith("SELECT"):
        sql_query = ai_output
        print(f"[AI Chatbot] Executing SQL: {sql_query}")
        
        # 3. Execute the SQL safely (read-only)
        try:
            # SQLAlchemy text() wrapper
            result = db.session.execute(text(sql_query))
            
            # Fetch results and convert to list of dicts
            rows = result.fetchall()
            keys = result.keys()
            db_results = [dict(zip(keys, row)) for row in rows]
            
            # 4. Feed results back to Gemini for a natural language summary
            summary_prompt = f"""
            The user asked: "{user_message}"
            I ran this SQL query: "{sql_query}"
            And I got these results from the database: {db_results}
            
            Please provide a friendly, natural language response answering the user's question based ONLY on these results.
            Keep it concise. Do not mention that you ran a SQL query, just give the answer naturally.
            """
            
            summary_response = model.generate_content(summary_prompt)
            final_answer = summary_response.text.strip()
            
            return jsonify({"success": True, "answer": final_answer, "sql_executed": sql_query})
            
        except Exception as e:
            traceback.print_exc()
            return jsonify({
                "success": False, 
                "message": "The AI generated an invalid database query. Please rephrase your question.",
                "error": str(e),
                "sql_attempted": sql_query
            }), 400
            
    else:
        # 3b. AI returned a direct answer (no DB query needed)
        return jsonify({"success": True, "answer": ai_output, "sql_executed": None})
