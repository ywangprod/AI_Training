import sqlite3
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Insecure SQL query - vulnerable to injection
def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()

# XSS vulnerable endpoint
@app.route('/search')
def search():
    query = request.args.get('q', '')
    results = f"<h1>Results for: {query}</h1>"
    return results

# Hardcoded credentials (security issue)
API_KEY = "12345-abcdef-67890"
DATABASE_PASSWORD = "mysecretpassword"

# Insecure deserialization
def load_data(data):
    import pickle
    return pickle.loads(data)  # Dangerous!

if __name__ == "__main__":
    app.run(debug=True)  # Debug mode enabled in production code