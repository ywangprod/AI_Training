import os
import openai
import glob
import json
from pathlib import Path

def review_code(file_path):
    """Review a single file for security issues"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if len(content) > 12000:  # Handle large files
            content = content[:6000] + "\n\n...[truncated for length]..." + content[-6000:]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a senior security engineer. Review code for security vulnerabilities and provide specific recommendations."},
                {"role": "user", "content": f"Review this {file_path} for security issues:\n\n{content}"}
            ],
            max_tokens=1000
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error reviewing {file_path}: {str(e)}"

def main():
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Define file patterns to review
    patterns = [
        '**/*.py',
        '**/*.js',
        '**/*.ts',
        '**/*.java',
        '**/*.go',
        '**/*.rb',
        '**/*.php',
        '**/*.cs'
    ]

    results = {}

    for pattern in patterns:
        for file_path in glob.glob(pattern, recursive=True):
            if any(ignore in file_path for ignore in ['node_modules', '.git', 'venv', '__pycache__']):
                continue

            print(f"Reviewing {file_path}...")
            review = review_code(file_path)
            results[file_path] = review

    # Save results
    with open('security_review.md', 'w') as f:
        f.write("# Security Code Review Results\n\n")
        for file_path, review in results.items():
            f.write(f"## {file_path}\n\n")
            f.write(f"{review}\n\n")
            f.write("---\n\n")

    print("Security review completed. Results saved to security_review.md")

if __name__ == "__main__":
    main()