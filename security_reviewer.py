#!/usr/bin/env python3
import argparse
import openai
import os
import glob
import sys
from pathlib import Path

class SecurityReviewer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key")

        openai.api_key = self.api_key
        self.ignore_patterns = ['node_modules', '.git', 'venv', '__pycache__', '.env']

    def review_file(self, file_path):
        try:
            content = Path(file_path).read_text(encoding='utf-8')

            # Skip very large files or binary files
            if len(content) > 20000:
                return f"File too large for review ({len(content)} characters). Please review manually."

            if '\x00' in content:
                return "Binary file detected. Skipping security review."

            # Truncate very long files but keep important parts
            if len(content) > 10000:
                content = content[:4000] + "\n\n...[middle truncated for brevity]...\n\n" + content[-4000:]

            print(f"Reviewing {file_path}...")

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Using 3.5 for cost efficiency
                messages=[
                    {
                        "role": "system",
                        "content": """You are a security expert analyzing code for vulnerabilities. For each file:
1. Identify security vulnerabilities with severity level (Critical/High/Medium/Low/Info)
2. Provide specific code examples of issues
3. Suggest fixes with code examples when possible
4. Reference relevant OWASP Top 10 categories
5. Keep responses concise but thorough"""
                    },
                    {
                        "role": "user",
                        "content": f"""Perform security code review for {file_path}. Focus on:
- Input validation issues
- Injection vulnerabilities (SQL, XSS, etc.)
- Authentication/authorization flaws
- Insecure dependencies
- Sensitive data exposure
- Cryptographic weaknesses
- Business logic flaws
- Configuration issues

Code content:
{content}"""
                    }
                ],
                max_tokens=1000,
                temperature=0.1  # Low temperature for more deterministic responses
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error reviewing {file_path}: {str(e)}"

    def run_review(self, path):
        results = {}

        if os.path.isfile(path):
            results[path] = self.review_file(path)
        else:
            # Review common source code files
            patterns = [
                '**/*.py', '**/*.js', '**/*.jsx', '**/*.ts', '**/*.tsx',
                '**/*.java', '**/*.go', '**/*.rb', '**/*.php', '**/*.cpp',
                '**/*.h', '**/*.cs', '**/*.html', '**/*.xml', '**/*.json',
                '**/Dockerfile', '**/*.yaml', '**/*.yml'
            ]

            for pattern in patterns:
                for file_path in glob.glob(os.path.join(path, pattern), recursive=True):
                    if any(ignore in file_path for ignore in self.ignore_patterns):
                        continue
                    results[file_path] = self.review_file(file_path)
                    # Brief pause to avoid rate limiting
                    time.sleep(1)

        return results

def main():
    parser = argparse.ArgumentParser(description='Security Code Reviewer')
    parser.add_argument('path', help='File or directory to review', nargs='?', default='.')
    parser.add_argument('--api-key', help='OpenAI API key', default=os.getenv('OPENAI_API_KEY'))
    parser.add_argument('--output', '-o', help='Output file', default='security_review.md')

    args = parser.parse_args()

    try:
        reviewer = SecurityReviewer(args.api_key)
        results = reviewer.run_review(args.path)

        # Write results to markdown file
        with open(args.output, 'w') as f:
            f.write("# Security Code Review Results\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for file_path, review in results.items():
                f.write(f"## {file_path}\n\n")
                f.write(f"{review}\n\n")
                f.write("---\n\n")

        print(f"Security review completed. Results saved to {args.output}")

        # Show summary
        critical_count = 0
        high_count = 0
        for review in results.values():
            if "critical" in review.lower():
                critical_count += 1
            if "high" in review.lower() and "highly" not in review.lower():
                high_count += 1

        print(f"Summary: {critical_count} critical issues, {high_count} high issues found")

    except ValueError as e:
        print(f"Error: {e}")
        print("\nHow to set up your API key:")
        print("1. Get an API key from https://platform.openai.com/api-keys")
        print("2. Set environment variable: export OPENAI_API_KEY='your-key-here'")
        print("3. Or pass it directly: python security_reviewer.py --api-key 'your-key-here' .")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import time
    main()