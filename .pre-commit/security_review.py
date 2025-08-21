#!/usr/bin/env python3
import sys
import openai
import os

def main():
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY not set")
        return 0
    
    openai.api_key = api_key
    
    for file_path in sys.argv[1:]:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Faster for pre-commit
                messages=[
                    {"role": "system", "content": "Quick security review for pre-commit. Flag critical issues only."},
                    {"role": "user", "content": f"Quick security check: {file_path}\n\n{content[:4000]}"}
                ],
                max_tokens=500
            )
            
            review = response.choices[0].message.content
            if "critical" in review.lower() or "vulnerability" in review.lower():
                print(f"Security issues in {file_path}:")
                print(review)
                return 1  # Fail commit if critical issues found
                
        except Exception as e:
            print(f"Error reviewing {file_path}: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())