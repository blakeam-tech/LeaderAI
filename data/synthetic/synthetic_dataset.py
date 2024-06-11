import pandas as pd
import openai
import json

# Load the dataset
df = pd.read_csv('training_dataset.csv')
json_data = df.to_json(orient='records')

# Load prompts once
with open('/Users/blake/LeaderAI/prompts/prompt_a.txt') as f:
    inaccurate_response_prompt = f.read()

with open('/Users/blake/LeaderAI/prompts/prompt_b.txt') as f:
    broad_response_prompt = f.read()

with open('/Users/blake/LeaderAI/prompts/prompt_c.txt') as f:
    ideal_response_prompt = f.read()

# Initialize OpenAI client
client = openai.OpenAI(api_key='sk-proj-cnT1KkofCEUdTDPHeVWLT3BlbkFJY1p393Se9HPFPDuvbHTh')  # Replace 'your-api-key-here' with your actual API key

def generate_synthetic_example(prompt):
    required_keys = ['Model Context', 'Response', 'Response Score (1 to 10)', 'Additional Feedback', 'Cluster']
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at generating synthetic data."},
                {"role": "user", "content": prompt}
            ]
        )
        # Correctly accessing the message content from the API response
        response_json = json.loads(response.choices[0].message.content)  # Changed 'message.content' to 'text'

        if set(response_json.keys()) == set(required_keys):
            return response_json
    except Exception as e:
        print(f"Error generating synthetic example: {e}")
        return None

# Generate synthetic examples
synthetic_examples = []
for i in range(500):
    for prompt in [inaccurate_response_prompt, broad_response_prompt, ideal_response_prompt]:
        synthetic_example = generate_synthetic_example(prompt)
        if synthetic_example:
            synthetic_examples.append(synthetic_example)

# Load the original JSON data
original_data = json.loads(json_data)

# Append synthetic examples to the original data
original_data.extend(synthetic_examples)

# Convert the updated data back to JSON format
updated_json_data = json.dumps(original_data, indent=2)

# Save the updated JSON data to a file
with open('updated_training_dataset.json', 'w') as f:
    f.write(updated_json_data)

print("Synthetic examples added and saved to 'updated_training_dataset.json'")