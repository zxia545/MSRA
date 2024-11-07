import asyncio
import os
import json
from openai import OpenAI
from datasets import load_dataset

# Step 1: Define file paths and set up the OpenAI client
output_file = 'llama3.1_70b_umls.json'
port = 8003
client = OpenAI(
    base_url=f"http://localhost:{port}/v1",
    api_key="token-abc123"  # Replace with an actual token if needed
)

# Step 2: Load existing data if available
existing_samples = []

if os.path.exists(output_file):
    print(f"Loading existing data from '{output_file}'...")
    with open(output_file, 'r', encoding='utf-8') as f:
        existing_samples = json.load(f)

# Convert loaded data to a set of unique identifiers for easier comparison
existing_keys = {(sample['split'], sample['head'], sample['relation'], sample['tail']) for sample in existing_samples}

# Step 3: Load UMLS dataset and add new samples if they don't already exist
umls_dataset = load_dataset('VLyb/UMLS')
splits = ['train', 'validation', 'test']

all_samples = existing_samples.copy()  # Start with the existing data

for split in splits:
    samples = umls_dataset[split]
    for sample in samples:
        head = sample['head']
        relation = sample['relation']
        tail = sample['tail']
        unique_key = (split, head, relation, tail)
        
        # Only add the sample if it's not already present in the existing data
        if unique_key not in existing_keys:
            prompt = (
                f"Write a single, clear sentence that uses the exact words '{head}' and '{tail}', "
                f"explicitly connecting them with the relation '{relation}'. Do not add any extra information."
            )
            all_samples.append({
                'split': split,  # Add the split name
                'head': head,
                'relation': relation,
                'tail': tail,
                'prompt': prompt,
                'generated_sentence': None  # Placeholder for the generated sentence
            })

# Step 4: Set up function to generate responses
def generate_sentence(sample):
    prompt = sample['prompt']
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates sentences based on given relationships."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model='gpt-4',
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )
        # Correct way to access the response as per your requirement
        generated_sentence = response.choices[0].message.content
        sample['generated_sentence'] = generated_sentence.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        sample['generated_sentence'] = None

# Step 5: Generate sentences with concurrency control
async def run_with_semaphore(samples):
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

    async def run_with_semaphore_wrapper(sample):
        if sample['generated_sentence'] is None:  # Only generate if not already done
            async with semaphore:
                await asyncio.to_thread(generate_sentence, sample)

    tasks = [run_with_semaphore_wrapper(sample) for sample in samples]
    await asyncio.gather(*tasks)

# Step 6: Run the asyncio event loop if necessary
asyncio.run(run_with_semaphore(all_samples))

# Step 7: Save the results to JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_samples, f, ensure_ascii=False, indent=4)

print(f"The generated sentences have been saved to '{output_file}'.")
