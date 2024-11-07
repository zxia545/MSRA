import asyncio
from openai import OpenAI
from datasets import load_dataset
import json

# Step 1: Load the datasets from Hugging Face
umls_dataset = load_dataset('VLyb/UMLS')

# Step 2: Select all train dataset
umls_samples = umls_dataset['train']

# Step 3: Create prompts based on the head, relation, and tail of each sample
def create_prompt(head, relation, tail):
    prompt = (
        f"Write a single, clear sentence that uses the exact words '{head}' and '{tail}', "
        f"explicitly connecting them with the relation '{relation}'. Do not add any extra information."
    )
    return prompt

# Prepare the data with prompts
all_samples = []

for sample in umls_samples:
    head = sample['head']
    relation = sample['relation']
    tail = sample['tail']
    prompt = create_prompt(head, relation, tail)
    all_samples.append({'head': head, 'relation': relation, 'tail': tail, 'prompt': prompt})

# Step 4: Set up the OpenAI client to use the local vllm API
port = 8003
client = OpenAI(
    base_url=f"http://localhost:{port}/v1",
    api_key="token-abc123"  # Replace this with an actual token if required for your vllm server setup
)

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
        generated_sentence = response['choices'][0]['message']['content']
        sample['generated_sentence'] = generated_sentence.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        sample['generated_sentence'] = None

async def run_with_semaphore(samples):
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

    async def run_with_semaphore_wrapper(sample):
        async with semaphore:
            await asyncio.to_thread(generate_sentence, sample)

    tasks = [run_with_semaphore_wrapper(sample) for sample in samples]
    await asyncio.gather(*tasks)

# Run the asyncio event loop
asyncio.run(run_with_semaphore(all_samples))

# Step 5: Save the results as a JSON file
with open('generated_sentences.json', 'w', encoding='utf-8') as f:
    json.dump(all_samples, f, ensure_ascii=False, indent=4)

print("The generated sentences have been saved to 'generated_sentences.json'.")
