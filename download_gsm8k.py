from datasets import load_dataset
import json
import argparse 

def main():
    parser = argparse.ArgumentParser(description='Download examples from the GSM8K dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Dataset split to use (train or test)')
    parser.add_argument('--num_examples', type=int, default=64,
                        help='Number of examples to download. Use -1 for all examples')
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset("openai/gsm8k", "main")

    # Get examples
    if args.num_examples == -1:
        data = dataset[args.split]
    else:
        # FIXED LINE: Removed the extra `range()`
        data = dataset[args.split].select(range(min(args.num_examples, len(dataset[args.split]))))

    # Convert to list of dicts
    examples = [{"question": item["question"], "answer": item["answer"]} for item in data]

    # Save to JSON
    output_file = f'gsm8k_{args.split}_{args.num_examples}.json'
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f'Downloaded {len(examples)} examples from the GSM8K {args.split} split')
    print(f'Saved to {output_file}')

if __name__ == '__main__':
    main()