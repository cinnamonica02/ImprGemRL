from datasets import load_dataset
import json
import argparse 


def main():
    parser = argparse.ArgumentParser(description='Download examples from the GSM8K dataset')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Dataset split to use (train or test)')
    parser.add_argument('--num_examples', type=str, default=64,
                        help='Number of examples to download. Use -1 for all examples')
    args = parser.parse_args()

    # loading in dataset

    dataset = load_dataset("openai/gsm8k", "main")

    # get all examples if num_examples is -1, otherwise get specified number of examples
    if args.num_examples == -1:
        data = dataset[args.split]
    else:
        data = dataset[args.split].select(range(range(min(args.num_examples, len(dataset[args.split])))))

    # Convert to a list of dict 

    examples = []
    for item in data:
        examples.append({
            'question': item['question'],
            'answer': item['answer'],
        })

        # create filename w split and no. of examples
        output_file = f'gsm8k_{args.split}_{args.num_examples}.json'

    # save to json file

    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)


    print(f'Downloaded {len(examples)} examples from the GSM8K {args.split} split')
    print(f'Saved to {output_file}')
    print(f'First example: {examples[0]}')
    print('Question:', examples[0]['question'])
    print('Answer:', examples[0]['answer'])


if __name__ == '__main__':
    main()
