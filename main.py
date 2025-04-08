import argparse
from transformers import pipeline
from rouge_score import rouge_scorer

def main():
    args = parseCommands()

    # load files and combine text
    text = "".join(load_text(file) for file in args.files)

    try:
        summary, precision, recall, f1_score, accuracy = summarize_text(text, args.length, args.model)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

    print(f"\nSummary:\n{summary}")
    print("\nEvaluation Metrics:")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall: {recall:.4f}")
    print(f"    F1 Score: {f1_score:.4f}")
    print(f"    Accuracy: {accuracy:.4f}")
    print()

def summarize_text(text, summary_len, model_name):
    """Summarize the given text using a pre-trained model."""
    summarizer = pipeline("summarization", model=model_name)
    max_len, min_len = get_summary_len(text, summary_len)
    print("MIN:", min_len)
    print("MAX:", max_len)
    print()
    print("\nSummarizing text...")

    summary: str = summarizer(text, min_length=min_len, max_length=max_len, do_sample=False)[0]["summary_text"] # type: ignore

    # Compute F1 score using ROUGE-1
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = scorer.score(text, summary)

    precision = scores["rouge1"].precision
    recall = scores["rouge1"].recall
    f1_score = scores["rouge1"].fmeasure
    accuracy = recall  # Approximate accuracy as recall (since it's about how much of the original text is retained)

    return summary, precision, recall, f1_score, accuracy


def get_summary_len(text: str, summary_len):
    text_len_by_word = len(text.strip().split(" "))

    if summary_len == "short":
        max_length, min_length = calc_range(text_len_by_word, 0.30)

    elif summary_len == "mid":
        max_length, min_length = calc_range(text_len_by_word, 0.50)

    elif summary_len == "long":
        max_length, min_length = calc_range(text_len_by_word, 0.70)

    else:
        # by default it will be treated as mid
        max_length, min_length = calc_range(text_len_by_word, 0.50)

    return max_length, min_length

def calc_range(text_len, percentage):
    avg = int(text_len * percentage)
    added_range = int(text_len * 0.1)
    max_length, min_length = avg + added_range, avg - added_range
    return max_length, min_length

def parseCommands():
    parser = argparse.ArgumentParser(description="A CLI for text summarization.")
    parser.add_argument("files", type=str, nargs="+", help="One or more text files to summarize.")
    parser.add_argument("--model", type=str, default="facebook/bart-large-cnn", help="The Hugging Face model to use for summarization (default: facebook/bart-large-cnn).")
    parser.add_argument("--length",
        type=str,
        choices=["short", "mid", "long"],
        default="mid",
        help="Choose the length of the summary: short, mid, or long."
    )

    return parser.parse_args()

"""Load text from a file."""
def load_text(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read().strip()
        if not content:
            raise ValueError(f"Error: '{filename}' is empty.")
        return content

if __name__ == "__main__":
    main()
