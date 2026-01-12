"""
Error analysis module for toxicity classification.
"""

import pandas as pd
import os
import config


def extract_misclassified(results_df):
    """
    Extract all misclassified examples from results.

    Returns:
        tuple: (false_positives_df, false_negatives_df)
    """
    valid_df = results_df[results_df['predicted_label'] != -1].copy()

    # False Positives: Non-toxic classified as toxic
    false_positives = valid_df[
        (valid_df['true_label'] == 0) & (valid_df['predicted_label'] == 1)
    ].copy()

    # False Negatives: Toxic classified as non-toxic
    false_negatives = valid_df[
        (valid_df['true_label'] == 1) & (valid_df['predicted_label'] == 0)
    ].copy()

    return false_positives, false_negatives


def extract_invalid_predictions(results_df):
    """
    Extract samples where the model gave invalid responses.
    """
    invalid_df = results_df[results_df['predicted_label'] == -1].copy()
    return invalid_df


def analyze_text_patterns(df, label):
    """
    Analyze patterns in misclassified texts.
    """
    if len(df) == 0:
        return {}

    analysis = {
        'count': len(df),
        'avg_length': df['text'].str.len().mean(),
        'min_length': df['text'].str.len().min(),
        'max_length': df['text'].str.len().max(),
        'samples': df['text'].head(10).tolist(),
        'raw_responses': df['raw_response'].head(10).tolist() if 'raw_response' in df.columns else []
    }

    return analysis


def generate_error_report(results_df, output_path=None):
    """
    Generate a comprehensive error analysis report.
    """
    output_path = output_path or f"{config.REPORTS_DIR}/error_analysis.txt"
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    false_positives, false_negatives = extract_misclassified(results_df)
    invalid_predictions = extract_invalid_predictions(results_df)

    fp_analysis = analyze_text_patterns(false_positives, "False Positive")
    fn_analysis = analyze_text_patterns(false_negatives, "False Negative")
    inv_analysis = analyze_text_patterns(invalid_predictions, "Invalid")

    report = []
    report.append("="*70)
    report.append("ERROR ANALYSIS REPORT")
    report.append("Toxicity Classification with llama3.2:3b")
    report.append("="*70)

    # Summary
    report.append("\n" + "="*70)
    report.append("SUMMARY")
    report.append("="*70)
    total = len(results_df)
    valid = len(results_df[results_df['predicted_label'] != -1])
    correct = results_df['correct'].sum()
    report.append(f"Total samples: {total}")
    report.append(f"Valid predictions: {valid}")
    report.append(f"Correct predictions: {correct}")
    report.append(f"False Positives: {len(false_positives)}")
    report.append(f"False Negatives: {len(false_negatives)}")
    report.append(f"Invalid responses: {len(invalid_predictions)}")

    # False Positives Analysis
    report.append("\n" + "="*70)
    report.append("FALSE POSITIVES ANALYSIS")
    report.append("(Non-toxic text incorrectly classified as toxic)")
    report.append("="*70)

    if fp_analysis:
        report.append(f"\nCount: {fp_analysis['count']}")
        report.append(f"Average text length: {fp_analysis['avg_length']:.1f} characters")
        report.append(f"Length range: {fp_analysis['min_length']} - {fp_analysis['max_length']} characters")
        report.append("\nExample texts:")
        for i, (text, resp) in enumerate(zip(fp_analysis['samples'], fp_analysis['raw_responses']), 1):
            report.append(f"\n{i}. Text: {text[:200]}..." if len(text) > 200 else f"\n{i}. Text: {text}")
            report.append(f"   Model response: {resp}")
    else:
        report.append("\nNo false positives found.")

    # False Negatives Analysis
    report.append("\n" + "="*70)
    report.append("FALSE NEGATIVES ANALYSIS")
    report.append("(Toxic text incorrectly classified as non-toxic)")
    report.append("="*70)

    if fn_analysis:
        report.append(f"\nCount: {fn_analysis['count']}")
        report.append(f"Average text length: {fn_analysis['avg_length']:.1f} characters")
        report.append(f"Length range: {fn_analysis['min_length']} - {fn_analysis['max_length']} characters")
        report.append("\nExample texts:")
        for i, (text, resp) in enumerate(zip(fn_analysis['samples'], fn_analysis['raw_responses']), 1):
            report.append(f"\n{i}. Text: {text[:200]}..." if len(text) > 200 else f"\n{i}. Text: {text}")
            report.append(f"   Model response: {resp}")
    else:
        report.append("\nNo false negatives found.")

    # Invalid Predictions Analysis
    if len(invalid_predictions) > 0:
        report.append("\n" + "="*70)
        report.append("INVALID PREDICTIONS ANALYSIS")
        report.append("(Model responses that could not be parsed)")
        report.append("="*70)

        report.append(f"\nCount: {inv_analysis['count']}")
        report.append("\nExample texts and responses:")
        for i, (text, resp) in enumerate(zip(inv_analysis['samples'], inv_analysis['raw_responses']), 1):
            report.append(f"\n{i}. Text: {text[:200]}..." if len(text) > 200 else f"\n{i}. Text: {text}")
            report.append(f"   Model response: {resp}")

    # Insights and Patterns
    report.append("\n" + "="*70)
    report.append("INSIGHTS AND PATTERNS")
    report.append("="*70)

    report.append("\nPotential patterns in errors:")

    if len(false_positives) > len(false_negatives):
        report.append("- Model tends to be overly cautious, classifying some benign text as toxic")
        report.append("- Consider: sarcasm, strong opinions, or emphatic language may trigger false positives")
    elif len(false_negatives) > len(false_positives):
        report.append("- Model tends to miss some toxic content")
        report.append("- Consider: subtle toxicity, coded language, or context-dependent harm may be missed")
    else:
        report.append("- Errors are balanced between false positives and false negatives")

    if fp_analysis and fp_analysis.get('avg_length', 0) > 150:
        report.append("- Longer texts tend to be misclassified as toxic (false positives)")

    if fn_analysis and fn_analysis.get('avg_length', 0) < 50:
        report.append("- Shorter toxic texts may be harder to detect (false negatives)")

    # Recommendations
    report.append("\n" + "="*70)
    report.append("RECOMMENDATIONS FOR IMPROVEMENT")
    report.append("="*70)
    report.append("\n1. Review and potentially expand few-shot examples")
    report.append("2. Consider adding more diverse examples of edge cases")
    report.append("3. Experiment with prompt engineering (clearer instructions)")
    report.append("4. Try different temperature settings for inference")
    report.append("5. Consider using a larger model for improved accuracy")
    report.append("6. Fine-tuning on domain-specific data could help")

    report_text = "\n".join(report)

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nError analysis report saved to {output_path}")

    return report_text


def save_misclassified_samples(results_df):
    """
    Save misclassified samples to separate CSV files for further analysis.
    """
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    false_positives, false_negatives = extract_misclassified(results_df)
    invalid_predictions = extract_invalid_predictions(results_df)

    if len(false_positives) > 0:
        fp_path = f"{config.REPORTS_DIR}/false_positives.csv"
        false_positives.to_csv(fp_path, index=False)
        print(f"False positives saved to {fp_path}")

    if len(false_negatives) > 0:
        fn_path = f"{config.REPORTS_DIR}/false_negatives.csv"
        false_negatives.to_csv(fn_path, index=False)
        print(f"False negatives saved to {fn_path}")

    if len(invalid_predictions) > 0:
        inv_path = f"{config.REPORTS_DIR}/invalid_predictions.csv"
        invalid_predictions.to_csv(inv_path, index=False)
        print(f"Invalid predictions saved to {inv_path}")


def run_error_analysis(results_df):
    """
    Run complete error analysis pipeline.
    """
    print("\n" + "="*50)
    print("RUNNING ERROR ANALYSIS")
    print("="*50)

    report = generate_error_report(results_df)
    save_misclassified_samples(results_df)

    # Print summary to console
    false_positives, false_negatives = extract_misclassified(results_df)
    invalid = extract_invalid_predictions(results_df)

    print(f"\nError Summary:")
    print(f"  False Positives: {len(false_positives)}")
    print(f"  False Negatives: {len(false_negatives)}")
    print(f"  Invalid Predictions: {len(invalid)}")

    return report


if __name__ == "__main__":
    # Load results and run error analysis
    try:
        results_df = pd.read_csv(f"{config.RESULTS_DIR}/predictions.csv")
        run_error_analysis(results_df)
    except FileNotFoundError:
        print("Results file not found. Please run inference.py first.")
