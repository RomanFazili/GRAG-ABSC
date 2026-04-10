import json
from pathlib import Path
from collections import defaultdict
import statistics

def load_results(file_path):
    """Load JSONL results file"""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON line: {line[:50]}...")
    return results

def extract_metrics(results):
    """Extract accuracy and weighted f1 from results"""
    metrics = []
    for result in results:
        metric = {
            'model': result['model'],
            'ontology_selection': result['ontology_selection_method'],
            'ontology_format': result['ontology_format'],
            'accuracy': result['classification_report']['accuracy'] * 100,
            'weighted_f1': result['classification_report']['weighted avg']['f1-score'] * 100
        }
        metrics.append(metric)
    return metrics

def aggregate_metrics(all_metrics):
    """Aggregate metrics by model, ontology selection, and format"""
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for metric in all_metrics:
        model = metric['model']
        onto_sel = metric['ontology_selection']
        onto_fmt = metric['ontology_format']
        
        aggregated[model][onto_sel][onto_fmt].append({
            'accuracy': metric['accuracy'],
            'weighted_f1': metric['weighted_f1']
        })
    
    # Calculate averages
    averaged = {}
    for model, onto_data in aggregated.items():
        averaged[model] = {}
        for onto_sel, fmt_data in onto_data.items():
            averaged[model][onto_sel] = {}
            for fmt, scores in fmt_data.items():
                avg_acc = statistics.mean([s['accuracy'] for s in scores])
                avg_f1 = statistics.mean([s['weighted_f1'] for s in scores])
                averaged[model][onto_sel][fmt] = {
                    'accuracy': avg_acc,
                    'weighted_f1': avg_f1
                }
    
    return averaged

def format_model_name(model_path):
    """Extract clean model name from path"""
    model_names = {
        'meta-llama/Llama-3.2-3B-Instruct': 'Llama-3.2-3B-Instruct',
        'meta-llama/Llama-3.1-8B-Instruct': 'Llama-3.1-8B-Instruct',
        'google/gemma-3-12b-it': 'Gemma-3-12B-it',
        'google/gemma-4-E4B-it': 'Gemma-4-E4B-it'
    }
    return model_names.get(model_path, model_path)

def format_format_name(fmt):
    """Map format codes to display names"""
    format_map = {
        'xml': 'XML/RDF',
        'n3': 'Notation3',
        'nt': 'N-Triples',
        'turtle': 'Turtle'
    }
    return format_map.get(fmt, fmt)

def generate_latex_table(averaged, metric_type='accuracy', use_booktabs=True):
    """Generate LaTeX table for accuracy or weighted f1
    
    Args:
        averaged: Dictionary of aggregated metrics
        metric_type: 'accuracy' or 'weighted_f1'
        use_booktabs: If True, uses booktabs package for better formatting
    """
    
    # Determine metric display name and label suffix
    if metric_type == 'accuracy':
        metric_key = 'accuracy'
        caption_suffix = 'accuracy (%)'
        label_suffix = 'acc'
        metric_format = '{:.1f}'
    else:
        metric_key = 'weighted_f1'
        caption_suffix = 'weighted F1 score (%)'
        label_suffix = 'f1'
        metric_format = '{:.1f}'
    
    # Sort models for consistent ordering
    model_order = ['meta-llama/Llama-3.2-3B-Instruct', 'google/gemma-3-12b-it', 'meta-llama/Llama-3.1-8B-Instruct', 'google/gemma-4-E4B-it']
    format_order = ['xml', 'n3', 'nt', 'turtle']
    format_names = [format_format_name(f) for f in format_order]
    
    if use_booktabs:
        # Generate LaTeX with booktabs
        top_rule = '\\toprule'
        mid_rule = '\\midrule'
        bot_rule = '\\bottomrule'
        cmidrule = '\\cmidrule(lr)'
        arraystretch = '1.15'
    else:
        # Standard LaTeX version (works without booktabs package)
        top_rule = '\\hline'
        mid_rule = '\\hline'
        bot_rule = '\\hline'
        cmidrule = ''
        arraystretch = '1.3'
    
    # Build header
    format_headers = ' & '.join(format_names)
    latex_start = f"""\\begin{{table}}[htbp]
\\centering
\\renewcommand{{\\arraystretch}}{{{arraystretch}}}
\\caption{{Model {caption_suffix} across ontology formats and ontology injection (aggregated over demonstration selection strategies and datasets)}}
\\label{{tab:format_comparison_{label_suffix}}}
\\small
\\begin{{tabular}}{{llccccc}}
{top_rule}
Model & Ontology & {format_headers} \\\\
{mid_rule}
"""
    
    latex = latex_start
    
    # Add data rows
    for model_idx, model in enumerate(model_order):
        model_display = format_model_name(model)
        
        # Get No Ontology value (XML/RDF)
        no_onto_val = None
        if model in averaged and 'Nothing' in averaged[model] and 'xml' in averaged[model]['Nothing']:
            no_onto_val = averaged[model]['Nothing']['xml'][metric_key]
        
        # Process Partial and Full ontology
        for onto_idx, onto_sel in enumerate(['Partial', 'Full']):
            onto_display = f'{onto_sel.lower()} ontology'
            
            # Get values for each format for this ontology selection
            values = []
            for fmt in format_order:
                if model in averaged and onto_sel in averaged[model] and fmt in averaged[model][onto_sel]:
                    val = averaged[model][onto_sel][fmt][metric_key]
                    values.append(val)
                else:
                    values.append(None)
            
            # Find max value in this row to bold
            max_val = max([v for v in values if v is not None]) if any(v is not None for v in values) else None
            
            # Format values with bold for max
            formatted_values = []
            for val in values:
                if val is None:
                    formatted_values.append('--')
                elif val == max_val:
                    formatted_values.append(f'\\textbf{{{metric_format.format(val)}}}')
                else:
                    formatted_values.append(metric_format.format(val))
            
            # Format No Ontology value
            no_onto_display = '--'
            if no_onto_val is not None:
                no_onto_display = metric_format.format(no_onto_val)
            
            # Add row
            formatted_row = ' & '.join(formatted_values)
            if use_booktabs:
                # Use multirow for booktabs version
                if onto_idx == 0:  # First ontology (Partial)
                    latex += f"\\multirow{{2}}{{*}}{{{model_display}}} & {onto_display} & {no_onto_display} & {formatted_row} \\\\\n"
                else:  # Second ontology (Full)
                    latex += f" & {onto_display} & {no_onto_display} & {formatted_row} \\\\\n"
                
                # Add spacing after Full ontology
                if onto_idx == 1:
                    latex += "\\addlinespace[0.5em]\n"
            else:
                # Basic version without multirow or addlinespace
                latex += f"{model_display if onto_idx == 0 else ''} & {onto_display} & {no_onto_display} & {formatted_row} \\\\\n"
                
                # Add extra spacing in basic version
                if onto_idx == 1:
                    latex += "\\rule{0pt}{2.5ex}\n"
    
    latex += f"""{bot_rule}
\\end{{tabular}}
\\end{{table}}
"""
    
    return latex



def main():
    # Define result files
    results_dir = Path(__file__).parent
    files = [
        results_dir / 'final_results_fabian.jsonl',
        results_dir / 'final_results_mink.jsonl',
        results_dir / 'final_results_roman.jsonl',
        results_dir / 'final_results_roman2.jsonl'
    ]
    
    # Load and process all results
    all_metrics = []
    for file in files:
        if file.exists():
            print(f"Loading {file.name}...")
            results = load_results(file)
            metrics = extract_metrics(results)
            all_metrics.extend(metrics)
    
    print(f"Total results loaded: {len(all_metrics)}")
    
    # Aggregate metrics
    averaged = aggregate_metrics(all_metrics)
    
    # Generate tables with booktabs (recommended)
    accuracy_table = generate_latex_table(averaged, 'accuracy', use_booktabs=True)
    f1_table = generate_latex_table(averaged, 'weighted_f1', use_booktabs=True)
    
    # Generate tables without booktabs (fallback for compilation issues)
    accuracy_table_basic = generate_latex_table(averaged, 'accuracy', use_booktabs=False)
    f1_table_basic = generate_latex_table(averaged, 'weighted_f1', use_booktabs=False)
    
    # Print tables
    print("\n" + "="*80)
    print("ACCURACY TABLE (with booktabs)")
    print("="*80)
    print(accuracy_table)
    
    print("\n" + "="*80)
    print("WEIGHTED F1 SCORE TABLE (with booktabs)")
    print("="*80)
    print(f1_table)
    
    # Save booktabs versions
    with open(results_dir / 'accuracy_table.tex', 'w') as f:
        f.write(accuracy_table)
    print(f"\nAccuracy table saved to: accuracy_table.tex")
    
    with open(results_dir / 'f1_score_table.tex', 'w') as f:
        f.write(f1_table)
    print(f"F1 score table saved to: f1_score_table.tex")
    
    # Save basic versions (without booktabs)
    with open(results_dir / 'accuracy_table_basic.tex', 'w') as f:
        f.write(accuracy_table_basic)
    print(f"Accuracy table (no booktabs) saved to: accuracy_table_basic.tex")
    
    with open(results_dir / 'f1_score_table_basic.tex', 'w') as f:
        f.write(f1_table_basic)
    print(f"F1 score table (no booktabs) saved to: f1_score_table_basic.tex")
    
    print("\n" + "="*80)
    print("REQUIRED PACKAGES")
    print("="*80)
    print("For the standard tables: Add to your preamble:")
    print("  \\usepackage{booktabs}")
    print("\nFor the _basic.tex versions: No additional packages needed")
    print("(only core LaTeX functionality)")


if __name__ == '__main__':
    main()
