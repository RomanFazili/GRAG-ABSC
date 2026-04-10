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
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results

def extract_metrics(results):
    """Extract metrics from results"""
    metrics = []
    for result in results:
        metric = {
            'model': result['model'],
            'demonstration_selection_method': result['demonstration_selection_method'],
            'accuracy': result['classification_report']['accuracy'] * 100,
            'weighted_f1': result['classification_report']['weighted avg']['f1-score'] * 100
        }
        metrics.append(metric)
    return metrics

def format_model_display_name(model_path):
    """Extract clean model name for display"""
    model_name_map = {
        'meta-llama/Llama-3.2-3B-Instruct': 'Llama-3.2-3B',
        'meta-llama/Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
        'google/gemma-3-12b-it': 'Gemma-3-12B',
        'google/gemma-4-E4B-it': 'Gemma-4-E4B'
    }
    return model_name_map.get(model_path, model_path)

def aggregate_by_demo_selection(all_metrics):
    """Aggregate metrics by model and demonstration selection method"""
    aggregated = defaultdict(lambda: defaultdict(list))
    
    for metric in all_metrics:
        model = metric['model']
        demo_sel = metric['demonstration_selection_method']
        
        aggregated[model][demo_sel].append({
            'accuracy': metric['accuracy'],
            'weighted_f1': metric['weighted_f1']
        })
    
    # Calculate averages
    averaged = {}
    for model, demo_data in aggregated.items():
        averaged[model] = {}
        for demo_sel, scores in demo_data.items():
            avg_acc = statistics.mean([s['accuracy'] for s in scores])
            avg_f1 = statistics.mean([s['weighted_f1'] for s in scores])
            averaged[model][demo_sel] = {
                'accuracy': avg_acc,
                'weighted_f1': avg_f1
            }
    
    return averaged

def generate_latex_tables(averaged):
    """Generate LaTeX tables for accuracy and F1-score"""
    
    # Model order and display names
    model_order = [
        'meta-llama/Llama-3.2-3B-Instruct',
        'google/gemma-4-E4B-it',
        'meta-llama/Llama-3.1-8B-Instruct',
        'google/gemma-3-12b-it'
    ]
    demo_selection_order = ['BM25', 'SimCSE', 'Graph']
    
    # ACCURACY TABLE
    accuracy_table = r"""\begin{table}[htbp]
\centering
\caption{Average accuracy (\%) across demonstration selection strategies (averaged over datasets, ontology settings, and formats)}
\label{tab:acc_avg_comparison}
\begin{tabular}{lccc}
\toprule
Model & BM25 & SimCSE & Graph \\
\midrule
"""
    
    # Add data rows for accuracy
    row_data_acc = []
    for model in model_order:
        model_display = format_model_display_name(model)
        values = []
        
        for demo_sel in demo_selection_order:
            if model in averaged and demo_sel in averaged[model]:
                val = averaged[model][demo_sel]['accuracy']
                values.append(val)
            else:
                values.append(None)
        
        # Find max value to bold
        max_val = max([v for v in values if v is not None]) if any(v is not None for v in values) else None
        
        # Format row
        formatted_values = []
        for val in values:
            if val is None:
                formatted_values.append('--')
            elif val == max_val:
                formatted_values.append(f'\\textbf{{{val:.1f}}}')
            else:
                formatted_values.append(f'{val:.1f}')
        
        row = f"{model_display:20} & {' & '.join(formatted_values)} \\\\\n"
        accuracy_table += row
        row_data_acc.append((model_display, values))
    
    accuracy_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # F1-SCORE TABLE
    f1_table = r"""\begin{table}[htbp]
\centering
\caption{Average weighted F1-score (\%) across demonstration selection strategies (averaged over datasets, ontology settings, and formats)}
\label{tab:f1_avg_comparison}
\begin{tabular}{lccc}
\toprule
Model & BM25 & SimCSE & Graph \\
\midrule
"""
    
    # Add data rows for F1
    row_data_f1 = []
    for model in model_order:
        model_display = format_model_display_name(model)
        values = []
        
        for demo_sel in demo_selection_order:
            if model in averaged and demo_sel in averaged[model]:
                val = averaged[model][demo_sel]['weighted_f1']
                values.append(val)
            else:
                values.append(None)
        
        # Find max value to bold
        max_val = max([v for v in values if v is not None]) if any(v is not None for v in values) else None
        
        # Format row
        formatted_values = []
        for val in values:
            if val is None:
                formatted_values.append('--')
            elif val == max_val:
                formatted_values.append(f'\\textbf{{{val:.1f}}}')
            else:
                formatted_values.append(f'{val:.1f}')
        
        row = f"{model_display:20} & {' & '.join(formatted_values)} \\\\\n"
        f1_table += row
        row_data_f1.append((model_display, values))
    
    f1_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return accuracy_table, f1_table, row_data_acc, row_data_f1

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
    
    print(f"Total results loaded: {len(all_metrics)}\n")
    
    # Aggregate by demonstration selection method
    averaged = aggregate_by_demo_selection(all_metrics)
    
    # Generate LaTeX tables
    accuracy_table, f1_table, acc_data, f1_data = generate_latex_tables(averaged)
    
    # Print data summary
    print("=" * 80)
    print("ACCURACY DATA SUMMARY")
    print("=" * 80)
    for model_name, values in acc_data:
        print(f"{model_name:20} | BM25: {values[0]:6.1f} | SimCSE: {values[1]:6.1f} | Graph: {values[2]:6.1f}")
    
    print("\n" + "=" * 80)
    print("F1-SCORE DATA SUMMARY")
    print("=" * 80)
    for model_name, values in f1_data:
        print(f"{model_name:20} | BM25: {values[0]:6.1f} | SimCSE: {values[1]:6.1f} | Graph: {values[2]:6.1f}")
    
    # Print tables
    print("\n" + "=" * 80)
    print("ACCURACY TABLE (LATEX)")
    print("=" * 80)
    print(accuracy_table)
    
    print("=" * 80)
    print("F1-SCORE TABLE (LATEX)")
    print("=" * 80)
    print(f1_table)
    
    # Save to files
    with open(results_dir / 'demo_selection_accuracy.tex', 'w') as f:
        f.write(accuracy_table)
    print(f"Accuracy table saved to: demo_selection_accuracy.tex")
    
    with open(results_dir / 'demo_selection_f1.tex', 'w') as f:
        f.write(f1_table)
    print(f"F1 table saved to: demo_selection_f1.tex")

if __name__ == '__main__':
    main()
