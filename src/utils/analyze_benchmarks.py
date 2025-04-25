import os
import json
import base64
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
import re
import pickle

def load_benchmark_data(model_name, output_dir='visualizations'):
    """Load benchmark data from real results"""
    return extract_real_benchmark_data(model_name, output_dir)

def extract_data_from_html_files(model_name, output_dir):
    """Extract data from HTML visualization files"""
    model_dir = os.path.join(output_dir, model_name)
    benchmark_data = {
        'model': model_name,
        'visualizations': {}
    }
    
    if not os.path.exists(model_dir):
        return benchmark_data
    
    for viz_file in os.listdir(model_dir):
        if viz_file.endswith('.html'):
            viz_path = os.path.join(model_dir, viz_file)
            viz_name = viz_file.replace('.html', '')
            
            try:
                with open(viz_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                soup = BeautifulSoup(html_content, 'html.parser')
                scripts = soup.find_all('script')
                
                for script in scripts:
                    if script.string and 'var data' in script.string:
                        # Extract the data variable
                        lines = script.string.strip().split('\n')
                        for line in lines:
                            if line.strip().startswith('var data'):
                                # Extract the JSON data
                                json_str = line.strip().replace('var data = ', '').rstrip(';')
                                try:
                                    data = json.loads(json_str)
                                    benchmark_data['visualizations'][viz_name] = data
                                    break
                                except:
                                    pass
            except Exception as e:
                print(f"Error extracting data from {viz_path}: {e}")
    
    # Also try to extract metrics from index.html if it exists
    index_path = os.path.join(model_dir, 'index.html')
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract overall improvement
            improvement_span = soup.select('.improvement')
            if improvement_span:
                overall_improvement = improvement_span[0].text.strip('%').strip()
                benchmark_data['overall_improvement'] = float(overall_improvement)
            
            # Extract metrics from table
            metrics_table = {}
            table_rows = soup.select('table tr')[1:]  # Skip header row
            
            for row in table_rows:
                cells = row.select('td')
                if len(cells) >= 4:
                    metric_name = cells[0].text.strip()
                    baseline_value = float(cells[1].text.strip())
                    episodic_value = float(cells[2].text.strip())
                    improvement = cells[3].text.strip('%').strip()
                    # Remove + sign if present
                    if improvement.startswith('+'):
                        improvement = improvement[1:]
                    improvement = float(improvement)
                    
                    metrics_table[metric_name] = {
                        'baseline': baseline_value,
                        'episodic': episodic_value,
                        'improvement': improvement
                    }
            
            benchmark_data['metrics_table'] = metrics_table
        except Exception as e:
            print(f"Error extracting metrics from index.html: {e}")
    
    return benchmark_data

def extract_real_benchmark_data(model_name, output_dir='visualizations'):
    """Extract real benchmark data from visualization files"""
    # First check if detailed results file exists
    detailed_results_files = []
    for file in os.listdir(output_dir):
        if file.startswith('benchmark_detailed_results_') and file.endswith('.txt'):
            detailed_results_files.append(os.path.join(output_dir, file))
    
    # Sort by modification time to get the most recent one
    if detailed_results_files:
        detailed_results_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        detailed_results_path = detailed_results_files[0]
        
        # Extract data from the detailed results file
        try:
            model_results = extract_metrics_from_detailed_results(detailed_results_path)
            if model_name in model_results:
                return {
                    'model': model_name,
                    'visualizations': create_visualizations_from_metrics(model_results[model_name]),
                    'metrics_table': model_results[model_name]['metrics_table'],
                    'overall_improvement': model_results[model_name].get('overall_improvement', 0)
                }
        except Exception as e:
            print(f"Error extracting metrics from detailed results: {e}")
    
    # Fall back to extracting from HTML files
    return extract_data_from_html_files(model_name, output_dir)

def extract_metrics_from_detailed_results(file_path):
    """Extract metrics from detailed results text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        model_sections = re.split(r'={10,}\s+BENCHMARKING\s+([\w\.\-]+)\s+={10,}', content)
        
        model_results = {}
        current_model = None
        
        for i, section in enumerate(model_sections):
            if i == 0:  # Skip the file header
                continue
                
            if i % 2 == 1:  # This is a model name
                current_model = section.strip()
                model_results[current_model] = {
                    'metrics_table': {},
                    'baseline': {},
                    'episodic': {}
                }
            else:  # This is the model's content
                if current_model:
                    # Extract Baseline Metrics
                    baseline_match = re.search(r'Baseline Metrics:(.*?)(?=Episodic Memory Metrics:)', section, re.DOTALL)
                    if baseline_match:
                        baseline_text = baseline_match.group(1).strip()
                        metrics = parse_metrics_section(baseline_text)
                        model_results[current_model]['baseline'] = metrics
                    
                    # Extract Episodic Memory Metrics
                    episodic_match = re.search(r'Episodic Memory Metrics:(.*?)(?=Improvements:)', section, re.DOTALL)
                    if episodic_match:
                        episodic_text = episodic_match.group(1).strip()
                        metrics = parse_metrics_section(episodic_text)
                        model_results[current_model]['episodic'] = metrics
                    
                    # Extract Improvements
                    improvements_match = re.search(r'Improvements:(.*?)(?=\n\n|$)', section, re.DOTALL)
                    if improvements_match:
                        improvements_text = improvements_match.group(1).strip()
                        improvements = {}
                        
                        for line in improvements_text.split('\n'):
                            if ':' in line:
                                key, value = line.split(':', 1)
                                key = key.strip()
                                value = value.strip().rstrip('%')
                                if value.startswith('+'):
                                    value = value[1:]
                                try:
                                    improvements[key] = float(value)
                                except ValueError:
                                    continue
                        
                        # Calculate overall improvement
                        if improvements:
                            overall_improvement = sum(improvements.values()) / len(improvements)
                            model_results[current_model]['overall_improvement'] = overall_improvement
                        
                        # Create metrics table
                        metrics_table = {}
                        for key in model_results[current_model]['baseline'].keys():
                            if key in model_results[current_model]['episodic'] and key in improvements:
                                metrics_table[key] = {
                                    'baseline': model_results[current_model]['baseline'][key],
                                    'episodic': model_results[current_model]['episodic'][key],
                                    'improvement': improvements[key]
                                }
                        
                        model_results[current_model]['metrics_table'] = metrics_table
        
        return model_results
    
    except Exception as e:
        print(f"Error parsing detailed results file: {e}")
        return {}

def parse_metrics_section(text):
    """Parse metrics from a text section"""
    metrics = {}
    for line in text.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
    return metrics

def create_visualizations_from_metrics(model_data):
    """Create visualization data structures from extracted metrics"""
    visualizations = {}
    
    # Comprehensive comparison
    if 'baseline' in model_data and 'episodic' in model_data:
        metrics = []
        traditional = []
        episodic = []
        
        for key in model_data['metrics_table'].keys():
            metrics.append(key)
            traditional.append(model_data['metrics_table'][key]['baseline'])
            episodic.append(model_data['metrics_table'][key]['episodic'])
        
        visualizations['comprehensive_comparison'] = {
            'metrics': metrics,
            'traditional': traditional,
            'episodic': episodic
        }
    
    # Improvement summary
    if 'metrics_table' in model_data:
        metrics = []
        improvements = []
        
        for key, values in model_data['metrics_table'].items():
            metrics.append(key)
            improvements.append(values['improvement'])
        
        visualizations['improvement_summary'] = {
            'x': metrics,
            'y': improvements
        }
    
    return visualizations

def create_cross_model_comparison(output_dir='visualizations'):
    """Create comparison visualizations across all models"""
    # Get all model directories
    model_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item != 'all_models':
            model_dirs.append(item)
    
    if not model_dirs:
        return {}
    
    # Create all_models directory if it doesn't exist
    all_models_dir = os.path.join(output_dir, 'all_models')
    os.makedirs(all_models_dir, exist_ok=True)
    
    # Extract data for each model
    model_data = {}
    for model in model_dirs:
        data = extract_real_benchmark_data(model, output_dir)
        model_data[model] = data
    
    # Create comparison visualizations
    comparisons = {}
    
    # Create metrics comparison
    metrics_comparison = create_metrics_comparison(model_data, all_models_dir)
    comparisons['metrics_comparison'] = metrics_comparison
    
    # Create overall improvement comparison
    improvement_comparison = create_improvement_comparison(model_data, all_models_dir)
    comparisons['improvement_comparison'] = improvement_comparison
    
    return comparisons

def create_metrics_comparison(model_data, output_dir):
    """Create a comparison of metrics across all models"""
    # Common metrics across all models
    common_metrics = set()
    for model, data in model_data.items():
        if 'metrics_table' in data:
            common_metrics.update(data['metrics_table'].keys())
    
    common_metrics = list(common_metrics)
    
    # For each common metric, compare across models
    for metric in common_metrics:
        # Create a bar chart comparing this metric across models
        plt.figure(figsize=(12, 8))
        plt.style.use('ggplot')
        
        models = []
        baseline_values = []
        episodic_values = []
        
        for model, data in model_data.items():
            if 'metrics_table' in data and metric in data['metrics_table']:
                models.append(model)
                baseline_values.append(data['metrics_table'][metric]['baseline'])
                episodic_values.append(data['metrics_table'][metric]['episodic'])
        
        # Set up the bar positions
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        rects1 = ax.bar(x - width/2, baseline_values, width, label='Traditional LLM', color='#636EFA')
        rects2 = ax.bar(x + width/2, episodic_values, width, label='Episodic Memory', color='#EF553B')
        
        # Add labels and title
        metric_title = metric.replace('_', ' ').title()
        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel(f'{metric_title}', fontweight='bold')
        ax.set_title(f'{metric_title} Comparison Across Models', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        
        # Add data labels
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')
        
        add_labels(rects1)
        add_labels(rects2)
        
        # Add improvement percentages
        for i in range(len(models)):
            if baseline_values[i] > 0:
                improvement = ((episodic_values[i] - baseline_values[i]) / baseline_values[i]) * 100
                color = 'green' if improvement > 0 else 'red'
                sign = '+' if improvement > 0 else ''
                ax.annotate(f'{sign}{improvement:.1f}%',
                            xy=(x[i] + width/2, episodic_values[i]),
                            xytext=(0, 10),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color=color, fontweight='bold')
        
        plt.tight_layout()
        
        # Create images directory if it doesn't exist
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Save the figure to a file
        metric_filename = metric.lower().replace(' ', '_')
        img_path = os.path.join(images_dir, f"{metric_filename}_comparison.png")
        plt.savefig(img_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create HTML for all metrics comparisons
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cross-Model Metrics Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1, h2 { color: #333; }
            .summary { margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
            .vis-container { margin: 20px 0; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Cross-Model Metrics Comparison</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>This page shows comparisons of various metrics across all benchmarked models.</p>
            <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        <h2>Metric Comparisons</h2>
    """
    
    for metric in common_metrics:
        metric_title = metric.replace('_', ' ').title()
        metric_filename = metric.lower().replace(' ', '_')
        html_content += f"""
        <div class="vis-container">
            <h3>{metric_title} Comparison</h3>
            <img src="images/{metric_filename}_comparison.png" alt="{metric_title} Comparison">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(os.path.join(output_dir, 'metrics_comparison.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return common_metrics

def create_improvement_comparison(model_data, output_dir):
    """Create a comparison of overall improvements across all models"""
    plt.figure(figsize=(12, 8))
    plt.style.use('ggplot')
    
    models = []
    improvements = []
    
    for model, data in model_data.items():
        if 'overall_improvement' in data:
            models.append(model)
            improvements.append(data['overall_improvement'])
    
    if not models:
        return
    
    # Create a bar chart
    colors = ['#2ecc71' if imp >= 0 else '#e74c3c' for imp in improvements]
    bars = plt.bar(models, improvements, color=colors, edgecolor='#333333', alpha=0.8)
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        color = 'green' if height >= 0 else 'red'
        sign = '+' if height >= 0 else ''
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{sign}{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', color=color)
    
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Overall Improvement (%)', fontweight='bold')
    plt.title('Overall Improvement Comparison Across Models', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='#7f8c8d', linestyle='-', alpha=0.3)
    
    # Create images directory if it doesn't exist
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Save the figure to a file
    img_path = os.path.join(images_dir, f"overall_improvement_comparison.png")
    plt.savefig(img_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create HTML file
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Overall Improvement Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1, h2 { color: #333; }
            .summary { margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
            .vis-container { margin: 20px 0; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>Overall Improvement Comparison</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>This visualization shows the overall improvement percentage for each model when using episodic memory.</p>
            <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        <div class="vis-container">
            <img src="images/overall_improvement_comparison.png" alt="Overall Improvement Comparison">
        </div>
        
        <h2>Model Rankings</h2>
        <table border="1" cellpadding="5" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Overall Improvement</th>
            </tr>
    """
    
    # Add model rankings
    model_rankings = sorted(zip(models, improvements), key=lambda x: x[1], reverse=True)
    for i, (model, improvement) in enumerate(model_rankings, 1):
        sign = '+' if improvement >= 0 else ''
        color = 'green' if improvement >= 0 else 'red'
        html_content += f"""
            <tr>
                <td>{i}</td>
                <td>{model}</td>
                <td style="color: {color};">{sign}{improvement:.2f}%</td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(os.path.join(output_dir, 'overall_improvement_comparison.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Create index.html for all_models
    create_all_models_index(output_dir, models)
    
    return models

def create_all_models_index(output_dir, models):
    """Create an index.html file for the all_models directory"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>All Models Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1, h2 { color: #333; }
            .summary { margin: 20px 0; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }
            .nav-links { margin: 20px 0; }
            .nav-links a { margin-right: 20px; text-decoration: none; color: #3498db; font-weight: bold; }
            .nav-links a:hover { text-decoration: underline; }
            .model-list { margin: 20px 0; }
            .model-list a { display: block; margin: 10px 0; text-decoration: none; color: #333; }
            .model-list a:hover { text-decoration: underline; color: #3498db; }
        </style>
    </head>
    <body>
        <h1>All Models Comparison</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>This page provides access to comparisons across all benchmarked models.</p>
            <p><strong>Generated:</strong> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
        
        <div class="nav-links">
            <a href="overall_improvement_comparison.html">Overall Improvement Comparison</a>
            <a href="metrics_comparison.html">Metrics Comparison</a>
        </div>
        
        <h2>Individual Model Results</h2>
        <div class="model-list">
    """
    
    for model in models:
        html_content += f"""
            <a href="../{model}/index.html">{model}</a>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

def create_capabilities_radar(data, title, output_dir=None, viz_name=None):
    """Create a radar chart for capabilities"""
    plt.figure(figsize=(10, 8))
    plt.style.use('ggplot')
    
    # Extract data
    theta = data.get('theta', [])
    r = data.get('r', [])
    
    # Convert to radians and duplicate first point to close the loop
    theta_rad = np.radians(np.linspace(0, 360, len(theta), endpoint=False))
    theta_rad = np.append(theta_rad, theta_rad[0])
    r = np.append(r, r[0])
    
    # Create radar plot
    ax = plt.subplot(111, polar=True)
    ax.plot(theta_rad, r, 'o-', linewidth=2, color='#3498db')
    ax.fill(theta_rad, r, alpha=0.25, color='#3498db')
    ax.set_xticks(theta_rad[:-1])
    ax.set_xticklabels(theta, fontweight='bold')
    
    # Add data labels
    for t, rv in zip(theta_rad[:-1], r[:-1]):
        plt.text(t, rv + 0.05, f'{rv:.2f}', ha='center', va='center', fontweight='bold')
    
    plt.title(title, fontsize=14, fontweight='bold', y=1.1)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    
    # Save to a file if output_dir and viz_name are provided
    if output_dir and viz_name:
        # Create images directory if it doesn't exist
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Save the figure to a file
        img_path = os.path.join(images_dir, f"{viz_name}.png")
        plt.savefig(img_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return the relative path to the image
        return os.path.join('images', f"{viz_name}.png")

def create_comprehensive_comparison(data, title, output_dir=None, viz_name=None):
    """Create a bar chart for comprehensive comparison"""
    plt.figure(figsize=(12, 7))
    plt.style.use('ggplot')
    
    # Extract data
    metrics = data.get('metrics', [])
    traditional = data.get('traditional', [])
    episodic = data.get('episodic', [])
    
    # Set up the bar positions
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create the bars
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, traditional, width, label='Traditional LLM', color='#636EFA', edgecolor='#3A3BFA')
    rects2 = ax.bar(x + width/2, episodic, width, label='Episodic Memory', color='#EF553B', edgecolor='#CF3B21')
    
    # Add labels and title
    ax.set_xlabel('Metrics', fontweight='bold')
    ax.set_ylabel('Score (0-1)', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    
    # Add data labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
    
    add_labels(rects1)
    add_labels(rects2)
    
    # Add improvement percentages
    for i in range(len(metrics)):
        if traditional[i] > 0:
            improvement = ((episodic[i] - traditional[i]) / traditional[i]) * 100
            if improvement > 0:
                ax.annotate(f'+{improvement:.1f}%',
                            xy=(x[i] + width/2, episodic[i]),
                            xytext=(0, 10),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color='green', fontweight='bold')
    
    # Set y-axis limits
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save to a file if output_dir and viz_name are provided
    if output_dir and viz_name:
        # Create images directory if it doesn't exist
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Save the figure to a file
        img_path = os.path.join(images_dir, f"{viz_name}.png")
        plt.savefig(img_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return the relative path to the image
        return os.path.join('images', f"{viz_name}.png")

def create_bar_chart(data, title, output_dir=None, viz_name=None):
    """Create a simple bar chart"""
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    
    # Extract data
    x = data.get('x', [])
    y = data.get('y', [])
    
    # Create a colorful bar chart
    bars = plt.bar(x, y, color='#3498db', edgecolor='#2980b9', alpha=0.8)
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Metrics', fontweight='bold')
    plt.ylabel('Values', fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save to a file if output_dir and viz_name are provided
    if output_dir and viz_name:
        # Create images directory if it doesn't exist
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Save the figure to a file
        img_path = os.path.join(images_dir, f"{viz_name}.png")
        plt.savefig(img_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return the relative path to the image
        return os.path.join('images', f"{viz_name}.png")

def create_response_time_comparison(data, title, output_dir=None, viz_name=None):
    """Create a bar chart for response time comparison"""
    plt.figure(figsize=(12, 7))
    plt.style.use('ggplot')
    
    # Extract data
    x = data.get('x', [])
    traditional = data.get('traditional', [])
    episodic = data.get('episodic', [])
    
    # Set up the bar positions
    x_pos = np.arange(len(x))
    width = 0.35
    
    # Create the bars
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x_pos - width/2, traditional, width, label='Traditional LLM', color='#636EFA', edgecolor='#3A3BFA')
    rects2 = ax.bar(x_pos + width/2, episodic, width, label='Episodic Memory', color='#EF553B', edgecolor='#CF3B21')
    
    # Add labels and title
    ax.set_xlabel('Query Types', fontweight='bold')
    ax.set_ylabel('Response Time (ms)', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=45, ha='right')
    ax.legend()
    
    # Add data labels
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}ms',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
    
    add_labels(rects1)
    add_labels(rects2)
    
    # Add percentage differences
    for i in range(len(x)):
        if traditional[i] > 0:
            diff = ((episodic[i] - traditional[i]) / traditional[i]) * 100
            color = 'red' if diff > 0 else 'green'
            sign = '+' if diff > 0 else ''
            ax.annotate(f'{sign}{diff:.1f}%',
                        xy=(x_pos[i] + width/2, episodic[i]),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color=color, fontweight='bold')
    
    plt.tight_layout()
    
    # Save to a file if output_dir and viz_name are provided
    if output_dir and viz_name:
        # Create images directory if it doesn't exist
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Save the figure to a file
        img_path = os.path.join(images_dir, f"{viz_name}.png")
        plt.savefig(img_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return the relative path to the image
        return os.path.join('images', f"{viz_name}.png")

def create_improvement_summary(data, title, output_dir=None, viz_name=None):
    """Create a bar chart for improvement summary"""
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    
    # Extract data
    x = data.get('x', [])
    y = data.get('y', [])
    
    # Create a colorful bar chart with different colors based on value
    colors = ['#2ecc71' if val >= 0 else '#e74c3c' for val in y]
    bars = plt.bar(x, y, color=colors, edgecolor='#333333', alpha=0.8)
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        color = 'green' if height >= 0 else 'red'
        sign = '+' if height >= 0 else ''
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{sign}{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', color=color)
    
    plt.xlabel('Metrics', fontweight='bold')
    plt.ylabel('Improvement (%)', fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='#7f8c8d', linestyle='-', alpha=0.3)
    
    # Save to a file if output_dir and viz_name are provided
    if output_dir and viz_name:
        # Create images directory if it doesn't exist
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Save the figure to a file
        img_path = os.path.join(images_dir, f"{viz_name}.png")
        plt.savefig(img_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return the relative path to the image
        return os.path.join('images', f"{viz_name}.png")

def markdown_to_html(text):
    """Convert markdown text to HTML"""
    # Convert headers
    text = re.sub(r'##\s+(.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'###\s+(.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    
    # Convert bold text
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert italic text
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    # Convert numbered lists
    text = re.sub(r'^\d+\.\s+(.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li>.*?</li>\n)+', r'<ol>\n\g<0></ol>', text, flags=re.DOTALL)
    
    # Convert bullet lists
    text = re.sub(r'^\*\s+(.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(<li>.*?</li>\n)+', r'<ul>\n\g<0></ul>', text, flags=re.DOTALL)
    
    # Convert paragraphs
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Replace single newlines with spaces
    text = re.sub(r'\n\n+', '</p>\n\n<p>', text)  # Replace double newlines with paragraph breaks
    text = '<p>' + text + '</p>'  # Wrap in paragraphs
    
    return text

def analyze_with_groq(api_key, model_name, benchmark_data):
    """Use Groq to analyze the benchmark results"""
    client = Groq(api_key=api_key)
    
    # Prepare the prompt
    prompt = f"""
    You are an AI performance analyst. Analyze the following benchmark results comparing traditional LLM with Episodic Memory enhanced LLM.
    
    Benchmark Data:
    {json.dumps(benchmark_data, indent=2)}
    
    Please provide a comprehensive analysis covering:
    1. Overall performance comparison
    2. Key strengths of the Episodic Memory approach
    3. Areas where Episodic Memory shows the most improvement
    4. Potential limitations or areas for further improvement
    5. Recommendations for optimal use cases
    
    Format your response as a professional report with clear sections and insights.
    Use proper HTML formatting in your response, with <h2> for main sections, <h3> for subsections, <p> for paragraphs, 
    <ul>/<li> for bullet points, and <ol>/<li> for numbered lists. Do not use markdown formatting.
    """
    
    # Generate the analysis
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI performance analyst providing detailed insights on benchmark results. Format your response with proper HTML tags."},
            {"role": "user", "content": prompt}
        ],
        model=model_name,
        temperature=0.3,
        max_tokens=2000
    )
    
    return completion.choices[0].message.content

def generate_report(output_dir, model_name, output_file='benchmark_report.html'):
    """Generate a comprehensive report on the benchmark results"""
    # Load benchmark data
    benchmark_data = load_benchmark_data(model_name, output_dir)
    
    # Get Groq API key from environment
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is not set")
        return
    
    # Analyze the data with Groq
    analysis = analyze_with_groq(api_key, "llama3-8b-8192", benchmark_data)
    
    # Create visualizations and save them as files
    visualizations = {}
    
    # Process each visualization type
    viz_data = benchmark_data.get('visualizations', {})
    
    # Capabilities Radar
    if 'capabilities_radar' in viz_data:
        img_path = create_capabilities_radar(
            viz_data['capabilities_radar'],
            "Capabilities Radar",
            output_dir=output_dir,
            viz_name="capabilities_radar"
        )
        visualizations['capabilities_radar'] = img_path
    
    # Comprehensive Comparison
    if 'comprehensive_comparison' in viz_data:
        img_path = create_comprehensive_comparison(
            viz_data['comprehensive_comparison'],
            "Comprehensive Comparison",
            output_dir=output_dir,
            viz_name="comprehensive_comparison"
        )
        visualizations['comprehensive_comparison'] = img_path
    
    # Entity Recall Comparison
    if 'entity_recall_comparison' in viz_data:
        img_path = create_bar_chart(
            viz_data['entity_recall_comparison'],
            "Entity Recall Comparison",
            output_dir=output_dir,
            viz_name="entity_recall_comparison"
        )
        visualizations['entity_recall_comparison'] = img_path
    
    # Memory Accuracy
    if 'memory_accuracy' in viz_data:
        img_path = create_bar_chart(
            viz_data['memory_accuracy'],
            "Memory Accuracy",
            output_dir=output_dir,
            viz_name="memory_accuracy"
        )
        visualizations['memory_accuracy'] = img_path
    
    # Response Time Comparison
    if 'response_time_comparison' in viz_data:
        img_path = create_response_time_comparison(
            viz_data['response_time_comparison'],
            "Response Time Comparison",
            output_dir=output_dir,
            viz_name="response_time_comparison"
        )
        visualizations['response_time_comparison'] = img_path
    
    # Improvement Summary
    if 'improvement_summary' in viz_data:
        img_path = create_improvement_summary(
            viz_data['improvement_summary'],
            "Improvement Summary",
            output_dir=output_dir,
            viz_name="improvement_summary"
        )
        visualizations['improvement_summary'] = img_path
    
    # Read the template file
    with open('report_template.html', 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Replace placeholders
    html_report = template.replace('MODEL_NAME_PLACEHOLDER', model_name)
    html_report = html_report.replace('TIMESTAMP_PLACEHOLDER', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Format analysis text if it doesn't already contain HTML tags
    if '<h2>' not in analysis:
        analysis = markdown_to_html(analysis)
    html_report = html_report.replace('ANALYSIS_PLACEHOLDER', analysis)
    
    # Create visualizations HTML
    visualizations_html = ""
    for viz_name, img_path in visualizations.items():
        visualizations_html += f"""
        <div class="visualization">
            <h3>{viz_name.replace('_', ' ').title()}</h3>
            <img src="{img_path}" alt="{viz_name}" width="800">
        </div>
        """
    
    html_report = html_report.replace('VISUALIZATIONS_PLACEHOLDER', visualizations_html)
    
    # Write the report to file
    report_path = os.path.join(output_dir, output_file)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"Report generated at: {report_path}")
    return report_path

def print_summary(benchmark_data):
    """Print a summary of the benchmark results."""
    print("\n===== BENCHMARK SUMMARY =====")
    print(f"Model: {benchmark_data.get('model', 'Unknown')}")
    
    visualizations = benchmark_data.get('visualizations', {})
    
    if 'memory_accuracy' in visualizations:
        memory_data = visualizations['memory_accuracy']
        print("\nMemory Accuracy:")
        print(f"  Standard LLM: {memory_data.get('standard_accuracy', 0):.2f}%")
        print(f"  Memory-Enhanced: {memory_data.get('memory_accuracy', 0):.2f}%")
        print(f"  Improvement: {memory_data.get('improvement', 0):.2f}%")
    
    if 'response_time_comparison' in visualizations:
        time_data = visualizations['response_time_comparison']
        print("\nResponse Time:")
        print(f"  Standard LLM: {time_data.get('standard_time', 0):.2f}s")
        print(f"  Memory-Enhanced: {time_data.get('memory_time', 0):.2f}s")
        print(f"  Overhead: {time_data.get('overhead', 0):.2f}s ({time_data.get('overhead_percent', 0):.2f}%)")
    
    if 'entity_recall_comparison' in visualizations:
        entity_data = visualizations['entity_recall_comparison']
        print("\nEntity Recall:")
        print(f"  Standard LLM: {entity_data.get('standard_recall', 0):.2f}%")
        print(f"  Memory-Enhanced: {entity_data.get('memory_recall', 0):.2f}%")
        print(f"  Improvement: {entity_data.get('improvement', 0):.2f}%")
    
    print("\nTo view detailed visualizations, run with --generate-report flag")
    print("===============================")

def main(model_name=None, output_dir=None, generate_report_flag=False):
    """Main entry point for the benchmark analysis tool."""
    parser = argparse.ArgumentParser(description='Analyze benchmark results and generate a report')
    parser.add_argument('--model', type=str, default=model_name or os.environ.get('MODEL_NAME', 'llama3-8b-8192'),
                        help='Model name to analyze')
    parser.add_argument('--output-dir', type=str, default=output_dir or 'visualizations',
                        help='Directory containing visualization outputs')
    parser.add_argument('--output-file', type=str, default='benchmark_report.html',
                        help='Output file name for the report')
    parser.add_argument('--generate-report', action='store_true', default=generate_report_flag,
                        help='Generate a comprehensive report')
    
    args = parser.parse_args()
    
    if args.generate_report:
        report_path = generate_report(args.output_dir, args.model, args.output_file)
        
        # Open the report in the default browser
        if report_path:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
    else:
        print(f"Analyzing benchmark results for model: {args.model}")
        benchmark_data = load_benchmark_data(args.model, args.output_dir)
        print_summary(benchmark_data)

if __name__ == "__main__":
    main() 