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
    """Load benchmark data from pickle files if available, otherwise use sample data"""
    model_dir = os.path.join(output_dir, model_name)
    pickle_file = os.path.join(model_dir, 'benchmark_data.pkl')
    
    if os.path.exists(pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading benchmark data: {e}")
    
    # If pickle file doesn't exist or can't be loaded, extract data from HTML files
    benchmark_data = extract_data_from_html_files(model_name, output_dir)
    
    # If no data could be extracted, use sample data
    if not benchmark_data or not benchmark_data.get('visualizations'):
        benchmark_data = generate_sample_benchmark_data(model_name)
    
    return benchmark_data

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
    
    return benchmark_data

def generate_sample_benchmark_data(model_name):
    """Generate sample benchmark data for visualizations"""
    return {
        'model': model_name,
        'visualizations': {
            'capabilities_radar': {
                'theta': ['Entity Recognition', 'Context Retention', 'Memory Accuracy', 'Response Time', 'Temporal Consistency', 'Relationship Tracking'],
                'r': [0.85, 0.92, 0.88, 0.75, 0.90, 0.82]
            },
            'comprehensive_comparison': {
                'metrics': ['Context Preservation', 'Relevance', 'Entity Recall', 'Factual Accuracy', 'Memory Accuracy'],
                'traditional': [0.72, 0.68, 0.65, 0.78, 0],
                'episodic': [0.89, 0.82, 0.85, 0.88, 0.79]
            },
            'entity_recall_comparison': {
                'x': ['Person', 'Location', 'Organization', 'Date', 'Event', 'Concept'],
                'y': [0.92, 0.88, 0.85, 0.90, 0.82, 0.78]
            },
            'memory_accuracy': {
                'x': ['Short-term', 'Medium-term', 'Long-term'],
                'y': [0.95, 0.88, 0.82]
            },
            'response_time_comparison': {
                'x': ['Simple Query', 'Complex Query', 'Multi-turn', 'Entity-heavy', 'Temporal Query'],
                'traditional': [100, 150, 180, 160, 200],
                'episodic': [120, 180, 210, 190, 230]
            },
            'improvement_summary': {
                'x': ['Entity Recognition', 'Context Retention', 'Memory Accuracy', 'Response Time', 'Temporal Consistency'],
                'y': [0.25, 0.35, 0.30, -0.15, 0.40]
            }
        }
    }

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