import os
import argparse

def create_index_html(output_dir: str):
    """Create an index.html file for easy navigation of visualizations"""
    # Get all model directories
    model_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            model_dirs.append(item)
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Episodic Memory Benchmarking Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                line-height: 1.6;
            }
            h1, h2, h3 {
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .card {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .model-section {
                margin-bottom: 30px;
            }
            .viz-links {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            .viz-link {
                background-color: #f8f9fa;
                padding: 10px 15px;
                border-radius: 4px;
                text-decoration: none;
                color: #0366d6;
                transition: background-color 0.2s;
            }
            .viz-link:hover {
                background-color: #e9ecef;
            }
            .header {
                background-color: #24292e;
                color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 8px;
            }
            .no-results {
                background-color: #fff3cd;
                color: #856404;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Episodic Memory Benchmarking Results</h1>
                <p>Comparing traditional LLM with Episodic Memory Module performance</p>
            </div>
    """
    
    # Add model comparison link if it exists
    if os.path.exists(os.path.join(output_dir, "model_comparison.html")):
        html_content += """
            <div class="card">
                <h2>Model Comparison</h2>
                <div class="viz-links">
                    <a class="viz-link" href="model_comparison.html">View Cross-Model Comparison</a>
                </div>
            </div>
        """
    
    # Check if we have any model directories
    if not model_dirs:
        # Check if we have any HTML files directly in the output directory
        html_files = [f for f in os.listdir(output_dir) if f.endswith('.html') and f != 'index.html']
        
        if html_files:
            html_content += """
            <div class="card">
                <h2>Benchmark Results</h2>
                <div class="viz-links">
            """
            
            for html_file in html_files:
                # Create a nice display name from the filename
                display_name = html_file.replace('.html', '').replace('_', ' ').title()
                html_content += f'        <a class="viz-link" href="{html_file}">{display_name}</a>\n'
            
            html_content += """
                </div>
            </div>
            """
        else:
            html_content += """
            <div class="no-results">
                <h2>No benchmark results found</h2>
                <p>No visualization files were found. Please run the benchmarks first.</p>
            </div>
            """
    else:
        # Add sections for each model
        for model in model_dirs:
            model_path = os.path.join(output_dir, model)
            viz_files = [f for f in os.listdir(model_path) if f.endswith('.html')]
            
            if viz_files:
                html_content += f"""
                <div class="card model-section">
                    <h2>Model: {model}</h2>
                    <div class="viz-links">
                """
                
                for viz_file in viz_files:
                    # Create a nice display name from the filename
                    display_name = viz_file.replace('.html', '').replace('_', ' ').title()
                    html_content += f'        <a class="viz-link" href="{model}/{viz_file}">{display_name}</a>\n'
                
                html_content += """
                    </div>
                </div>
                """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html_content)
    
    print(f"Index file created at '{output_dir}/index.html'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create index.html for visualization navigation')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory containing visualization outputs')
    args = parser.parse_args()
    
    create_index_html(args.output_dir) 