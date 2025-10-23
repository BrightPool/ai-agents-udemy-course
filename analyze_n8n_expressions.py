#!/usr/bin/env python3
"""
Script to analyze n8n workflow JSON files and extract dynamic expressions.
Creates CSV reports for each agent workflow showing expression usage.
"""

import json
import os
import re
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple


class N8nExpressionAnalyzer:
    def __init__(self):
        # Pattern to match n8n expressions: {{ ... }} or expressions starting with =
        self.expression_patterns = [
            re.compile(r'\{\{(.*?)\}\}', re.DOTALL),  # {{ ... }} expressions
            re.compile(r'^=(.+)', re.DOTALL)  # expressions starting with =
        ]
    
    def extract_expressions_from_node(self, node: Dict[str, Any], node_name: str) -> List[Tuple[str, str, str]]:
        """Extract expressions from a single n8n node."""
        expressions = []
        
        def process_value(value: Any, context_path: str) -> None:
            """Recursively process values to find expressions."""
            if isinstance(value, str):
                # Check for {{ }} expressions
                matches = self.expression_patterns[0].findall(value)
                for match in matches:
                    expression_name = f"{node_name}: {context_path}"
                    expression_code = "{{ " + match.strip() + " }}"
                    explanation = self.explain_expression(match.strip(), node, context_path)
                    expressions.append((expression_name, expression_code, explanation))
                
                # Check for = expressions
                if value.startswith('='):
                    expression_name = f"{node_name}: {context_path}"
                    expression_code = value
                    explanation = self.explain_expression(value[1:], node, context_path)
                    expressions.append((expression_name, expression_code, explanation))
            
            elif isinstance(value, dict):
                for key, sub_value in value.items():
                    new_context = f"{context_path}.{key}" if context_path else key
                    process_value(sub_value, new_context)
            
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    new_context = f"{context_path}[{i}]" if context_path else f"[{i}]"
                    process_value(item, new_context)
        
        # Process node parameters
        if 'parameters' in node:
            process_value(node['parameters'], 'parameters')
        
        return expressions
    
    def explain_expression(self, expression: str, node: Dict[str, Any], context: str) -> str:
        """Generate explanation for an n8n expression."""
        expression = expression.strip()
        node_type = node.get('type', '').split('.')[-1] if node.get('type') else 'unknown'
        
        # Common expression patterns and their explanations
        if '$json' in expression and not '$(' in expression:
            if expression == '$json':
                return f"Accesses the entire JSON data from the previous node in the {node_type} node"
            elif '.' in expression:
                field = expression.split('.')[-1]
                return f"Retrieves the '{field}' field from the JSON data of the previous node"
            else:
                return f"Accesses JSON data from the previous node: {expression}"
        
        elif expression.startswith("$('") and "').item.json" in expression:
            # Pattern: $('NodeName').item.json.field
            match = re.search(r"\$\('([^']+)'\)\.item\.json(?:\.(.+))?", expression)
            if match:
                node_name = match.group(1)
                field = match.group(2) if match.group(2) else 'entire object'
                return f"Fetches data from the '{node_name}' node, specifically accessing the {field} field"
            else:
                return f"References data from a specific node: {expression}"
        
        elif expression.startswith("$('") and "').first().json" in expression:
            # Pattern: $('NodeName').first().json.field
            match = re.search(r"\$\('([^']+)'\)\.first\(\)\.json(?:\.(.+))?", expression)
            if match:
                node_name = match.group(1)
                field = match.group(2) if match.group(2) else 'entire object'
                return f"Gets the first item from the '{node_name}' node results, accessing the {field} field"
            else:
                return f"References first item from a specific node: {expression}"
        
        elif 'split(' in expression and 'length' in expression:
            if 'word' in expression.lower() or "split(' ')" in expression:
                return "Calculates word count by splitting text on spaces and counting the resulting array length"
            else:
                return "Counts elements by splitting text and measuring array length"
        
        elif 'Math.ceil(' in expression:
            return "Performs mathematical ceiling calculation, likely for reading time estimation based on word count"
        
        elif expression.startswith('$json['):
            # Form field access pattern
            match = re.search(r'\$json\[\'([^\']+)\'\]', expression)
            if match:
                field_name = match.group(1)
                return f"Retrieves user input from the form field labeled '{field_name}'"
            else:
                return f"Accesses form data using bracket notation: {expression}"
        
        elif '||' in expression:
            return f"Uses fallback logic - tries multiple data sources in order: {expression}"
        
        elif context and 'prompt' in context.lower():
            return f"Dynamic content injected into AI agent prompt: {expression}"
        
        elif context and ('message' in context.lower() or 'content' in context.lower()):
            return f"Message or content field populated with: {expression}"
        
        elif context and 'value' in context.lower():
            return f"Sets a variable value using: {expression}"
        
        else:
            return f"Dynamic expression in {node_type} node: {expression}"
    
    def analyze_workflow(self, json_file_path: str) -> List[Tuple[str, str, str]]:
        """Analyze a complete n8n workflow JSON file."""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
            
            all_expressions = []
            
            if 'nodes' in workflow_data:
                for node in workflow_data['nodes']:
                    node_name = node.get('name', 'Unnamed Node')
                    node_expressions = self.extract_expressions_from_node(node, node_name)
                    all_expressions.extend(node_expressions)
            
            return all_expressions
        
        except Exception as e:
            print(f"Error analyzing {json_file_path}: {str(e)}")
            return []
    
    def generate_csv_report(self, expressions: List[Tuple[str, str, str]], output_file: str):
        """Generate CSV report for expressions."""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Expression Name', 'Expression', 'Detailed Explanation'])
            
            for expression_name, expression_code, explanation in expressions:
                writer.writerow([expression_name, expression_code, explanation])


def main():
    """Main function to process all n8n workflow files."""
    n8n_dir = "/Users/rhysfishernewairblack/Documents/GitHub/ai-agents-udemy-course/n8n/ready-to-publish"
    output_dir = "/Users/rhysfishernewairblack/Documents/GitHub/ai-agents-udemy-course/n8n_expression_analysis"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = N8nExpressionAnalyzer()
    
    # Get all JSON files, excluding joke generator
    json_files = []
    for file in os.listdir(n8n_dir):
        if file.endswith('.json') and 'joke generator' not in file.lower():
            json_files.append(file)
    
    print(f"Found {len(json_files)} workflow files to analyze (excluding joke generator)")
    
    for json_file in json_files:
        print(f"\nAnalyzing: {json_file}")
        
        json_path = os.path.join(n8n_dir, json_file)
        expressions = analyzer.analyze_workflow(json_path)
        
        if expressions:
            # Generate output filename
            base_name = os.path.splitext(json_file)[0]
            output_file = os.path.join(output_dir, f"{base_name}_expressions.csv")
            
            analyzer.generate_csv_report(expressions, output_file)
            print(f"  ✓ Found {len(expressions)} expressions → {output_file}")
            
            # Also print to console for immediate viewing
            print("  Expressions found:")
            for expr_name, expr_code, explanation in expressions[:5]:  # Show first 5
                print(f"    • {expr_name}: {expr_code}")
            if len(expressions) > 5:
                print(f"    ... and {len(expressions) - 5} more")
        else:
            print(f"  ⚠ No expressions found in {json_file}")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print(f"📊 Total files processed: {len(json_files)}")


if __name__ == "__main__":
    main()