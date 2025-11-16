#!/usr/bin/env python3
"""
SQL Error Analysis Tool
Compares predicted SQL queries with ground truth to identify error patterns.
"""

import re
import sys
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any

class SQLErrorAnalyzer:
    def __init__(self, predicted_file: str, ground_truth_file: str):
        self.predicted_file = predicted_file
        self.ground_truth_file = ground_truth_file
        self.predicted_queries = []
        self.ground_truth_queries = []
        
    def load_queries(self):
        """Load queries from both files"""
        try:
            with open(self.predicted_file, 'r') as f:
                self.predicted_queries = [line.strip() for line in f if line.strip()]
            
            with open(self.ground_truth_file, 'r') as f:
                self.ground_truth_queries = [line.strip() for line in f if line.strip()]
                
            print(f"Loaded {len(self.predicted_queries)} predicted queries")
            print(f"Loaded {len(self.ground_truth_queries)} ground truth queries")
            print(f"Analyzing first {min(len(self.predicted_queries), len(self.ground_truth_queries))} queries\n")
            
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            sys.exit(1)
    
    def extract_select_columns(self, query: str) -> str:
        """Extract SELECT columns from query"""
        match = re.search(r'SELECT\s+(?:DISTINCT\s+)?(.+?)\s+FROM', query, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def extract_table_aliases(self, query: str) -> List[str]:
        """Extract table aliases from FROM clause"""
        from_match = re.search(r'FROM\s+(.+?)(?:\s+WHERE|$)', query, re.IGNORECASE | re.DOTALL)
        if from_match:
            from_clause = from_match.group(1)
            # Find table_alias patterns
            aliases = re.findall(r'\b(\w+_\d+)\b', from_clause)
            return aliases
        return []
    
    def check_missing_operators(self, query: str) -> List[str]:
        """Check for missing comparison operators in time conditions"""
        missing_ops = []
        
        # Pattern: column_name followed by number without operator
        patterns = [
            r'arrival_time\s+(\d+)(?!\d)',
            r'departure_time\s+(\d+)(?!\d)',
            r'flight_1\.arrival_time\s+(\d+)(?!\d)',
            r'flight_1\.departure_time\s+(\d+)(?!\d)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query)
            if matches:
                # Check if there's actually an operator nearby
                for match in matches:
                    # Look for the context around this match
                    context_pattern = f"(arrival_time|departure_time)\\s*[=<>]=?\\s*{match}"
                    if not re.search(context_pattern, query):
                        missing_ops.append(f"Missing operator before {match}")
        
        return missing_ops
    
    def check_syntax_errors(self, query: str) -> List[str]:
        """Check for basic syntax errors"""
        errors = []
        
        # Unbalanced parentheses
        open_parens = query.count('(')
        close_parens = query.count(')')
        if open_parens != close_parens:
            errors.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
        
        # Malformed AND/OR conditions
        if ' AND( ' in query or ' OR( ' in query:
            errors.append("Malformed AND/OR conditions")
        
        # Incomplete queries
        if query.strip().endswith(('WHERE', 'AND', 'OR', ',')):
            errors.append("Incomplete query")
        
        return errors
    
    def analyze_query_pair(self, pred_query: str, gt_query: str, query_idx: int) -> Dict[str, Any]:
        """Analyze a single predicted vs ground truth query pair"""
        analysis = {
            'query_idx': query_idx,
            'predicted': pred_query,
            'ground_truth': gt_query,
            'errors': []
        }
        
        # 1. Missing operators
        missing_ops = self.check_missing_operators(pred_query)
        if missing_ops:
            analysis['errors'].extend([('missing_operator', op) for op in missing_ops])
        
        # 2. SELECT column differences
        pred_cols = self.extract_select_columns(pred_query)
        gt_cols = self.extract_select_columns(gt_query)
        if pred_cols != gt_cols and pred_cols and gt_cols:
            analysis['errors'].append(('wrong_select_columns', {
                'predicted': pred_cols,
                'ground_truth': gt_cols
            }))
        
        # 3. Duplicate aliases
        pred_aliases = self.extract_table_aliases(pred_query)
        if len(pred_aliases) != len(set(pred_aliases)):
            duplicates = [alias for alias in set(pred_aliases) if pred_aliases.count(alias) > 1]
            analysis['errors'].append(('duplicate_aliases', {
                'all_aliases': pred_aliases,
                'duplicates': duplicates
            }))
        
        # 4. Syntax errors
        syntax_errors = self.check_syntax_errors(pred_query)
        if syntax_errors:
            analysis['errors'].extend([('syntax_error', error) for error in syntax_errors])
        
        # 5. Excessive complexity (too many tables)
        table_count = len(pred_aliases)
        if table_count > 6:
            analysis['errors'].append(('excessive_complexity', f"{table_count} tables"))
        
        # 6. Query length comparison
        if len(pred_query) > len(gt_query) * 1.5:  # 50% longer
            analysis['errors'].append(('overly_complex', f"Predicted query much longer than ground truth"))
        
        return analysis
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete error analysis"""
        self.load_queries()
        
        # Analyze query pairs
        analyses = []
        error_stats = defaultdict(int)
        error_examples = defaultdict(list)
        
        num_queries = min(len(self.predicted_queries), len(self.ground_truth_queries))
        
        for i in range(num_queries):
            analysis = self.analyze_query_pair(
                self.predicted_queries[i], 
                self.ground_truth_queries[i], 
                i + 1
            )
            analyses.append(analysis)
            
            # Collect error statistics
            for error_type, error_detail in analysis['errors']:
                error_stats[error_type] += 1
                if len(error_examples[error_type]) < 5:  # Keep first 5 examples
                    error_examples[error_type].append((i + 1, error_detail, analysis))
        
        return {
            'total_queries': num_queries,
            'analyses': analyses,
            'error_stats': dict(error_stats),
            'error_examples': dict(error_examples)
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted analysis results"""
        print("=" * 70)
        print("🔍 SQL ERROR ANALYSIS RESULTS")
        print("=" * 70)
        
        total_queries = results['total_queries']
        error_stats = results['error_stats']
        error_examples = results['error_examples']
        
        # Print summary statistics
        print(f"\n📊 ERROR FREQUENCY ANALYSIS ({total_queries} queries)")
        print("-" * 50)
        
        if not error_stats:
            print("✅ No major errors detected!")
            return
        
        for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_queries) * 100
            error_name = error_type.replace('_', ' ').title()
            print(f"{error_name:<25}: {count:>3}/{total_queries} ({percentage:>5.1f}%)")
        
        # Calculate overall error rate
        queries_with_errors = len([a for a in results['analyses'] if a['errors']])
        error_free_rate = ((total_queries - queries_with_errors) / total_queries) * 100
        
        print(f"\n📈 OVERALL STATISTICS")
        print("-" * 50)
        print(f"Queries with errors: {queries_with_errors}/{total_queries}")
        print(f"Error-free queries: {total_queries - queries_with_errors}/{total_queries} ({error_free_rate:.1f}%)")
        print(f"Average errors per query: {sum(error_stats.values()) / total_queries:.2f}")
    
    def save_detailed_results(self, results: Dict[str, Any], output_file: str):
        """Save detailed analysis results to a text file with complete queries"""
        total_queries = results['total_queries']
        error_stats = results['error_stats']
        error_examples = results['error_examples']
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("🔍 DETAILED SQL ERROR ANALYSIS RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write(f"� ERROR FREQUENCY ANALYSIS ({total_queries} queries)\n")
            f.write("-" * 60 + "\n")
            
            if not error_stats:
                f.write("✅ No major errors detected!\n")
                return
            
            for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_queries) * 100
                error_name = error_type.replace('_', ' ').title()
                f.write(f"{error_name:<25}: {count:>3}/{total_queries} ({percentage:>5.1f}%)\n")
            
            # Detailed examples with complete queries
            f.write(f"\n🚨 DETAILED ERROR EXAMPLES (COMPLETE QUERIES)\n")
            f.write("=" * 80 + "\n")
            
            for error_type, examples in error_examples.items():
                if examples:
                    f.write(f"\n### {error_type.replace('_', ' ').upper()} ###\n")
                    f.write("=" * 60 + "\n")
                    
                    for i, (query_idx, error_detail, analysis) in enumerate(examples):
                        f.write(f"\nExample {i+1} - Query {query_idx}:\n")
                        f.write("-" * 40 + "\n")
                        
                        if error_type == 'missing_operator':
                            f.write(f"Issue: {error_detail}\n")
                            f.write(f"\nPREDICTED QUERY:\n{analysis['predicted']}\n")
                            f.write(f"\nGROUND TRUTH QUERY:\n{analysis['ground_truth']}\n")
                        
                        elif error_type == 'wrong_select_columns':
                            f.write(f"Issue: Wrong SELECT columns\n")
                            f.write(f"Predicted columns: {error_detail['predicted']}\n")
                            f.write(f"Expected columns:  {error_detail['ground_truth']}\n")
                            f.write(f"\nPREDICTED QUERY:\n{analysis['predicted']}\n")
                            f.write(f"\nGROUND TRUTH QUERY:\n{analysis['ground_truth']}\n")
                        
                        elif error_type == 'duplicate_aliases':
                            f.write(f"Issue: Duplicate table aliases\n")
                            f.write(f"Duplicated aliases: {error_detail['duplicates']}\n")
                            f.write(f"All aliases found: {error_detail['all_aliases']}\n")
                            f.write(f"\nPREDICTED QUERY:\n{analysis['predicted']}\n")
                            f.write(f"\nGROUND TRUTH QUERY:\n{analysis['ground_truth']}\n")
                        
                        elif error_type == 'syntax_error':
                            f.write(f"Issue: {error_detail}\n")
                            f.write(f"\nPREDICTED QUERY:\n{analysis['predicted']}\n")
                            f.write(f"\nGROUND TRUTH QUERY:\n{analysis['ground_truth']}\n")
                        
                        elif error_type == 'excessive_complexity':
                            f.write(f"Issue: {error_detail}\n")
                            f.write(f"\nPREDICTED QUERY:\n{analysis['predicted']}\n")
                            f.write(f"\nGROUND TRUTH QUERY:\n{analysis['ground_truth']}\n")
                        
                        else:
                            f.write(f"Issue: {error_detail}\n")
                            f.write(f"\nPREDICTED QUERY:\n{analysis['predicted']}\n")
                            f.write(f"\nGROUND TRUTH QUERY:\n{analysis['ground_truth']}\n")
                        
                        f.write("\n" + "~" * 80 + "\n")
            
            # Improvement recommendations
            f.write(f"\n🎯 IMPROVEMENT RECOMMENDATIONS\n")
            f.write("=" * 60 + "\n")
            
            if 'missing_operator' in error_stats:
                f.write("1. MISSING OPERATORS:\n")
                f.write("   - Add training examples with explicit comparison operators\n")
                f.write("   - Focus on time conditions: 'arrival_time < 900', 'departure_time = 1700'\n\n")
            
            if 'duplicate_aliases' in error_stats:
                f.write("2. DUPLICATE ALIASES:\n")
                f.write("   - Improve alias generation consistency\n")
                f.write("   - Add constraints to prevent reusing table aliases\n\n")
            
            if 'wrong_select_columns' in error_stats:
                f.write("3. SELECT COLUMNS:\n")
                f.write("   - Better column selection training\n")
                f.write("   - Include more examples with different column combinations\n\n")
            
            if 'syntax_error' in error_stats:
                f.write("4. SYNTAX ERRORS:\n")
                f.write("   - Add syntax validation during training\n")
                f.write("   - Use SQL parsing validation in the loss function\n\n")
            
            if 'excessive_complexity' in error_stats:
                f.write("5. COMPLEXITY:\n")
                f.write("   - Simplify query generation approach\n")
                f.write("   - Constrain maximum number of table joins\n\n")
            
            # Overall statistics
            queries_with_errors = len([a for a in results['analyses'] if a['errors']])
            error_free_rate = ((total_queries - queries_with_errors) / total_queries) * 100
            
            f.write(f"📈 OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total queries analyzed: {total_queries}\n")
            f.write(f"Queries with errors: {queries_with_errors}/{total_queries}\n")
            f.write(f"Error-free queries: {total_queries - queries_with_errors}/{total_queries} ({error_free_rate:.1f}%)\n")
            f.write(f"Average errors per query: {sum(error_stats.values()) / total_queries:.2f}\n")


def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze_sql_errors.py <predicted_file> <ground_truth_file>")
        print("Example: python analyze_sql_errors.py /Users/hb25/Downloads/dev.sql data/dev.sql")
        sys.exit(1)
    
    predicted_file = sys.argv[1]
    ground_truth_file = sys.argv[2]
    
    # Generate output filename
    output_file = "sql_error_analysis_detailed.txt"
    
    analyzer = SQLErrorAnalyzer(predicted_file, ground_truth_file)
    results = analyzer.run_analysis()
    
    # Print summary to console
    analyzer.print_results(results)
    
    # Save detailed results to file
    analyzer.save_detailed_results(results, output_file)
    print(f"\n💾 Detailed analysis saved to: {output_file}")
    print(f"📄 File contains complete queries for all error examples")


if __name__ == "__main__":
    main()