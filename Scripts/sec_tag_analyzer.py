import json
import os
from collections import defaultdict
import pandas as pd
import time

class SECTagAnalyzer:
    def __init__(self, json_directory="Data/sec_data/json"):
        self.json_directory = json_directory
        self.tag_stats = defaultdict(int)
        self.tag_value_counts = defaultdict(int)
        self.tag_examples = defaultdict(list)
        self.unit_types = defaultdict(set)
        
    def analyze_json_files(self):
        """Analyze all JSON files in the directory to find GAAP tags and their frequencies."""
        json_files = [f for f in os.listdir(self.json_directory) if f.endswith('.json')]
        total_files = len(json_files)
        
        print(f"Found {total_files} JSON files to process...")
        
        for i, filename in enumerate(json_files, 1):
            if i % 25 == 0:
                print(f"Processing file {i} of {total_files} ({(i/total_files*100):.1f}%)")
                
            with open(os.path.join(self.json_directory, filename), 'r') as f:
                try:
                    data = json.load(f)
                    ticker = filename.replace('.json', '')
                    self._process_company_data(data, ticker)
                except json.JSONDecodeError:
                    print(f"Error reading {filename}")
        
        print("Finished processing all files!")

    def _process_company_data(self, data, ticker):
        if 'facts' in data and 'us-gaap' in data['facts']:
            gaap_data = data['facts']['us-gaap']
            
            for tag, content in gaap_data.items():
                self.tag_stats[tag] += 1
                
                if 'units' in content:
                    self.unit_types[tag].update(content['units'].keys())
                    
                    for unit_type, values in content['units'].items():
                        self.tag_value_counts[tag] += len(values)
                        
                        if values:
                            example = {
                                'ticker': ticker,
                                'unit': unit_type,
                                'value': values[0].get('val'),
                                'frame': values[0].get('frame', ''),
                                'form': values[0].get('form', '')
                            }
                            if len(self.tag_examples[tag]) < 3:
                                self.tag_examples[tag].append(example)

    def get_frequency_dataframe(self):
        print("Creating frequency DataFrame...")
        df = pd.DataFrame([
            {
                'tag': tag,
                'company_count': company_count,
                'value_count': self.tag_value_counts[tag],
                'avg_values_per_company': round(self.tag_value_counts[tag] / company_count, 2),
                'units': ', '.join(self.unit_types[tag]) 
            }
            for tag, company_count in self.tag_stats.items()
        ])
        return df.sort_values('company_count', ascending=False)

def main():
    analyzer = SECTagAnalyzer()
    start_time = time.time()
    
    print("Starting analysis...")
    analyzer.analyze_json_files()
    
    df = analyzer.get_frequency_dataframe()
    
    output_file = "C:/Users/user/Repos/thesis_clustering_portfolio/Data/sec_data/tag_frequencies.csv"
    print(f"Saving results to {output_file}...")
    df.to_csv(output_file, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis complete in {elapsed_time:.1f} seconds!")
    print(f"Found {len(df)} unique GAAP tags")
    print(f"Results saved to {output_file}")

    print("\nTop 10 most frequent tags by company count:")
    print(df[['tag', 'company_count', 'value_count', 'avg_values_per_company']].head(10))

    print("\nTop 10 tags by total number of values:")
    print(df[['tag', 'company_count', 'value_count', 'avg_values_per_company']].nlargest(10, 'value_count'))

if __name__ == "__main__":
    main()