"""
Data generator for creating test cases
Location: src/data_generator.py
"""

import random
import json
from typing import List, Dict, Tuple
import os


class DataGenerator:
    """Generate test data for the aligned-table benchmark."""
    
    # Entity templates with attributes
    ENTITY_TEMPLATES = {
        "countries": {
            "attributes": ["Capital", "Population", "Area (km²)", "Currency", "Language"],
            "data": [
                ["USA", "Washington D.C.", "331M", "9,833,520", "USD", "English"],
                ["China", "Beijing", "1.4B", "9,596,961", "CNY", "Mandarin"],
                ["Japan", "Tokyo", "125M", "377,975", "JPY", "Japanese"],
                ["Germany", "Berlin", "83M", "357,022", "EUR", "German"],
                ["Brazil", "Brasília", "213M", "8,515,767", "BRL", "Portuguese"],
                ["India", "New Delhi", "1.38B", "3,287,263", "INR", "Hindi"],
                ["France", "Paris", "67M", "643,801", "EUR", "French"],
                ["UK", "London", "68M", "242,495", "GBP", "English"],
            ]
        },
        "companies": {
            "attributes": ["Founded", "Revenue", "Employees", "Industry"],
            "data": [
                ["Apple", "1976", "$394B", "164,000", "Technology"],
                ["Microsoft", "1975", "$198B", "221,000", "Technology"],
                ["Amazon", "1994", "$514B", "1,540,000", "E-commerce"],
                ["Google", "1998", "$283B", "190,000", "Technology"],
                ["Tesla", "2003", "$81B", "127,000", "Automotive"],
                ["Meta", "2004", "$117B", "86,000", "Technology"],
                ["Netflix", "1997", "$31B", "12,800", "Entertainment"],
                ["Samsung", "1938", "$221B", "267,000", "Electronics"],
            ]
        },
        "movies": {
            "attributes": ["Director", "Year", "Rating", "Box Office"],
            "data": [
                ["Avatar", "James Cameron", "2009", "7.9/10", "$2.9B"],
                ["Titanic", "James Cameron", "1997", "7.9/10", "$2.2B"],
                ["Inception", "Christopher Nolan", "2010", "8.8/10", "$836M"],
                ["Interstellar", "Christopher Nolan", "2014", "8.7/10", "$701M"],
                ["The Dark Knight", "Christopher Nolan", "2008", "9.0/10", "$1.0B"],
                ["Parasite", "Bong Joon-ho", "2019", "8.5/10", "$258M"],
                ["Pulp Fiction", "Quentin Tarantino", "1994", "8.9/10", "$213M"],
                ["The Godfather", "Francis Ford Coppola", "1972", "9.2/10", "$291M"],
            ]
        },
        "books": {
            "attributes": ["Author", "Year", "Pages", "Genre"],
            "data": [
                ["1984", "George Orwell", "1949", "328", "Dystopian"],
                ["To Kill a Mockingbird", "Harper Lee", "1960", "324", "Fiction"],
                ["The Great Gatsby", "F. Scott Fitzgerald", "1925", "180", "Fiction"],
                ["Pride and Prejudice", "Jane Austen", "1813", "432", "Romance"],
                ["The Catcher in the Rye", "J.D. Salinger", "1951", "277", "Fiction"],
                ["Harry Potter", "J.K. Rowling", "1997", "309", "Fantasy"],
                ["The Hobbit", "J.R.R. Tolkien", "1937", "310", "Fantasy"],
                ["Moby Dick", "Herman Melville", "1851", "635", "Adventure"],
            ]
        },
        "universities": {
            "attributes": ["Location", "Founded", "Students", "Ranking"],
            "data": [
                ["Harvard", "Cambridge, MA", "1636", "23,000", "#1"],
                ["Stanford", "Stanford, CA", "1885", "17,000", "#2"],
                ["MIT", "Cambridge, MA", "1861", "11,500", "#3"],
                ["Oxford", "Oxford, UK", "1096", "24,000", "#4"],
                ["Cambridge", "Cambridge, UK", "1209", "24,000", "#5"],
                ["Yale", "New Haven, CT", "1701", "14,000", "#6"],
                ["Princeton", "Princeton, NJ", "1746", "8,500", "#7"],
                ["Columbia", "New York, NY", "1754", "33,000", "#8"],
            ]
        },
        "products": {
            "attributes": ["Brand", "Price", "Rating", "Category"],
            "data": [
                ["iPhone 15", "Apple", "$999", "4.8/5", "Smartphone"],
                ["Galaxy S24", "Samsung", "$899", "4.7/5", "Smartphone"],
                ["MacBook Pro", "Apple", "$2,499", "4.9/5", "Laptop"],
                ["ThinkPad X1", "Lenovo", "$1,799", "4.6/5", "Laptop"],
                ["AirPods Pro", "Apple", "$249", "4.7/5", "Headphones"],
                ["PS5", "Sony", "$499", "4.8/5", "Gaming"],
                ["Xbox Series X", "Microsoft", "$499", "4.7/5", "Gaming"],
                ["Kindle", "Amazon", "$139", "4.6/5", "E-reader"],
            ]
        },
        "athletes": {
            "attributes": ["Sport", "Country", "Medals", "Active Years"],
            "data": [
                ["Michael Phelps", "Swimming", "USA", "28 Olympic", "2000-2016"],
                ["Usain Bolt", "Athletics", "Jamaica", "8 Olympic", "2004-2017"],
                ["Serena Williams", "Tennis", "USA", "23 Grand Slams", "1995-2022"],
                ["Lionel Messi", "Football", "Argentina", "7 Ballon d'Or", "2004-present"],
                ["LeBron James", "Basketball", "USA", "4 NBA Titles", "2003-present"],
                ["Roger Federer", "Tennis", "Switzerland", "20 Grand Slams", "1998-2022"],
                ["Simone Biles", "Gymnastics", "USA", "32 Olympic/World", "2013-present"],
                ["Tom Brady", "American Football", "USA", "7 Super Bowls", "2000-2023"],
            ]
        },
        "cities": {
            "attributes": ["Country", "Population", "Area (km²)", "Founded"],
            "data": [
                ["Tokyo", "Japan", "14M", "2,194", "1457"],
                ["New York", "USA", "8.3M", "784", "1624"],
                ["London", "UK", "9M", "1,572", "47 AD"],
                ["Paris", "France", "2.2M", "105", "3rd century BC"],
                ["Shanghai", "China", "24M", "6,341", "1291"],
                ["Mumbai", "India", "20M", "603", "1507"],
                ["São Paulo", "Brazil", "12M", "1,521", "1554"],
                ["Dubai", "UAE", "3.3M", "4,114", "1833"],
            ]
        }
    }
    
    def __init__(self, config: Dict):
        """
        Initialize the data generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.test_config = config.get('test', {})
        
    def generate_test_cases(self, num_samples: int = None) -> List[Dict]:
        """
        Generate test cases for the benchmark.
        
        Args:
            num_samples: Number of test cases to generate
            
        Returns:
            List of test case dictionaries
        """
        if num_samples is None:
            num_samples = self.test_config.get('num_samples', 100)
        
        test_cases = []
        entity_types = list(self.ENTITY_TEMPLATES.keys())
        
        for i in range(num_samples):
            # Randomly select entity type
            entity_type = random.choice(entity_types)
            
            # Generate table dimensions
            num_rows = random.randint(
                self.test_config.get('min_rows', 3),
                self.test_config.get('max_rows', 8)
            )
            num_cols = random.randint(
                self.test_config.get('min_cols', 3),
                self.test_config.get('max_cols', 6)
            )
            
            # Generate table data
            table_data, context = self._generate_table(
                entity_type, num_rows, num_cols
            )
            
            # Random table format for this test case
            table_format = random.choice(
                self.test_config.get('table_formats', ['latex', 'markdown', 'text'])
            )
            
            test_case = {
                'id': i,
                'entity_type': entity_type,
                'table_data': table_data,
                'context': context,
                'table_format': table_format,
                'num_rows': num_rows,
                'num_cols': num_cols
            }
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_table(self, entity_type: str, num_rows: int, num_cols: int) -> Tuple[List[List[str]], str]:
        """
        Generate table data for a specific entity type.
        
        Args:
            entity_type: Type of entity
            num_rows: Number of rows
            num_cols: Number of columns (including entity column)
            
        Returns:
            Tuple of (table_data, context_description)
        """
        template = self.ENTITY_TEMPLATES[entity_type]
        available_data = template['data']
        attributes = template['attributes']
        
        # Sample rows
        sampled_rows = random.sample(available_data, min(num_rows, len(available_data)))
        
        # Select columns (first column is always entity name, then sample attributes)
        num_attr_cols = min(num_cols - 1, len(attributes))
        selected_attrs = random.sample(attributes, num_attr_cols)
        
        # Build table data
        table_data = []
        for row in sampled_rows:
            entity_name = row[0]
            row_data = [entity_name]
            
            for attr in selected_attrs:
                attr_idx = attributes.index(attr) + 1
                if attr_idx < len(row):
                    row_data.append(row[attr_idx])
                else:
                    row_data.append("N/A")
            
            table_data.append(row_data)
        
        # Generate context description
        context = self._generate_context(entity_type, table_data, selected_attrs)
        
        return table_data, context
    
    def _generate_context(self, entity_type: str, table_data: List[List[str]], attributes: List[str]) -> str:
        """
        Generate textual description of the table.
        
        Args:
            entity_type: Type of entity
            table_data: The table data
            attributes: List of attribute names
            
        Returns:
            Context description string
        """
        context_parts = []
        context_parts.append(f"This table contains information about various {entity_type}.")
        
        # Describe columns
        col_names = [entity_type.rstrip('s').capitalize()] + attributes
        context_parts.append(f"The table has {len(col_names)} columns: {', '.join(col_names)}.")
        
        # Describe each row
        for idx, row in enumerate(table_data, 1):
            row_desc = f"Row {idx}: " + ", ".join(row)
            context_parts.append(row_desc)
        
        return " ".join(context_parts)
    
    def save_test_cases(self, test_cases: List[Dict], output_file: str):
        """
        Save test cases to a JSON file.
        
        Args:
            test_cases: List of test case dictionaries
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
    
    def load_test_cases(self, input_file: str) -> List[Dict]:
        """
        Load test cases from a JSON file.
        
        Args:
            input_file: Input file path
            
        Returns:
            List of test case dictionaries
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)