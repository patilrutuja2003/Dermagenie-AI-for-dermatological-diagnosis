import sqlite3
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from transformers import pipeline
import requests
import json
from datetime import datetime, timedelta
import threading
import schedule
import time
import os

# Initialize summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class DermatologistDatabase:
    def __init__(self, db_path="dermatologist_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create dermatologists table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dermatologists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                qualifications TEXT NOT NULL,
                specializations TEXT,
                latitude REAL,
                longitude REAL,
                address TEXT,
                contact TEXT,
                email TEXT,
                availability TEXT,
                rating REAL,
                review_count INTEGER,
                last_updated TIMESTAMP
            )
        ''')
        
        # Create specialization mapping table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS condition_specialist_mapping (
                condition_name TEXT,
                specialist_id INTEGER,
                FOREIGN KEY(specialist_id) REFERENCES dermatologists(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_dermatologist(self, data):
        """Add a new dermatologist to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO dermatologists (
                name, qualifications, specializations, latitude, longitude,
                address, contact, email, availability, rating, review_count,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['name'], data['qualifications'], json.dumps(data['specializations']),
            data['latitude'], data['longitude'], data['address'], data['contact'],
            data['email'], json.dumps(data['availability']), data.get('rating', 0),
            data.get('review_count', 0), datetime.now().isoformat()
        ))
        
        specialist_id = cursor.lastrowid
        for condition in data['specializations']:
            cursor.execute('''
                INSERT INTO condition_specialist_mapping (condition_name, specialist_id)
                VALUES (?, ?)
            ''', (condition, specialist_id))
        
        conn.commit()
        conn.close()
    
    def find_nearest_specialists(self, condition, user_location, max_distance=50):
        """Find nearest specialists for a specific condition"""
        user_lat, user_lon = user_location
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT d.* FROM dermatologists d
            JOIN condition_specialist_mapping m ON d.id = m.specialist_id
            WHERE m.condition_name = ?
        ''', (condition,))
        
        specialists = []
        for row in cursor.fetchall():
            distance = geodesic(
                (user_lat, user_lon),
                (row['latitude'], row['longitude'])
            ).miles
            
            if distance <= max_distance:
                specialist_data = dict(row)
                specialist_data['distance'] = round(distance, 2)
                specialist_data['specializations'] = json.loads(row['specializations'])
                specialist_data['availability'] = json.loads(row['availability'])
                specialists.append(specialist_data)
        
        conn.close()
        
        # Sort by distance and rating
        return sorted(specialists, key=lambda x: (x['distance'], -x['rating']))

class MedicalJournalAggregator:
    def __init__(self, cache_dir="journal_cache"):
        self.cache_dir = cache_dir
        self.summarizer = summarizer
        self.cache = {}
        self.last_update = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch_and_summarize_articles(self, condition, max_articles=5):
        """Fetch and summarize articles for a specific condition"""
        # Check cache first
        cache_key = condition.lower().replace(" ", "_")
        if cache_key in self.cache:
            if (datetime.now() - self.last_update[cache_key]).days < 7:
                return self.cache[cache_key]
        
        # Fetch articles from multiple sources
        articles = []
        
        # PubMed articles
        try:
            pubmed_articles = fetch_pubmed_articles(search_pubmed(condition)[:max_articles])
            
            for article in pubmed_articles:
                try:
                    # Get abstract
                    abstract = article.get('abstract', '')
                    if not abstract:
                        continue
                    
                    # Summarize abstract
                    summary = self.summarizer(abstract, max_length=150, min_length=50)[0]['summary_text']
                    
                    articles.append({
                        'title': article['title'],
                        'authors': article['authors'],
                        'journal': article.get('journal', 'Unknown'),
                        'publication_date': article.get('publication_date', 'Unknown'),
                        'doi': article.get('doi', 'N/A'),
                        'summary': summary,
                        'relevance_score': self._calculate_relevance_score(abstract, condition)
                    })
                except Exception as e:
                    print(f"Error processing article: {str(e)}")
                    continue
            
            # Sort by relevance score
            articles.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Update cache
            self.cache[cache_key] = articles
            self.last_update[cache_key] = datetime.now()
            
        except Exception as e:
            print(f"Error fetching articles: {str(e)}")
            if cache_key in self.cache:  # Return cached data if available
                return self.cache[cache_key]
            return []
        
        return articles
    
    def _calculate_relevance_score(self, abstract, condition):
        """Calculate relevance score based on keyword matching and citation count"""
        # This is a simple implementation - could be enhanced with ML models
        keywords = condition.lower().split()
        score = sum(1 for keyword in keywords if keyword in abstract.lower())
        return score

# Background task to update medical journal cache
def update_journal_cache(aggregator):
    """Update the journal cache periodically"""
    while True:
        for condition in aggregator.cache.keys():
            try:
                aggregator.fetch_and_summarize_articles(condition)
            except Exception as e:
                print(f"Error updating cache for {condition}: {str(e)}")
        time.sleep(24 * 60 * 60)  # Update daily

def start_background_updates(aggregator):
    """Start background updates for journal cache"""
    update_thread = threading.Thread(
        target=update_journal_cache,
        args=(aggregator,),
        daemon=True
    )
    update_thread.start()
