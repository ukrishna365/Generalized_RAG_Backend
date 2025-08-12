#!/usr/bin/env python3
"""
Simple CLI interface for the RAG Query Engine
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query_engine import RAGQueryEngine

def main():
    print("RAG Query Engine - CLI Interface")
    print("=" * 50)
    print("Type 'quit' to exit")
    print()
    
    # Initialize the RAG engine
    try:
        engine = RAGQueryEngine()
        print("RAG Engine initialized successfully")
    except Exception as e:
        print(f"Error initializing RAG Engine: {e}")
        return
    
    # Interactive query loop
    while True:
        try:
            # Get user input
            query = input("\nEnter your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                print("Please enter a question")
                continue
            
            # Process the query
            print("\nProcessing query...")
            result = engine.query(query)
            
            # Display results
            print(f"\nAnswer:")
            print(f"   {result['answer']}")
            
            print(f"\nSources ({len(result['sources'])} items):")
            for i, source in enumerate(result['sources'], 1):
                file_name = source.get('file_name', 'Unknown')
                text_preview = source.get('text_markdown', '')[:100] + "..." if source.get('text_markdown') else "No text"
                print(f"   {i}. {file_name}: {text_preview}")
            
            print(f"\nProcessed at: {result['timestamp']}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Clean up
    engine.close()

if __name__ == "__main__":
    main() 