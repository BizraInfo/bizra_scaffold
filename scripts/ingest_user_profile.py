import os
import json
import sys
import re
import yaml
import asyncio
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

# Add workspace root to path
sys.path.append(os.getcwd())

# Import BIZRA Memory System
try:
    from core.memory.agent_memory import AgentMemorySystem, MemoryTier, MemoryItem
    from core.snr_scorer import SNRScorer, SNRMetrics
except ImportError:
    print("Warning: Could not import BIZRA core modules. Memory injection will be simulated.")
    AgentMemorySystem = None

def load_chat_files(root_dir):
    chat_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                chat_files.append(os.path.join(root, file))
    return chat_files

def extract_messages_from_mapping(mapping):
    messages = []
    for key, value in mapping.items():
        if not value: continue
        if "message" in value and value["message"]:
            msg = value["message"]
            if msg.get("author") and msg["author"]["role"] == "user":
                content = msg.get("content", {})
                parts = content.get("parts", [])
                text = "".join([str(p) for p in parts])
                if text:
                    messages.append(text)
    return messages

def extract_user_data(file_path):
    messages = []
    try:
        # Try UTF-8 first
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            # Fallback to latin-1 or cp1252 if utf-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    data = json.load(f)
            except Exception:
                # Last resort: ignore errors
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)
            
        # Case 1: Standard ChatGPT export (list of conversations)
        if isinstance(data, list):
            for convo in data:
                if "mapping" in convo:
                    messages.extend(extract_messages_from_mapping(convo["mapping"]))
                    
        # Case 2: Single conversation export (dict with mapping)
        elif isinstance(data, dict):
            if "mapping" in data:
                messages.extend(extract_messages_from_mapping(data["mapping"]))
            # Case 3: Recursive structure (node with children) - less common in exports but possible
            # We'll stick to mapping for now as it's the standard export format.
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return messages

def analyze_user_profile(messages):
    keywords = Counter()
    topics = Counter()
    
    # Define topic keywords
    topic_map = {
        "rust": ["rust", "cargo", "crate", "axum", "tokio"],
        "python": ["python", "pip", "script", "pandas", "numpy"],
        "architecture": ["architecture", "system", "design", "pattern", "microservice"],
        "security": ["security", "auth", "crypto", "hash", "verify", "audit"],
        "bizra": ["bizra", "node-0", "genesis", "peak", "masterpiece"],
        "pat_sat": ["pat", "sat", "agent", "team", "autonomous"],
        "optimization": ["optimize", "performance", "speed", "latency", "gpu"],
        "interdisciplinary": ["interdisciplinary", "bridge", "synthesis", "cross-domain"],
    }
    
    print(f"Analyzing {len(messages)} messages...")
    
    for msg in messages:
        # Tokenize simple
        words = re.findall(r'\w+', msg.lower())
        # Filter common stop words (very basic list)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "should", "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those"}
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        keywords.update(filtered_words)
        
        for topic, tags in topic_map.items():
            if any(tag in msg.lower() for tag in tags):
                topics[topic] += 1
                
    return keywords, topics

async def inject_into_memory(profile_data: Dict[str, Any]):
    """Inject derived profile into Agent Memory System."""
    if not AgentMemorySystem:
        print("Skipping memory injection (core modules not found).")
        return

    print("\nInitializing Agent Memory System...")
    # Initialize dependencies
    snr_scorer = SNRScorer()
    memory_system = AgentMemorySystem(snr_scorer=snr_scorer)
    
    # Create Semantic Memory for User Profile
    # We treat the user profile as a high-level semantic fact
    
    profile_content = json.dumps(profile_data, indent=2)
    
    print("Injecting User Profile into Semantic Memory...")
    
    # 1. Store the raw profile as a semantic memory
    await memory_system.remember(
        content=f"User Profile Derived from Chat History: {profile_content}",
        tier=MemoryTier.SEMANTIC,
        domains={"user_profile", "meta_analysis"},
        tags={"profile", "interests", "history"},
        snr_score=0.95, # High confidence derived data
        metadata={
            "source": "ingest_user_profile.py",
            "timestamp": "2025-12-28",
            "type": "derived_profile"
        }
    )
    
    # 2. Store individual top topics as separate semantic memories for better retrieval
    top_topics = profile_data.get("top_topics", {})
    for topic, count in top_topics.items():
        await memory_system.remember(
            content=f"User has strong interest in '{topic}' (mentioned {count} times in history).",
            tier=MemoryTier.SEMANTIC,
            domains={"user_profile", topic},
            tags={"interest", topic},
            snr_score=0.9,
            metadata={"count": count}
        )
        
    print("Memory injection complete.")

def main():
    chat_dir = r"C:\bizra_scaffold\chat data sample"
    if not os.path.exists(chat_dir):
        print(f"Directory not found: {chat_dir}")
        return

    print(f"Scanning {chat_dir}...")
    files = load_chat_files(chat_dir)
    print(f"Found {len(files)} JSON files.")
    
    all_user_msgs = []
    for f in files:
        msgs = extract_user_data(f)
        all_user_msgs.extend(msgs)
        
    print(f"Extracted {len(all_user_msgs)} user messages.")
    
    if not all_user_msgs:
        print("No user messages found. Check file formats.")
        return

    keywords, topics = analyze_user_profile(all_user_msgs)
    
    print("\nTop Topics:")
    for t, c in topics.most_common():
        print(f"- {t}: {c}")
        
    # Prepare Profile Data
    profile_data = {
        "total_messages_analyzed": len(all_user_msgs),
        "top_topics": dict(topics.most_common()),
        "top_keywords": dict(keywords.most_common(50))
    }
        
    # Generate Profile Artifact
    os.makedirs("config", exist_ok=True)
    profile_path = "config/user_profile_derived.yaml"
    with open(profile_path, "w", encoding='utf-8') as f:
        f.write("# Derived User Profile from Chat History\n")
        f.write(f"# Generated by scripts/ingest_user_profile.py\n")
        yaml.dump(profile_data, f)
            
    print(f"\nProfile written to {profile_path}")
    
    # Inject into Memory
    if AgentMemorySystem:
        asyncio.run(inject_into_memory(profile_data))

if __name__ == "__main__":
    main()
