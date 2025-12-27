import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import glob

class ChatNode:
    def __init__(self, node_id: str, role: str, content: str, timestamp: float, parent_id: Optional[str] = None):
        self.node_id = node_id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.parent_id = parent_id
        self.children: List['ChatNode'] = []
        self.snr_score: float = 0.0

    def to_dict(self):
        return {
            "id": self.node_id,
            "role": self.role,
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "timestamp": self.timestamp,
            "snr_score": self.snr_score
        }

class ConversationGraph:
    def __init__(self, title: str, root_node: Optional[ChatNode] = None):
        self.title = title
        self.nodes: Dict[str, ChatNode] = {}
        self.root = root_node
        self.edges: List[Dict[str, str]] = []

    def add_node(self, node: ChatNode):
        self.nodes[node.node_id] = node
        if node.parent_id:
            self.edges.append({"source": node.parent_id, "target": node.node_id})
            if node.parent_id in self.nodes:
                self.nodes[node.parent_id].children.append(node)

    def calculate_snr(self):
        """
        Calculate Signal-to-Noise Ratio for each node.
        Heuristic:
        - High information density (keywords / total words)
        - Structure (bullet points, code blocks)
        - Explicit 'Gem' markers
        """
        for node in self.nodes.values():
            if not node.content:
                continue
            
            text = node.content
            length = len(text)
            if length == 0:
                continue

            # Heuristics
            has_code = "```" in text
            has_lists = "- " in text or "1. " in text
            has_gems = "gem" in text.lower() or "key result" in text.lower() or "insight" in text.lower()
            
            score = 1.0
            if has_code: score += 2.0
            if has_lists: score += 1.5
            if has_gems: score += 3.0
            
            # Normalize by length (penalize very short or extremely verbose without structure)
            if length < 50: score *= 0.5
            
            node.snr_score = score

class ChatIngestor:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.conversations: List[ConversationGraph] = []

    def parse_file(self, file_path: str) -> Optional[ConversationGraph]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            title = data.get('title', 'Untitled')
            mapping = data.get('mapping', {})
            
            graph = ConversationGraph(title)
            
            # First pass: create nodes
            for node_id, node_data in mapping.items():
                message = node_data.get('message')
                if not message:
                    continue
                
                author = message.get('author', {})
                role = author.get('role', 'unknown')
                
                content_obj = message.get('content', {})
                parts = content_obj.get('parts', [])
                text_content = "".join([str(p) for p in parts])
                
                create_time = message.get('create_time') or 0.0
                
                parent_id = node_data.get('parent')
                
                node = ChatNode(node_id, role, text_content, create_time, parent_id)
                graph.add_node(node)
            
            # Calculate SNR
            graph.calculate_snr()
            
            return graph
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def ingest_all(self):
        # Recursive search for .json files
        search_pattern = os.path.join(self.root_path, "**", "*.json")
        files = glob.glob(search_pattern, recursive=True)
        
        print(f"Found {len(files)} conversation files.")
        
        for file_path in files:
            # Skip non-conversation jsons if any (e.g. manifests)
            if "manifest" in file_path.lower():
                continue
                
            graph = self.parse_file(file_path)
            if graph:
                self.conversations.append(graph)
        
        print(f"Successfully ingested {len(self.conversations)} conversations.")

    def get_golden_gems(self, top_k: int = 5):
        all_nodes = []
        for conv in self.conversations:
            for node in conv.nodes.values():
                if node.role == 'assistant': # Only look at AI responses for gems usually
                    all_nodes.append((node, conv.title))
        
        # Sort by SNR
        all_nodes.sort(key=lambda x: x[0].snr_score, reverse=True)
        
        return all_nodes[:top_k]

    def export_gems_to_markdown(self, output_path: str, top_k: int = 5):
        gems = self.get_golden_gems(top_k)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# BIZRA: The Recovered Masterpiece\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Source:** Living Knowledge Base (Chat History)\n")
            f.write(f"**Method:** SNR Optimization & Graph of Thoughts Extraction\n\n")
            
            for i, (node, title) in enumerate(gems):
                f.write(f"## Gem #{i+1}: {title}\n")
                f.write(f"**SNR Score:** {node.snr_score}\n")
                f.write(f"**Timestamp:** {datetime.fromtimestamp(node.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("### Content\n\n")
                f.write(node.content)
                f.write("\n\n---\n\n")
        
        print(f"Exported {top_k} gems to {output_path}")

if __name__ == "__main__":
    import os
    # Use environment variables for personal paths - NEVER hardcode
    chat_data_path = os.environ.get("BIZRA_CHAT_DATA", "./data/chat")
    evidence_path = os.environ.get("BIZRA_EVIDENCE_PATH", "./evidence")
    
    ingestor = ChatIngestor(chat_data_path)
    ingestor.ingest_all()
    
    output_file = os.path.join(evidence_path, "RECOVERED_MASTERPIECE.md")
    ingestor.export_gems_to_markdown(output_file, top_k=5)

