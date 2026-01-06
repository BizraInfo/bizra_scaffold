import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import glob

# Configure logging
logger = logging.getLogger(__name__)

class ChatNode:
    """
    Represents a single node in a conversation graph.
    
    Attributes:
        node_id: Unique identifier for the node
        role: Role of the author (e.g., 'user', 'assistant', 'system')
        content: Text content of the message
        timestamp: Unix timestamp when the message was created
        parent_id: ID of the parent node, if any
        children: List of child nodes
        snr_score: Signal-to-Noise Ratio score for content quality
    """
    
    def __init__(self, node_id: str, role: str, content: str, timestamp: float, parent_id: Optional[str] = None):
        self.node_id = node_id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.parent_id = parent_id
        self.children: List['ChatNode'] = []
        self.snr_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to dictionary representation.
        
        Returns:
            Dictionary containing node metadata and truncated content
        """
        return {
            "id": self.node_id,
            "role": self.role,
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "timestamp": self.timestamp,
            "snr_score": self.snr_score
        }

class ConversationGraph:
    """
    Represents a conversation as a directed graph of chat nodes.
    
    Attributes:
        title: Title of the conversation
        nodes: Dictionary mapping node IDs to ChatNode objects
        root: Root node of the conversation tree
        edges: List of edges between nodes
        snr_weights: Configurable weights for SNR scoring
    """
    
    # Default SNR scoring weights (configurable)
    DEFAULT_SNR_WEIGHTS = {
        'code_bonus': 2.0,
        'list_bonus': 1.5,
        'gem_bonus': 3.0,
        'short_penalty': 0.5,
        'min_length': 50
    }
    
    def __init__(self, title: str, root_node: Optional[ChatNode] = None, snr_weights: Optional[Dict[str, float]] = None):
        """
        Initialize a conversation graph.
        
        Args:
            title: Title of the conversation
            root_node: Optional root node
            snr_weights: Optional custom SNR scoring weights
        """
        self.title = title
        self.nodes: Dict[str, ChatNode] = {}
        self.root = root_node
        self.edges: List[Dict[str, str]] = []
        self.snr_weights = snr_weights or self.DEFAULT_SNR_WEIGHTS.copy()

    def add_node(self, node: ChatNode) -> None:
        """
        Add a node to the conversation graph.
        
        Args:
            node: ChatNode to add
        """
        if not node or not node.node_id:
            logger.warning("Attempted to add invalid node")
            return
            
        self.nodes[node.node_id] = node
        if node.parent_id:
            self.edges.append({"source": node.parent_id, "target": node.node_id})
            if node.parent_id in self.nodes:
                self.nodes[node.parent_id].children.append(node)

    def calculate_snr(self) -> None:
        """
        Calculate Signal-to-Noise Ratio for each node.
        
        The SNR score is a heuristic measure of content quality based on:
        - High information density (keywords / total words)
        - Structure (bullet points, code blocks)
        - Explicit 'Gem' markers (insights, key results)
        - Content length normalization
        
        Scoring is configurable via snr_weights dictionary.
        """
        for node in self.nodes.values():
            # Skip nodes with no content
            if not node.content:
                node.snr_score = 0.0
                continue
            
            text = node.content
            length = len(text)
            
            # Edge case: empty content after stripping
            if length == 0:
                node.snr_score = 0.0
                continue

            # Heuristic feature detection
            has_code = "```" in text
            has_lists = "- " in text or "1. " in text
            has_gems = "gem" in text.lower() or "key result" in text.lower() or "insight" in text.lower()
            
            # Base score
            score = 1.0
            
            # Apply configurable bonuses
            if has_code:
                score += self.snr_weights.get('code_bonus', 2.0)
            if has_lists:
                score += self.snr_weights.get('list_bonus', 1.5)
            if has_gems:
                score += self.snr_weights.get('gem_bonus', 3.0)
            
            # Normalize by length (penalize very short messages)
            min_length = self.snr_weights.get('min_length', 50)
            if length < min_length:
                score *= self.snr_weights.get('short_penalty', 0.5)
            
            node.snr_score = score
            if node.node_id:  # Defensive check for logging
                logger.debug(f"Node {node.node_id[:8]}... SNR score: {score:.2f}")

class ChatIngestor:
    """
    Ingests and processes chat conversation data from JSON files.
    
    Attributes:
        root_path: Root directory to search for conversation files
        conversations: List of parsed ConversationGraph objects
        snr_weights: Optional custom SNR scoring weights
    """
    
    def __init__(self, root_path: str, snr_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the chat ingestor.
        
        Args:
            root_path: Root directory containing conversation JSON files
            snr_weights: Optional custom weights for SNR calculation
        
        Raises:
            ValueError: If root_path is empty or invalid
        """
        if not root_path or not isinstance(root_path, str):
            raise ValueError("root_path must be a non-empty string")
        
        self.root_path = root_path
        self.conversations: List[ConversationGraph] = []
        self.snr_weights = snr_weights

    def parse_file(self, file_path: str) -> Optional[ConversationGraph]:
        """
        Parse a single conversation JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            ConversationGraph object if successful, None otherwise
        """
        if not file_path or not os.path.exists(file_path):
            logger.error(f"Invalid or non-existent file path: {file_path}")
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate required fields
            if not isinstance(data, dict):
                logger.warning(f"Invalid JSON structure in {file_path}: expected dict")
                return None
                
            title = data.get('title', 'Untitled')
            mapping = data.get('mapping', {})
            
            if not isinstance(mapping, dict):
                logger.warning(f"Invalid mapping structure in {file_path}")
                return None
            
            graph = ConversationGraph(title, snr_weights=self.snr_weights)
            
            # First pass: create nodes with defensive handling
            for node_id, node_data in mapping.items():
                if not isinstance(node_data, dict):
                    logger.debug(f"Skipping invalid node_data for {node_id}")
                    continue
                    
                message = node_data.get('message')
                if not message or not isinstance(message, dict):
                    continue
                
                author = message.get('author', {})
                if not isinstance(author, dict):
                    author = {}
                role = author.get('role', 'unknown')
                
                content_obj = message.get('content', {})
                if not isinstance(content_obj, dict):
                    content_obj = {}
                parts = content_obj.get('parts', [])
                if not isinstance(parts, list):
                    parts = []
                text_content = "".join([str(p) for p in parts])
                
                create_time = message.get('create_time') or 0.0
                parent_id = node_data.get('parent')
                
                node = ChatNode(node_id, role, text_content, create_time, parent_id)
                graph.add_node(node)
            
            # Calculate SNR
            graph.calculate_snr()
            
            logger.info(f"Successfully parsed {file_path}: {len(graph.nodes)} nodes")
            return graph
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}", exc_info=True)
            return None

    def ingest_all(self) -> int:
        """
        Recursively search for and ingest all conversation JSON files.
        
        Returns:
            Number of successfully ingested conversations
        """
        if not os.path.exists(self.root_path):
            logger.error(f"Root path does not exist: {self.root_path}")
            return 0
            
        # Recursive search for .json files
        search_pattern = os.path.join(self.root_path, "**", "*.json")
        files = glob.glob(search_pattern, recursive=True)
        
        logger.info(f"Found {len(files)} conversation files in {self.root_path}")
        
        for file_path in files:
            # Skip non-conversation jsons (e.g. manifests, config files)
            if "manifest" in file_path.lower():
                logger.debug(f"Skipping manifest file: {file_path}")
                continue
                
            graph = self.parse_file(file_path)
            if graph:
                self.conversations.append(graph)
        
        logger.info(f"Successfully ingested {len(self.conversations)} conversations")
        return len(self.conversations)

    def get_golden_gems(self, top_k: int = 5) -> List[tuple]:
        """
        Get the top-k highest SNR nodes (golden gems).
        
        Args:
            top_k: Number of top gems to retrieve
            
        Returns:
            List of tuples (node, conversation_title) sorted by SNR score
        """
        if top_k <= 0:
            logger.warning(f"Invalid top_k value: {top_k}, using default 5")
            top_k = 5
            
        all_nodes = []
        for conv in self.conversations:
            for node in conv.nodes.values():
                # Only look at AI responses for gems usually
                if node.role == 'assistant':
                    all_nodes.append((node, conv.title))
        
        # Sort by SNR score (descending)
        all_nodes.sort(key=lambda x: x[0].snr_score, reverse=True)
        
        logger.info(f"Retrieved top {min(top_k, len(all_nodes))} gems from {len(all_nodes)} assistant nodes")
        return all_nodes[:top_k]

    def export_gems_to_markdown(self, output_path: str, top_k: int = 5) -> bool:
        """
        Export top-k gems to a Markdown file.
        
        Args:
            output_path: Path where the markdown file should be written
            top_k: Number of top gems to export
            
        Returns:
            True if export succeeded, False otherwise
        """
        if not output_path:
            logger.error("output_path cannot be empty")
            return False
            
        try:
            gems = self.get_golden_gems(top_k)
            
            if not gems:
                logger.warning("No gems found to export")
                return False
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and output_dir != '.' and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            
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
            
            logger.info(f"Exported {len(gems)} gems to {output_path}")
            return True
            
        except IOError as e:
            logger.error(f"I/O error writing to {output_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error exporting gems: {e}", exc_info=True)
            return False

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Allow configuration via environment variables
    root_path = os.environ.get('CHAT_DATA_PATH', "C:\\bizra_scaffold\\chat data sample")
    output_file = os.environ.get('OUTPUT_PATH', "C:\\bizra_scaffold\\evidence\\RECOVERED_MASTERPIECE.md")
    
    # Parse TOP_K with error handling
    try:
        top_k = int(os.environ.get('TOP_K', '5'))
    except ValueError as e:
        logger.warning(f"Invalid TOP_K environment variable, using default: {e}")
        top_k = 5
    
    try:
        ingestor = ChatIngestor(root_path)
        num_conversations = ingestor.ingest_all()
        
        if num_conversations > 0:
            success = ingestor.export_gems_to_markdown(output_file, top_k=top_k)
            if success:
                logger.info("Export completed successfully")
            else:
                logger.error("Export failed")
        else:
            logger.warning("No conversations ingested")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)

