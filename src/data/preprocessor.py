import re
import logging

logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, lowercase: bool = True, remove_special_chars: bool = True):
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
    
    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_special_chars:
            text = re.sub(r'[^а-яa-z0-9\s]', '', text)
        
        text = ' '.join(text.split())
        
        return text
