import types
import joblib

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def hash_codeobj(code):
    """Return hashed version of a code object"""
    bytecode = code.co_code
    consts = code.co_consts
    
    consts = [hash_codeobj(c) if isinstance(c, types.CodeType) else c 
              for c in consts]
    
    return joblib.hash((bytecode, consts))
