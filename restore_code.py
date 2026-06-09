import json
import os

log_file = r'C:\Users\kayam\.gemini\antigravity\brain\43389105-80e3-421b-a33d-df8aece26958\.system_generated\logs\transcript.jsonl'

files_state = {
    r'c:\Users\kayam\Documents\Projeler\DeepSign-TID\app\server.py': '',
    r'c:\Users\kayam\Documents\Projeler\DeepSign-TID\app\pytorch_predictor.py': '',
}

# First, read the baseline content of the files as they currently exist on disk
try:
    with open(r'c:\Users\kayam\Documents\Projeler\DeepSign-TID\app\server.py', 'r', encoding='utf-8') as f:
        files_state[r'c:\Users\kayam\Documents\Projeler\DeepSign-TID\app\server.py'] = f.read()
    with open(r'c:\Users\kayam\Documents\Projeler\DeepSign-TID\app\pytorch_predictor.py', 'r', encoding='utf-8') as f:
        files_state[r'c:\Users\kayam\Documents\Projeler\DeepSign-TID\app\pytorch_predictor.py'] = f.read()
except Exception:
    pass

def apply_replacement(content, target, replacement, allow_multiple):
    if allow_multiple:
        return content.replace(target, replacement)
    else:
        return content.replace(target, replacement, 1)

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line)
        except Exception:
            continue
            
        step = obj.get('step_index', 0)
        
        if step > 90: # Stop before step 92
            break
            
        if obj.get('type') == 'PLANNER_RESPONSE':
            tool_calls = obj.get('tool_calls', [])
            for tc in tool_calls:
                name = tc.get('name')
                args = tc.get('args', {})
                
                if name == 'write_to_file':
                    path = args.get('TargetFile', '')
                    code = args.get('CodeContent', '')
                    files_state[path] = code
                
                elif name == 'replace_file_content':
                    path = args.get('TargetFile', '')
                    target = args.get('TargetContent', '')
                    repl = args.get('ReplacementContent', '')
                    
                    if path in files_state and files_state[path]:
                        files_state[path] = files_state[path].replace(target, repl, 1)
                        
                elif name == 'multi_replace_file_content':
                    path = args.get('TargetFile', '')
                    chunks_str = args.get('ReplacementChunks', '[]')
                    try:
                        if isinstance(chunks_str, str):
                            chunks = json.loads(chunks_str)
                        else:
                            chunks = chunks_str
                        if path in files_state and files_state[path]:
                            content = files_state[path]
                            for c in chunks:
                                target = c['TargetContent']
                                repl = c['ReplacementContent']
                                content = apply_replacement(content, target, repl, c.get('AllowMultiple', False))
                            files_state[path] = content
                    except Exception as e:
                        print("Error parsing chunks:", e)

# Write the reconstructed files back to the app directory!
for path, content in files_state.items():
    if not path or not content:
        continue
    # write to disk
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Restored {path}")
    except Exception as e:
        print(f"Failed to write {path}: {e}")
