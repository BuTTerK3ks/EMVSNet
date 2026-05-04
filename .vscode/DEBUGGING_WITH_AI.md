# Best Practices: Using AI Agent with Debugger

## What the AI Can See
✅ **Can See:**
- Running processes (debugpy, Python processes)
- Code files and their contents
- Terminal output (if redirected to console)
- Log files written to disk
- Exception stack traces in terminal

❌ **Cannot See:**
- Debugger UI state (breakpoints, current line, variables, call stack)
- Variable values at runtime
- Debug console output directly
- Internal debugger state

## Best Practices for Collaboration

### 1. **Share Debugger State Verbally**
When paused at a breakpoint, share:
- **Current line number** and file
- **Variable values** from Variables panel (copy/paste)
- **Call stack** from Call Stack panel
- **Exception details** if any
- **What you expected vs. what happened**

### 2. **Use Logging for Persistent Debug Info**
Add logging that writes to files the AI can read:

```python
import logging

# Set up file logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

# In your code, log important state:
logger.debug(f"Variable x = {x}, shape = {x.shape}")
logger.debug(f"Call stack: {traceback.format_stack()}")
```

### 3. **Use Debug Console for Quick Inspection**
When paused, use the Debug Console to:
- Evaluate expressions: `print(variable_name)`
- Inspect objects: `dir(object)`
- Check values: `variable_name.shape`, `variable_name.dtype`
- Then copy the output to share with AI

### 4. **Capture Exception Details**
Add exception logging to catch errors:

```python
import traceback
import sys

try:
    # your code
except Exception as e:
    error_details = {
        'exception': str(e),
        'type': type(e).__name__,
        'traceback': traceback.format_exc(),
        'locals': {k: str(v) for k, v in locals().items()}
    }
    # Write to file
    with open('error.log', 'w') as f:
        import json
        json.dump(error_details, f, indent=2)
    raise
```

### 5. **Use Conditional Breakpoints with Logging**
Instead of just stopping, log and continue:

```python
# In your code, add conditional logging
if some_condition:
    import json
    debug_info = {
        'batch_idx': batch_idx,
        'loss': float(loss),
        'variable_shapes': {k: list(v.shape) for k, v in locals().items() if hasattr(v, 'shape')}
    }
    with open('debug_state.json', 'w') as f:
        json.dump(debug_info, f, indent=2)
```

### 6. **Share Terminal Output**
The AI can read terminal output. Make sure:
- `redirectOutput: true` is set (already in your config)
- `console: "integratedTerminal"` is set (already in your config)
- Share terminal output when asking for help

### 7. **Use Debugger Features Effectively**
- **Watch expressions**: Add variables to watch, then share their values
- **Debug console**: Evaluate expressions and share results
- **Exception breakpoints**: Break on exceptions, then share the state
- **Conditional breakpoints**: Break only when conditions are met

### 8. **Create Debug Helper Scripts**
Create a helper script to dump debug state:

```python
# debug_helpers.py
def dump_state(locals_dict, filename='debug_state.json'):
    """Dump current local variables to JSON file"""
    import json
    import numpy as np
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return {
                'type': type(obj).__name__,
                'shape': list(obj.shape),
                'dtype': str(obj.dtype),
                'min': float(obj.min()) if hasattr(obj, 'min') else None,
                'max': float(obj.max()) if hasattr(obj, 'max') else None,
            }
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    serializable = {k: convert_to_serializable(v) 
                   for k, v in locals_dict.items() 
                   if not k.startswith('_')}
    
    with open(filename, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"Debug state saved to {filename}")
```

Then in your code:
```python
from debug_helpers import dump_state
# At breakpoint:
dump_state(locals())
```

## Example Workflow

1. **Set breakpoint** at suspicious line
2. **Run debugger** until it pauses
3. **Inspect variables** in Debug Console:
   ```
   print(f"loss: {loss}, shape: {loss.shape}")
   print(f"sample keys: {sample.keys()}")
   ```
4. **Copy output** and share with AI
5. **Or dump state** using helper function
6. **Share the file** or its contents with AI

## Quick Tips

- **Be specific**: "Paused at line 284, variable `loss` is NaN" is better than "it's broken"
- **Share context**: Include relevant code snippets and what you're trying to achieve
- **Use file logging**: Write debug info to files the AI can read directly
- **Leverage terminal**: Since `redirectOutput: true`, terminal output is visible to AI
- **Exception details**: Always share full exception tracebacks

