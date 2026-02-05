"""Code Execution Agent with approval gates and sandboxing.

Executes code safely with human approval for risky operations.
"""

from typing import Dict, Any, Optional  
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import ast
import subprocess
import sys
import signal
import threading
from state.state import OrchestratorState
from llm.router import LLMRouter

# NOTE: `resource` is a Unix-only stdlib module (not available on Windows).
try:
    import resource  # type: ignore
except Exception:  # pragma: no cover
    resource = None


class CodeValidator:
    """Validates code for safety before execution."""
    
    # Dangerous imports/operations
    DANGEROUS_IMPORTS = [
        'os.system', 'os.popen', 'os.exec', 'subprocess', 'eval', 'exec',
        'compile', '__import__', 'open', 'file', 'input', 'raw_input',
        'socket', 'urllib', 'requests', 'shutil', 'pickle', 'marshal'
    ]
    
    DANGEROUS_PATTERNS = [
        (r'import\s+os', 'OS module import'),
        (r'import\s+subprocess', 'Subprocess import'),
        (r'__import__', 'Dynamic import'),
        (r'eval\s*\(', 'Eval function'),
        (r'exec\s*\(', 'Exec function'),
        (r'open\s*\(', 'File operations'),
        (r'subprocess\.', 'Subprocess calls'),
        (r'os\.system', 'OS system calls'),
        (r'\.remove\s*\(', 'File deletion'),
        (r'\.rmdir\s*\(', 'Directory deletion'),
        (r'socket\.', 'Network operations'),
    ]
    
    @classmethod
    def validate(cls, code: str) -> Dict[str, Any]:
        """Validate code for safety.
        
        Args:
            code: Python code string
            
        Returns:
            Validation result dict
        """
        issues = []
        risk_level = "low"
        
        # Check for dangerous patterns
        for pattern, description in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Potentially dangerous: {description}")
                risk_level = "high"
        
        # Try to parse Python syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
            risk_level = "medium"
        
        # Check for file system operations
        if any(op in code for op in ['open(', 'write(', 'delete', 'remove']):
            risk_level = "high" if risk_level == "low" else risk_level
            if "File operations detected" not in [i.split(':')[0] for i in issues]:
                issues.append("File operations detected")
        
        requires_approval = risk_level in ["medium", "high"] or len(issues) > 0
        
        return {
            "is_safe": len(issues) == 0 and risk_level == "low",
            "risk_level": risk_level,
            "issues": issues,
            "requires_approval": requires_approval
        }


class CodeExecutor:
    """Safe code executor with sandboxing and resource limits.
    
    WARNING: This is a simplified sandbox. For production, use Docker containers
    or restricted Python environments (PyPy sandbox, RestrictedPython, etc.).
    """
    
    @staticmethod
    def execute_safe(code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute code in a safe environment with resource limits.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Execution result dict with output, error, success
        """
        # In production, use proper sandboxing (Docker, restricted Python, etc.)
        # This version adds resource limits and timeout handling
        
        # Set resource limits (memory: 128MB, CPU: timeout seconds) where supported.
        if resource is not None:
            try:
                # Limit memory to 128MB
                resource.setrlimit(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
                # Limit CPU time
                resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
            except (ValueError, OSError):
                # Resource limits may not be available on all platforms / configurations
                pass
        
        try:
            # Create restricted globals - remove dangerous builtins
            safe_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'abs': abs,
                    'round': round,
                    'bool': bool,
                    'type': type,
                    'isinstance': isinstance,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    # Explicitly remove dangerous functions
                }
            }
            
            # Remove dangerous builtins
            dangerous_builtins = ['__import__', 'eval', 'exec', 'open', 'file', 'input', 'raw_input',
                                'compile', 'reload', '__builtins__']
            for dangerous in dangerous_builtins:
                if dangerous in safe_globals['__builtins__']:
                    del safe_globals['__builtins__'][dangerous]
            
            # Execute with timeout using threading
            exec_globals = safe_globals.copy()
            exec_locals = {}
            
            # Capture stdout
            import io
            from contextlib import redirect_stdout
            
            output_buffer = io.StringIO()
            execution_error = None
            
            def execute_code():
                nonlocal execution_error
                try:
                    with redirect_stdout(output_buffer):
                        exec(code, exec_globals, exec_locals)
                except Exception as e:
                    execution_error = str(e)
            
            # Execute in a thread with timeout
            thread = threading.Thread(target=execute_code)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout)
            
            if thread.is_alive():
                # Timeout occurred
                return {
                    "success": False,
                    "output": output_buffer.getvalue(),
                    "error": f"Code execution timeout after {timeout} seconds"
                }
            
            if execution_error:
                return {
                    "success": False,
                    "output": output_buffer.getvalue(),
                    "error": execution_error
                }
            
            output = output_buffer.getvalue()
            return {
                "success": True,
                "output": output,
                "error": None
            }
                
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}"
            }


class CodeAgent:
    """Code execution agent with approval gates."""
    
    def __init__(
        self,
        llm_router: LLMRouter,
        temperature: float = 0.1,
        max_retries: int = 2
    ):
        """Initialize code agent.
        
        Args:
            llm_model: LLM model for code generation
            temperature: Temperature for generation
            max_retries: Maximum retry attempts
        """
        self._temperature = temperature
        self.llm_router = llm_router
        self.parser = StrOutputParser()
        self.max_retries = max_retries
        self.validator = CodeValidator()
        self.executor = CodeExecutor()
        
        self.code_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Python code generation assistant. Generate clean, safe Python code.

Rules:
- Generate only Python code
- Use safe, standard library functions
- Avoid file I/O, network operations, system calls
- Include print statements for output
- Return ONLY the code, no explanations or markdown
- Keep code simple and focused on the task"""),
            ("human", """User Request: {query}

Previous attempts (if any):
{previous_attempts}

Generate Python code:""")
        ])

    def generate_code(self, state: OrchestratorState) -> str:
        """Generate Python code from user request.
        
        Args:
            state: Orchestrator state
            
        Returns:
            Generated Python code
        """
        # Build previous attempts context
        previous_attempts = ""
        if state.get("retry_count", 0) > 0:
            previous_attempts = f"Previous code: {state.get('code_to_execute', '')}\n"
            previous_attempts += f"Errors: {', '.join(state.get('errors', [])[-2:])}"
        
        msgs = self.code_generation_prompt.format_messages(
            query=state["user_query"],
            previous_attempts=previous_attempts,
        )
        resp = self.llm_router.invoke(msgs, state=state, temperature=self._temperature)
        code = getattr(resp, "content", str(resp))
        
        # Clean code (remove markdown code blocks if present)
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        code = code.strip()
        
        return code
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate generated code.
        
        Args:
            code: Python code string
            
        Returns:
            Validation result
        """
        return self.validator.validate(code)
    
    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute code agent workflow with approval gates.
        
        Args:
            state: Orchestrator state
            
        Returns:
            Updated state
        """
        # CRITICAL: enforce invariants at node entry (prevents KeyError-class failures)
        from state.normalize import normalize_state, ensure_metadata, ensure_errors, ensure_intent
        state = normalize_state(state)  # type: ignore[arg-type]
        ensure_metadata(state)
        ensure_errors(state)
        ensure_intent(state)
        
        retry_count = state.get("retry_count", 0)
        
        try:
            # Check if we are resuming from an approval with existing code
            if state.get("approved") and state.get("code_to_execute"):
                 code = state["code_to_execute"]
                 # Skip generation and go straight to execution
                 state["execution_status"] = "executing"
                 result = self.executor.execute_safe(code)
                 
                 if result["success"]:
                     state["execution_result"] = result["output"]
                     state["execution_status"] = "completed"
                     state["final_answer"] = f"Code executed successfully (Approved):\n\n```python\n{code}\n```\n\nOutput:\n{result['output']}"
                     state["confidence_score"] = 0.90
                 else:
                     raise Exception(result["error"])
                 
                 # Add to conversation history
                 state["messages"].append({
                     "role": "assistant",
                     "content": state["final_answer"],
                     "metadata": {"agent": "code", "code": code, "approved": True}
                 })
                 return state

            # Generate code
            code = self.generate_code(state)
            state["code_to_execute"] = code
            
            # Validate code
            validation = self.validate_code(code)
            state["tool_outputs"]["code_validation"] = validation
            
            # Check if approval is required
            if validation["requires_approval"]:
                state["approval_required"] = True
                state["approval_reason"] = f"Code validation issues: {', '.join(validation['issues'])}"
                state["risk_level"] = validation["risk_level"]
                state["execution_status"] = "requires_approval"
                return state
            
            # Check if approved (if previously required)
            if state.get("approval_required") and not state.get("approved", False):
                state["execution_status"] = "pending"
                return state
            
            # Execute code
            state["execution_status"] = "executing"
            result = self.executor.execute_safe(code)
            
            if result["success"]:
                state["execution_result"] = result["output"]
                state["execution_status"] = "completed"
                state["final_answer"] = f"Code executed successfully:\n\n```python\n{code}\n```\n\nOutput:\n{result['output']}"
                state["confidence_score"] = 0.85
            else:
                raise Exception(result["error"])
            
            # Add to conversation history
            state["messages"].append({
                "role": "assistant",
                "content": state["final_answer"],
                "metadata": {"agent": "code", "code": code}
            })
            
        except Exception as e:
            error_msg = str(e)
            # Ensure errors list exists before append
            if "errors" not in state or not isinstance(state.get("errors"), list):
                state["errors"] = []
            state["errors"].append(error_msg)
            
            # Retry logic
            if retry_count < self.max_retries:
                state["retry_count"] = retry_count + 1
                state["execution_status"] = "failed"
                state["should_continue"] = True  # Signal to retry
            else:
                state["execution_status"] = "failed"
                state["final_answer"] = f"Failed to execute code after {self.max_retries} attempts. Errors: {', '.join(state.get('errors', []))}"
        
        return state

