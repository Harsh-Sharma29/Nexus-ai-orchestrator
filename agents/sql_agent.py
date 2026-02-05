"""SQL Agent with validation, retries, and safety checks.

Generates and executes SQL queries with validation and human approval for risky operations.
"""

from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword, DML
from state.state import OrchestratorState
from llm.router import LLMRouter


class SQLValidator:
    """Validates SQL queries for safety and correctness."""
    
    # Dangerous SQL patterns
    DANGEROUS_PATTERNS = [
        (r'\bDROP\s+TABLE\b', 'DROP TABLE operations'),
        (r'\bDELETE\s+FROM\b', 'DELETE operations'),
        (r'\bTRUNCATE\b', 'TRUNCATE operations'),
        (r'\bALTER\s+TABLE\b', 'ALTER TABLE operations'),
        (r'\bCREATE\s+TABLE\b', 'CREATE TABLE operations'),
        (r'\bINSERT\s+INTO\b', 'INSERT operations'),
        (r'\bUPDATE\s+.*\s+SET\b', 'UPDATE operations'),
        (r';\s*DROP', 'Multiple statements with DROP'),
        (r';\s*DELETE', 'Multiple statements with DELETE'),
        (r'--', 'SQL comments (potential injection)'),
        (r'/\*.*\*/', 'SQL block comments'),
    ]
    
    # Read-only safe patterns
    SAFE_PATTERNS = [
        r'\bSELECT\b',
        r'\bWITH\b',  # CTEs
        r'\bFROM\b',
        r'\bWHERE\b',
        r'\bGROUP\s+BY\b',
        r'\bORDER\s+BY\b',
        r'\bHAVING\b',
        r'\bJOIN\b',
        r'\bLIMIT\b',
    ]
    
    @classmethod
    def validate(cls, sql: str) -> Dict[str, Any]:
        """Validate SQL query for safety using pattern matching and AST parsing.
        
        Args:
            sql: SQL query string
            
        Returns:
            Validation result dict with is_safe, risk_level, issues
        """
        sql_upper = sql.upper().strip()
        
        # Check for dangerous patterns
        issues = []
        risk_level = "low"
        
        for pattern, description in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                issues.append(f"Potentially dangerous: {description}")
                risk_level = "high"
        
        # AST-based validation for more accurate detection
        try:
            parsed = sqlparse.parse(sql)
            for statement in parsed:
                # Check for DML operations using AST
                tokens = statement.tokens
                for token in tokens:
                    if token.ttype is Keyword:
                        upper_token = token.value.upper()
                        if upper_token in ['DELETE', 'DROP', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'CREATE']:
                            if f"Dangerous operation: {upper_token}" not in issues:
                                issues.append(f"Dangerous operation detected via AST: {upper_token}")
                                risk_level = "high"
        except Exception as e:
            # If AST parsing fails, fall back to pattern matching
            issues.append(f"SQL parsing warning: {str(e)}")
            risk_level = "medium"
        
        # Check if it's read-only
        has_safe_pattern = any(re.search(pattern, sql_upper, re.IGNORECASE) 
                               for pattern in cls.SAFE_PATTERNS)
        
        # Also check AST for SELECT statements
        try:
            parsed = sqlparse.parse(sql)
            for statement in parsed:
                tokens = statement.tokens
                for token in tokens:
                    if token.ttype is Keyword and token.value.upper() == 'SELECT':
                        has_safe_pattern = True
                        break
        except Exception:
            pass  # Fall back to pattern matching
        
        if not has_safe_pattern and not issues:
            issues.append("Query doesn't appear to be a SELECT query")
            risk_level = "medium"
        
        # Check for multiple statements
        if sql.count(';') > 1:
            issues.append("Multiple statements detected")
            risk_level = "high"
        
        is_safe = len(issues) == 0 and has_safe_pattern
        
        return {
            "is_safe": is_safe,
            "risk_level": risk_level,
            "issues": issues,
            "requires_approval": risk_level in ["medium", "high"]
        }


class SQLAgent:
    """SQL agent with validation and retry logic."""
    
    def __init__(
        self,
        llm_router: LLMRouter,
        temperature: float = 0.0,
        max_retries: int = 3
    ):
        """Initialize SQL agent.
        
        Args:
            llm_model: LLM model for SQL generation
            temperature: Temperature for generation
            max_retries: Maximum retry attempts
        """
        self._temperature = temperature
        self.llm_router = llm_router
        self.parser = StrOutputParser()
        self.max_retries = max_retries
        self.validator = SQLValidator()
        
        self.sql_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert. Generate accurate, safe SQL queries.

Rules:
- Only generate SELECT queries (read-only)
- Use proper SQL syntax
- Include appropriate WHERE clauses for filtering
- Add LIMIT clauses for large result sets
- Use table and column names from the provided schema
- Never generate DROP, DELETE, UPDATE, INSERT, or ALTER statements
- Return ONLY the SQL query, no explanations"""),
            ("human", """Database Schema:
{schema}

User Query: {query}

Previous attempts (if any):
{previous_attempts}

Generate SQL query:""")
        ])

    def _get_schema_context(self, state: OrchestratorState) -> str:
        """Get database schema context.
        
        Args:
            state: Orchestrator state
            
        Returns:
            Schema description string
        """
        schema = state.get("db_schema")
        if schema:
            # Format schema dict as readable string
            schema_str = "Tables and columns:\n"
            for table, columns in schema.items():
                schema_str += f"\n{table}:\n"
                for col in columns:
                    schema_str += f"  - {col}\n"
            return schema_str
        
        # Default schema if not provided
        return "Schema information not available. Use common table/column names."
    
    def generate_sql(self, state: OrchestratorState) -> str:
        """Generate SQL query from user request.
        
        Args:
            state: Orchestrator state
            
        Returns:
            Generated SQL query
        """
        schema_context = self._get_schema_context(state)
        
        # Build previous attempts context
        previous_attempts = ""
        if state.get("retry_count", 0) > 0:
            previous_attempts = f"Previous SQL: {state.get('generated_sql', '')}\n"
            previous_attempts += f"Errors: {', '.join(state.get('errors', [])[-2:])}"
        
        msgs = self.sql_generation_prompt.format_messages(
            schema=schema_context,
            query=state["user_query"],
            previous_attempts=previous_attempts,
        )
        resp = self.llm_router.invoke(msgs, state=state, temperature=self._temperature)
        sql = getattr(resp, "content", str(resp))
        
        # Clean SQL (remove markdown code blocks if present)
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        sql = sql.strip()
        
        return sql
    
    def validate_sql(self, sql: str) -> Dict[str, Any]:
        """Validate generated SQL.
        
        Args:
            sql: SQL query string
            
        Returns:
            Validation result
        """
        return self.validator.validate(sql)
    
    def execute_sql(self, state: OrchestratorState, sql: str) -> str:
        """Execute SQL query (mock implementation - replace with actual DB connection).
        
        Args:
            state: Orchestrator state
            sql: SQL query to execute
            
        Returns:
            Query results as string
        """
        # In production, use actual database connection from state["db_connection"]
        # This is a mock implementation for demonstration
        
        db_connection = state.get("db_connection")
        if not db_connection:
            return "Error: No database connection configured."
        
        # Mock execution - replace with actual DB execution
        # Example: using sqlalchemy or similar
        try:
            # Placeholder for actual execution
            # result = execute_query(db_connection, sql)
            # return format_results(result)
            
            return f"[Mock] Executed query: {sql}\n[In production, this would execute against the database]"
        except Exception as e:
            raise Exception(f"SQL execution error: {str(e)}")
    
    def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute SQL agent workflow with retries.
        
        Args:
            state: Orchestrator state
            
        Returns:
            Updated state
        """
        # CRITICAL: enforce invariants at node entry (prevents KeyError-class failures)
        from state.normalize import normalize_state, ensure_metadata, ensure_errors,ensure_intent
        state = normalize_state(state)  # type: ignore[arg-type]
        ensure_metadata(state)
        ensure_errors(state)
        ensure_intent(state)
        
        retry_count = state.get("retry_count", 0)
        
        try:
            # Check if we are resuming from an approval with existing SQL
            if state.get("approved") and state.get("generated_sql"):
                 sql = state["generated_sql"]
                 # Skip generation and go straight to execution
                 state["execution_status"] = "executing"
                 result = self.execute_sql(state, sql)
                 state["execution_result"] = result
                 state["execution_status"] = "completed"
                 state["final_answer"] = f"Query executed successfully (Approved):\n\n```sql\n{sql}\n```\n\nResults:\n{result}"
                 state["confidence_score"] = 0.95
                 
                 # Add to conversation history
                 state["messages"].append({
                     "role": "assistant",
                     "content": state["final_answer"],
                     "metadata": {"agent": "sql", "sql": sql, "approved": True}
                 })
                 return state

            # Generate SQL
            sql = self.generate_sql(state)
            state["generated_sql"] = sql
            
            # Validate SQL
            validation = self.validate_sql(sql)
            state["sql_validation_result"] = validation
            
            # Check if approval is required
            if validation["requires_approval"]:
                state["approval_required"] = True
                state["approval_reason"] = f"SQL validation issues: {', '.join(validation['issues'])}"
                state["risk_level"] = validation["risk_level"]
                state["execution_status"] = "requires_approval"
                return state
            
            # Check if approved (if previously required)
            if state.get("approval_required") and not state.get("approved", False):
                state["execution_status"] = "pending"
                return state
            
            # Execute SQL
            state["execution_status"] = "executing"
            result = self.execute_sql(state, sql)
            state["execution_result"] = result
            state["execution_status"] = "completed"
            state["final_answer"] = f"Query executed successfully:\n\n```sql\n{sql}\n```\n\nResults:\n{result}"
            state["confidence_score"] = 0.9 if validation["is_safe"] else 0.7
            
            # Add to conversation history
            state["messages"].append({
                "role": "assistant",
                "content": state["final_answer"],
                "metadata": {"agent": "sql", "sql": sql}
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
                state["final_answer"] = f"Failed to generate or execute SQL after {self.max_retries} attempts. Errors: {', '.join(state.get('errors', []))}"
        
        return state

