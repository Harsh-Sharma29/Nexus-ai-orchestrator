"""Multi-tenant configuration management.

Handles tenant-specific settings and isolation.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TenantTier(str, Enum):
    """Tenant subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TenantConfig:
    """Configuration for a tenant."""
    tenant_id: str
    tier: TenantTier = TenantTier.FREE
    max_documents: int = 10
    max_query_length: int = 1000
    allowed_agents: list = None
    db_connection: Optional[str] = None
    db_schema: Optional[Dict[str, Any]] = None
    llm_model: str = "gpt-4o-mini"
    enable_code_execution: bool = False
    enable_sql_execution: bool = False
    require_approval_for_risky_ops: bool = True
    max_retries: int = 3
    
    def __post_init__(self):
        """Set defaults for allowed agents based on tier."""
        if self.allowed_agents is None:
            if self.tier == TenantTier.FREE:
                self.allowed_agents = ["chat", "rag"]
            elif self.tier == TenantTier.BASIC:
                self.allowed_agents = ["chat", "rag", "research"]
            elif self.tier == TenantTier.PRO:
                self.allowed_agents = ["chat", "rag", "research", "sql", "code"]
            else:  # ENTERPRISE
                self.allowed_agents = ["chat", "rag", "research", "sql", "code"]


class TenantConfigManager:
    """Manages tenant configurations with isolation."""
    
    def __init__(self):
        """Initialize config manager."""
        self.configs: Dict[str, TenantConfig] = {}
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default tenant configurations."""
        # Default tenant
        self.configs["default"] = TenantConfig(
            tenant_id="default",
            tier=TenantTier.PRO,
            allowed_agents=["chat", "rag", "research", "sql", "code"],
            enable_code_execution=True,
            enable_sql_execution=True
        )
    
    def get_config(self, tenant_id: str) -> TenantConfig:
        """Get configuration for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Tenant configuration (creates default if not exists)
        """
        if tenant_id not in self.configs:
            # Create default config for new tenant
            self.configs[tenant_id] = TenantConfig(tenant_id=tenant_id)
        
        return self.configs[tenant_id]
    
    def set_config(self, tenant_id: str, config: TenantConfig):
        """Set configuration for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            config: Tenant configuration
        """
        self.configs[tenant_id] = config
    
    def update_config(self, tenant_id: str, **kwargs):
        """Update specific configuration fields.
        
        Args:
            tenant_id: Tenant identifier
            **kwargs: Configuration fields to update
        """
        config = self.get_config(tenant_id)
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    def is_agent_allowed(self, tenant_id: str, agent_name: str) -> bool:
        """Check if an agent is allowed for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            agent_name: Agent name (rag, sql, code, research, chat)
            
        Returns:
            True if agent is allowed
        """
        config = self.get_config(tenant_id)
        return agent_name.lower() in [a.lower() for a in config.allowed_agents]
    
    def validate_request(self, tenant_id: str, query: str, intent: str) -> Tuple[bool, Optional[str]]:
        """Validate a request against tenant limits.
        
        Args:
            tenant_id: Tenant identifier
            query: User query
            intent: Classified intent
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        config = self.get_config(tenant_id)
        
        # Check query length
        if len(query) > config.max_query_length:
            return False, f"Query exceeds maximum length of {config.max_query_length} characters"
        
        # Check agent access
        if not self.is_agent_allowed(tenant_id, intent):
            return False, f"Agent '{intent}' is not available for your tier ({config.tier.value})"
        
        # Check specific agent permissions
        if intent == "code" and not config.enable_code_execution:
            return False, "Code execution is not enabled for your tenant"
        
        if intent == "sql" and not config.enable_sql_execution:
            return False, "SQL execution is not enabled for your tenant"
        
        return True, None

