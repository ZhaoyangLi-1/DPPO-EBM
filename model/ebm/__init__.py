"""
Energy-Based Model (EBM) package for DPPO integration.
"""

from .ebm import EBM as OriginalEBM, EBMWrapper

# Create a wrapper class that can handle the configuration parameters
class EBMConfigWrapper:
    """Wrapper class that creates proper config structure for EBM."""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=256, num_layers=3, **kwargs):
        #b Create a config object with the expected structure
        class Config:
            def __init__(self):
                # Create the nested ebm config structure
                ebm_config = type('obj', (object,), {
                    'embed_dim': hidden_dim,
                    'state_dim': obs_dim,
                    'action_dim': action_dim,
                    'nhead': kwargs.get('nhead', 8),
                    'depth': num_layers,
                    'dropout': kwargs.get('dropout', 0.1),
                    'use_cls_token': kwargs.get('use_cls_token', False),
                    'num_views': kwargs.get('num_views', None)
                })()
                
                # Set the ebm attribute
                self.ebm = ebm_config
        
        self.config = Config()
        self.ebm = OriginalEBM(self.config)
    
    def forward(self, k_idx, views=None, poses=None, actions=None, **kwargs):
        """Forward pass that matches the EBM interface."""
        return self.ebm(k_idx=k_idx, views=views, poses=poses, actions=actions)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying EBM model."""
        return getattr(self.ebm, name)
    
    def to(self, device):
        """Delegate device movement to the underlying EBM model."""
        self.ebm = self.ebm.to(device)
        return self
    
    def parameters(self):
        """Delegate parameters to the underlying EBM model."""
        return self.ebm.parameters()
    
    def state_dict(self):
        """Delegate state_dict to the underlying EBM model."""
        try:
            return self.ebm.state_dict()
        except TypeError as e:
            # Handle PyTorch version compatibility issues
            import logging
            logging.warning(f"EBM state_dict() failed with error: {e}")
            # Fallback to manual state dict creation
            state_dict = {}
            for name, param in self.ebm.named_parameters():
                state_dict[name] = param.data.clone()
            for name, buffer in self.ebm.named_buffers():
                state_dict[name] = buffer.data.clone()
            return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Delegate load_state_dict to the underlying EBM model."""
        return self.ebm.load_state_dict(state_dict, strict=strict)

# Export the wrapper as EBM for Hydra compatibility
EBM = EBMConfigWrapper

__all__ = ['EBM', 'EBMWrapper']
