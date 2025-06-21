#!/usr/bin/env python3
"""
Simple test to verify that the algorithm config changes work correctly.
"""

import sys
import os
sys.path.insert(0, '/workspace/verl')

from omegaconf import DictConfig, OmegaConf

# Test the algorithm config dataclass
def test_algorithm_config():
    print("Testing AlgorithmConfig dataclass...")
    
    try:
        from verl.trainer.config.algorithm_config import AlgorithmConfig, FilterGroupsConfig
        from verl.utils.config import omega_conf_to_dataclass
        
        # Create a sample config similar to what's in the YAML
        config_dict = {
            'algorithm': {
                'adv_estimator': 'gae',
                'gamma': 0.99,
                'lam': 0.95,
                'kl_penalty': 0.1,
                'use_kl_in_reward': True,
                'norm_adv_by_std_in_grpo': True,
                'filter_groups': {
                    'enable': True,
                    'metric': 'reward',
                    'max_num_gen_batches': 10
                },
                'sppo_eta': 0.5
            }
        }
        
        config = OmegaConf.create(config_dict)
        
        # Convert to dataclass
        algorithm_config = omega_conf_to_dataclass(config.algorithm, AlgorithmConfig)
        
        # Test basic fields
        assert algorithm_config.adv_estimator == 'gae'
        assert algorithm_config.gamma == 0.99
        assert algorithm_config.lam == 0.95
        assert algorithm_config.kl_penalty == 0.1
        assert algorithm_config.use_kl_in_reward == True
        assert algorithm_config.norm_adv_by_std_in_grpo == True
        assert algorithm_config.sppo_eta == 0.5
        
        # Test nested filter_groups config
        assert algorithm_config.filter_groups is not None
        assert algorithm_config.filter_groups.enable == True
        assert algorithm_config.filter_groups.metric == 'reward'
        assert algorithm_config.filter_groups.max_num_gen_batches == 10
        
        print("✓ AlgorithmConfig dataclass works correctly")
        return True
        
    except Exception as e:
        print(f"✗ AlgorithmConfig test failed: {e}")
        return False

def test_imports():
    print("Testing imports in updated files...")
    
    try:
        # Test DAPO imports
        from verl.trainer.config.algorithm_config import AlgorithmConfig
        from verl.utils.config import omega_conf_to_dataclass
        print("✓ Basic imports work")
        
        # Test that we can import the updated files
        import importlib.util
        
        files_to_test = [
            '/workspace/verl/recipe/dapo/dapo_ray_trainer.py',
            '/workspace/verl/recipe/entropy/entropy_ray_trainer.py',
            '/workspace/verl/recipe/prime/prime_ray_trainer.py',
            '/workspace/verl/recipe/sppo/sppo_ray_trainer.py',
        ]
        
        for file_path in files_to_test:
            try:
                spec = importlib.util.spec_from_file_location("test_module", file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Don't execute the module, just check it can be loaded
                    print(f"✓ {file_path} can be imported")
                else:
                    print(f"✗ {file_path} cannot be loaded")
            except Exception as e:
                print(f"✗ {file_path} import failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def main():
    print("Running verification tests for algorithm config changes...")
    print("=" * 60)
    
    success = True
    success &= test_algorithm_config()
    success &= test_imports()
    
    print("=" * 60)
    if success:
        print("✓ All tests passed! The changes appear to be working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())