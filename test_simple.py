#!/usr/bin/env python3
"""
Simple test to verify that the algorithm config changes work correctly.
"""

import sys
import os
sys.path.insert(0, '/workspace/verl')

from omegaconf import DictConfig, OmegaConf

# Test the algorithm config dataclass directly
def test_algorithm_config():
    print("Testing AlgorithmConfig dataclass...")
    
    try:
        # Import just the config modules without the full verl package
        sys.path.insert(0, '/workspace/verl/verl/trainer/config')
        sys.path.insert(0, '/workspace/verl/verl/utils')
        
        from algorithm_config import AlgorithmConfig, FilterGroupsConfig
        from config import omega_conf_to_dataclass
        
        # Create a sample config similar to what's in the YAML
        config_dict = {
            'adv_estimator': 'gae',
            'gamma': 0.99,
            'lam': 0.95,
            'kl_penalty': 'kl',
            'use_kl_in_reward': True,
            'norm_adv_by_std_in_grpo': True,
            'filter_groups': {
                'enable': True,
                'metric': 'reward',
                'max_num_gen_batches': 10
            },
            'sppo_eta': 0.5
        }
        
        config = OmegaConf.create(config_dict)
        
        # Convert to dataclass
        algorithm_config = omega_conf_to_dataclass(config, AlgorithmConfig)
        
        # Test basic fields
        assert algorithm_config.adv_estimator == 'gae'
        assert algorithm_config.gamma == 0.99
        assert algorithm_config.lam == 0.95
        assert algorithm_config.kl_penalty == 'kl'
        assert algorithm_config.use_kl_in_reward == True
        assert algorithm_config.norm_adv_by_std_in_grpo == True
        assert algorithm_config.sppo_eta == 0.5
        
        # Test nested filter_groups config
        assert algorithm_config.filter_groups is not None
        assert algorithm_config.filter_groups.enable == True
        assert algorithm_config.filter_groups.metric == 'reward'
        assert algorithm_config.filter_groups.max_num_gen_batches == 10
        
        print("✓ AlgorithmConfig dataclass works correctly")
        print(f"  - adv_estimator: {algorithm_config.adv_estimator}")
        print(f"  - gamma: {algorithm_config.gamma}")
        print(f"  - filter_groups.enable: {algorithm_config.filter_groups.enable}")
        print(f"  - sppo_eta: {algorithm_config.sppo_eta}")
        return True
        
    except Exception as e:
        print(f"✗ AlgorithmConfig test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_syntax():
    print("Testing file syntax...")
    
    files_to_check = [
        '/workspace/verl/recipe/dapo/dapo_ray_trainer.py',
        '/workspace/verl/recipe/entropy/entropy_ray_trainer.py', 
        '/workspace/verl/recipe/prime/prime_ray_trainer.py',
        '/workspace/verl/recipe/sppo/sppo_ray_trainer.py',
        '/workspace/verl/examples/split_placement/split_monkey_patch.py',
        '/workspace/verl/recipe/dapo/main_dapo.py',
        '/workspace/verl/recipe/entropy/main_entropy.py',
        '/workspace/verl/recipe/prime/main_prime.py',
        '/workspace/verl/recipe/sppo/main_sppo.py',
        '/workspace/verl/examples/split_placement/main_ppo_split.py',
    ]
    
    success = True
    for file_path in files_to_check:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for syntax errors
            compile(content, file_path, 'exec')
            print(f"✓ {file_path} has valid syntax")
            
            # Check that config.algorithm references are replaced
            if 'config.algorithm.' in content:
                print(f"⚠ {file_path} still contains config.algorithm references")
                success = False
            
        except SyntaxError as e:
            print(f"✗ {file_path} has syntax error: {e}")
            success = False
        except Exception as e:
            print(f"✗ {file_path} check failed: {e}")
            success = False
    
    return success

def main():
    print("Running verification tests for algorithm config changes...")
    print("=" * 60)
    
    success = True
    success &= test_algorithm_config()
    success &= test_file_syntax()
    
    print("=" * 60)
    if success:
        print("✓ All tests passed! The changes appear to be working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())