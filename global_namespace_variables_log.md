# Global Namespace Variables Analysis
## Flexible CNN Architecture for Medical Physics Codebase

**Analysis Date:** 2024
**Scope:** All Python files excluding `main_run_functions` directory

---

## Summary

This document logs global namespace variables that need to be addressed in the codebase. These are variables that are:
1. Used but not passed as function parameters
2. Defined at module level and potentially modified
3. Referenced without explicit imports or definitions

---

## Files Analyzed

### Classes
- `classes/dataset.py`
- `classes/generators.py`
- `classes/discriminators.py`

### Functions - Helper
- `functions/helper/metrics.py`
- `functions/helper/loss_functions.py`
- `functions/helper/display_images.py`
- `functions/helper/weights_init.py`
- `functions/helper/timing.py`
- `functions/helper/cropping.py`
- `functions/helper/reconstruction_projection.py`

### Functions - Setup
- `functions/setup_notebook/setup_notebook.py`

### Config Files
- `config_net_dicts/search_spaces.py`
- Other config files in `config_net_dicts/`

---

## Detailed Findings

### 1. `functions/setup_notebook/setup_notebook.py`

**Issue Found:**
- **Line 39:** Variable `device` is created but never returned or used
  - Created in `setup_project_dirs()` function: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`
  - **Status:** Unused variable that should either be returned or removed

**Recommendation:**
- Either return `device` from `setup_project_dirs()` if it's needed, or remove it if not used

---

### 2. `classes/dataset.py`

**Status:** ✅ **NO ISSUES FOUND**
- All variables are properly scoped within functions
- No global namespace pollution detected
- Functions properly accept parameters

---

### 3. `classes/generators.py`

**Status:** ✅ **NO ISSUES FOUND**
- All variables are properly scoped within functions and class methods
- Config dictionary values are properly accessed via parameters
- No global namespace pollution detected

---

### 4. `classes/discriminators.py`

**Status:** ✅ **NO ISSUES FOUND**
- All variables are properly scoped within class methods
- Config dictionary values are properly accessed via parameters
- No global namespace pollution detected

---

### 5. `functions/helper/metrics.py`

**Status:** ✅ **NO ISSUES FOUND**
- All functions properly accept parameters
- No reliance on global variables
- Helper functions are well-contained

---

### 6. `functions/helper/loss_functions.py`

**Status:** ✅ **NO ISSUES FOUND**
- All functions properly accept parameters
- No reliance on global variables
- Loss calculation functions are self-contained

---

### 7. `functions/helper/display_images.py`

**Status:** ✅ **NO ISSUES FOUND**
- All functions properly accept parameters
- No reliance on global variables
- Visualization functions are self-contained

---

### 8. `functions/helper/weights_init.py`

**Status:** ✅ **NO ISSUES FOUND**
- Function properly accepts parameter (`m`)
- No global variables used

---

### 9. `functions/helper/timing.py`

**Status:** ✅ **NO ISSUES FOUND**
- Function properly accepts parameters
- No global variables used

---

### 10. `functions/helper/cropping.py`

**Status:** ✅ **NO ISSUES FOUND**
- All functions properly accept parameters
- No global variables used
- Uses `__all__` for proper module exports

---

### 11. `functions/helper/reconstruction_projection.py`

**Status:** ✅ **NO ISSUES FOUND**
- All functions properly accept parameters
- Config dictionary values accessed via parameters
- No global variables used

---

### 12. `config_net_dicts/search_spaces.py`

**Status:** ✅ **CONFIGURATION DICTIONARIES (ACCEPTABLE)**
- Module-level dictionaries (`config_RAY_SI`, `config_RAY_IS`, etc.) are **intentional configuration constants**
- These are meant to be imported and used as configuration templates
- **Action Required:** None - this is expected behavior for config files

**Note:** These dictionaries are:
- `config_RAY_SI`
- `config_RAY_IS`
- `config_RAY_SUP`
- `config_RAY_GAN`
- `config_GAN_RAY_cycle`
- `config_SUP_RAY_cycle`

---

## Issues Requiring Action

### Critical Issues

1. **`setup_notebook.py` - Unused variable `device`**
   - **File:** `functions/setup_notebook/setup_notebook.py`
   - **Line:** 39
   - **Issue:** Variable is created but never used or returned
   - **Recommendation:** 
     - Option A: Return it from the function if needed by callers
     - Option B: Remove it if not needed
   - **Priority:** Medium

---

## Notes on `main_run_functions` Directory

**Note:** The `main_run_functions` directory was excluded from this analysis as requested. However, initial inspection suggests these files (`train_test_visualize_SUP.py` and `tune.py`) contain many variables that appear to be global namespace variables, including:

- `train_SI`
- `run_mode`
- `tune_even_reporting`
- `tune_iter_per_report`
- `tune_restore`
- `tune_dataframe_path`
- `train_display_step`
- `tune_for`
- `tune_minutes`
- `tune_scheduler`
- `train_Supervisory_Sym`
- `train_test_GAN`
- `train_test_CYCLE`
- `num_CPUs`
- `num_GPUs`
- `config`
- `tune_storage_dirPath`
- `tune_exp_name`
- `run_config`
- `optim_metric`
- `min_max`
- `scheduler`
- `HyperOptSearch`

These will need to be addressed in a future refactoring pass.

---

## Recommendations

### General Best Practices

1. **Parameter Passing:** All functions should receive necessary values as parameters rather than relying on global state
2. **Configuration Objects:** Consider using configuration classes or dataclasses instead of dictionaries when appropriate
3. **Return Values:** Ensure all computed values that might be needed are returned from functions
4. **Constants:** Module-level constants (like config dictionaries) are acceptable, but should be clearly documented

### Next Steps

1. Fix the unused `device` variable in `setup_notebook.py`
2. Review and refactor `main_run_functions` directory (as planned)
3. Consider adding type hints to improve code clarity
4. Add docstrings documenting expected parameter sources for functions

---

## Conclusion

**Overall Status:** The codebase (excluding `main_run_functions`) is generally well-structured with minimal global namespace pollution. The only issue found is a minor unused variable in `setup_notebook.py`. Configuration dictionaries at module level are acceptable as they represent intentional configuration constants.

**Total Issues Found:** 1 (minor)

**Files with Issues:** 1 out of 23 analyzed files
