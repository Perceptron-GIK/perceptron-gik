# Security Considerations for Perceptron GIK

## Known Vulnerabilities

### PyTorch torch.load Vulnerability (CVE-2024-XXXXX)

**Status**: Present in PyTorch < 2.6.0  
**Severity**: High  
**CVSS Score**: TBD

#### Description
The `torch.load` function in PyTorch versions < 2.6.0 has a remote code execution vulnerability, even when using `weights_only=True`. The vulnerability allows arbitrary code execution when loading malicious `.pt` files.

#### Current Usage
The codebase uses `torch.load` in the following locations:
- `pretraining.py:419` - Loading preprocessed datasets (weights_only=False)
- `pretraining.py:539` - Loading preprocessed datasets (weights_only=False)
- `ml/models/basic_nn.py:570` - Loading model weights

#### Mitigation Strategy

**Short-term** (Current State):
- Only load `.pt` files from trusted sources
- Do not load user-uploaded or external `.pt` files
- Validate file integrity before loading

**Long-term** (Recommended):
1. **Upgrade PyTorch to version 2.6.0 or later** when available
   ```bash
   pip install torch>=2.6.0
   ```

2. **For model weights only**, consider using safer alternatives:
   ```python
   # Instead of:
   model.load_state_dict(torch.load(path))
   
   # Use:
   model.load_state_dict(torch.load(path, weights_only=True))  # After upgrading to 2.6.0
   ```

3. **For dataset files**, consider alternative serialization formats:
   - Use numpy's `.npz` format for arrays
   - Use JSON/CSV for metadata
   - Use HDF5 for large datasets

#### Risk Assessment
- **Current Risk Level**: Medium
  - Files are generated internally by trusted preprocessing pipeline
  - No external file loading from untrusted sources
  - Development/research environment, not production

- **Production Risk Level**: High
  - Would require secure file handling if deployed
  - Must upgrade PyTorch before production deployment

#### Action Items
- [ ] Monitor for PyTorch 2.6.0 release
- [ ] Test compatibility with PyTorch 2.6.0 when available
- [ ] Update requirements to specify minimum PyTorch version
- [ ] Add file integrity checks for loaded `.pt` files
- [ ] Consider migrating to safer serialization formats for datasets

## Other Security Considerations

### Data Validation
- ✅ All input data is validated and sanitized
- ✅ NaN/Inf values are handled properly
- ✅ Class labels are validated against known vocabulary

### Model Security
- ✅ No user-controlled input affects model architecture
- ✅ Model parameters are validated during creation
- ✅ Input dimensions are checked before forward pass

### Test Security
- ✅ Tests use isolated temporary directories
- ✅ No persistent test data in repository
- ✅ Fixed random seeds for reproducibility

## Reporting Security Issues

If you discover a security vulnerability, please report it to the project maintainers directly rather than opening a public issue.

## References
- PyTorch Security Advisory: https://github.com/pytorch/pytorch/security/advisories
- GHSA Database: https://github.com/advisories
