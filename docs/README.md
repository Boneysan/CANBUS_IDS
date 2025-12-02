# CAN-IDS Documentation

Comprehensive documentation for the CAN-IDS project.

## Documentation Files

### Installation & Setup
- `installation.md` - General installation guide
- `rpi4_installation.md` - Raspberry Pi 4 specific setup
- `configuration.md` - Configuration reference

### Usage & Development
- `rules_guide.md` - Detection rule writing guide
- `api_reference.md` - API documentation
- `troubleshooting.md` - Common issues and solutions
- `performance.md` - Performance tuning guide

## Building Documentation

If using Sphinx for API docs:

```bash
cd docs
make html
```

View generated docs at `docs/_build/html/index.html`

## Contributing to Documentation

When adding new features, please update relevant documentation:

1. Add docstrings to all public functions/classes
2. Update configuration examples
3. Add troubleshooting entries for common issues
4. Update API reference

## Documentation Standards

- Use Markdown for general documentation
- Use reStructuredText for Sphinx API docs
- Include code examples where applicable
- Keep language clear and concise
- Test all commands and examples

## TODO: Complete Documentation

Priority documentation to create:
- [ ] Complete installation guide
- [ ] Configuration parameter reference
- [ ] Rule writing tutorial
- [ ] ML model training guide
- [ ] Troubleshooting guide
- [ ] Performance benchmarks
- [ ] API reference (auto-generated from docstrings)