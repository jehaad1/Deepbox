# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |

## Reporting a Vulnerability

The Deepbox team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, report vulnerabilities by emailing: **[hi@jehaad.com](mailto:hi@jehaad.com)**

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- **Acknowledgment**: We'll acknowledge receipt of your vulnerability report within 48 hours
- **Updates**: We'll send you regular updates about our progress
- **Disclosure**: Once we've resolved the issue, we'll publicly disclose the vulnerability (with credit to you, if desired)

### Security Best Practices

When using Deepbox:

1. **Keep Dependencies Updated**: Although Deepbox has zero runtime dependencies, keep your development dependencies up to date
2. **Validate Input**: Always validate and sanitize user input before passing to Deepbox functions
3. **Use Type Safety**: Leverage TypeScript's type system to catch potential issues at compile time
4. **Review Code**: Review any code that processes untrusted data
5. **Monitor for Updates**: Watch the repository for security updates and patches

## Security Features

Deepbox is designed with security in mind:

- ✅ **Zero Runtime Dependencies**: No supply chain vulnerabilities from third-party packages
- ✅ **Type Safety**: Strict TypeScript prevents common bugs and vulnerabilities
- ✅ **Input Validation**: Comprehensive validation of all inputs
- ✅ **No Eval**: No use of `eval()` or other unsafe code execution
- ✅ **Memory Safety**: TypedArrays and proper bounds checking
- ✅ **No External Network Calls**: Library operates entirely locally

## Known Issues

Currently, there are no known security vulnerabilities in Deepbox 0.2.0.

## Security Updates

Security updates will be released as patch versions (e.g., 0.2.1, 0.2.2) and documented in the [CHANGELOG.md](CHANGELOG.md).

## Scope

This security policy applies to:

- The Deepbox npm package
- All code in the main repository
- Official examples and documentation

This policy does NOT apply to:

- Third-party packages that use Deepbox
- Unofficial forks or modifications
- User applications built with Deepbox

## Contact

For security concerns, please contact the maintainer:

- **Security Email**: [hi@jehaad.com](mailto:hi@jehaad.com)
- **Author**: Jehaad Aljohani
- **Website**: [https://deepbox.dev](https://deepbox.dev)
- **Repository**: [https://github.com/jehaad1/Deepbox](https://github.com/jehaad1/Deepbox)
- **Issues**: [https://github.com/jehaad1/Deepbox/issues](https://github.com/jehaad1/Deepbox/issues) (for non-security issues only)

## Acknowledgments

We thank the security researchers and community members who help keep Deepbox secure.

---

**Last Updated**: February 14, 2026
