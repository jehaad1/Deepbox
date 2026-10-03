# Security Policy

## Supported Versions

Security fixes are issued for supported release lines only.

| Version | Supported |
| --- | --- |
| `1.5.x` | Yes |
| `1.0.x` | Yes |
| `0.2.x` | No |
| `< 0.2.0` | No |

## Reporting a Vulnerability

Do not open public GitHub issues for security reports.

Report vulnerabilities to:

- Email: [hi@jehaad.com](mailto:hi@jehaad.com)

Please include:

- a short description of the issue
- affected Deepbox version, commit, or tag
- operating system and Node.js version
- reproduction steps
- proof of concept if available
- expected impact and likely attack surface

## Response Process

- Initial acknowledgment target: within 48 hours
- Triage target: within 7 days
- Fix and disclosure timing: coordinated case by case based on severity and reproducibility

If the report is valid, fixes are generally shipped in the next appropriate patch release on the supported line.

## Scope

This policy applies to:

- the published `deepbox` npm package
- code in the main Deepbox repository
- official examples and documentation maintained in this repo

This policy does not apply to:

- unofficial forks or downstream wrappers
- third-party packages that depend on Deepbox
- vulnerabilities in user applications that merely import Deepbox
- issues caused only by unsupported runtime environments

## Network-Aware Surfaces

Most Deepbox modules are local-only numerical code, but some dataset helpers can access external resources. Treat remote data and credentials as untrusted input.

Examples include:

- `fetchCSVDataset()`
- image dataset fetchers such as MNIST and CIFAR helpers
- text dataset fetchers such as IMDB and 20 Newsgroups helpers
- Kaggle integration helpers in `deepbox/datasets`

When using these features:

- validate downloaded content before using it in production pipelines
- scope and rotate Kaggle credentials appropriately
- avoid writing fetched artifacts to sensitive locations

## Secure Usage Guidance

- Keep Node.js and npm current on supported releases.
- Pin Deepbox to a supported version in production systems.
- Validate shapes, dtypes, paths, URLs, and untrusted payloads before passing them into your own application logic.
- Prefer Deepbox custom errors and strict TypeScript checks when building extensions around the library.
- Review file-system interactions around serialization and figure export when paths are influenced by user input.

## Disclosure

Security fixes are documented in:

- [CHANGELOG.md](CHANGELOG.md)
- GitHub releases
- the npm package release history

Public disclosure happens after a fix is available or a coordinated disclosure window has been agreed.

## Contact

- Maintainer: Jehaad Aljohani
- Security email: [hi@jehaad.com](mailto:hi@jehaad.com)
- Repository: [https://github.com/jehaad1/Deepbox](https://github.com/jehaad1/Deepbox)
- Website: [https://deepbox.dev](https://deepbox.dev)

Last updated: October 3, 2026
