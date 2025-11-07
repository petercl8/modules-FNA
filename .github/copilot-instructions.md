# Project general coding guidelines

## Code Style
- Avoid matlotlib unless necessary for specific customizations. Prefer Pandas built-in plotting functions for simplicity.
- Prefer modern Python (3.6+) features like f-strings and type hints

## Naming Conventions
- Use snake_case for function names and variable names
- For long variable names, combine snake_case and camelCase for readability (e.g., long_variableName)
- Use PascalCase for class names, interfaces, and type aliases
- Prefix private class members with underscore (_)
- Use ALL_CAPS for constants

## Code Quality
- Use meaningful variable and function names that clearly describe their purpose
- Include helpful comments for complex logic
- Avoid deeply nested code structures; refactor into smaller functions when necessary
- Avoid code duplication by creating reusable functions or components
- For complex numpy indexing, use intermediate variables with descriptive names
- Add simple error handling for user inputs and API calls