module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat',     // New feature
        'fix',      // Bug fix
        'docs',     // Documentation changes
        'chore',    // Maintenance tasks
        'refactor', // Code refactoring
        'test',     // Adding or updating tests
        'perf',     // Performance improvements
        'ci',       // CI/CD changes
        'style',    // Code style changes (formatting)
        'revert',   // Revert previous commit
        'build',    // Build system changes
      ],
    ],
    'subject-case': [2, 'always', 'lower-case'],
    'subject-empty': [2, 'never'],
    'subject-full-stop': [2, 'never', '.'],
    'header-max-length': [2, 'always', 100],
  },
};
