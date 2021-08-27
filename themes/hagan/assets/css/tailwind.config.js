module.exports = {
    theme: {
      extend: {}
    },
    darkMode: 'media',
    variants: {},
    plugins: [
      require('@tailwindcss/forms')({
        strategy: 'class',
      }),
    ],
    purge: {
      enabled: true,
      content: ['./src/**/*.html'],
    },
  }